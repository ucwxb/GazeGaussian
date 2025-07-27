import logging
import os
import random
import cv2
import imageio
import numpy as np
import torch
from PIL import Image
import trimesh
import torchvision
from torchvision import transforms
from tqdm import tqdm
from configs.gazegaussian_options import BaseOptions
from models.mesh_head import MeshHeadModule

from losses.gazenerf_loss import GazeNeRFLoss
from models.gaze_gaussian import GazeGaussianNet
from utils.logging import (log_all_images, log_losses, log_one_image,
                           log_one_number)
from utils.render_utils import RenderUtils
from utils.trainer_utils import eulurangle2Rmat
from models.discriminator import PatchGAN
from losses.gazenerf_loss import discriminator_loss, generator_loss

logger = logging.getLogger(__name__)

inv_trans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

trans_eval = transforms.Compose([transforms.Resize(size=(224, 224))])


class GazeGaussianTrainer():
    """Trainer code for GazeGaussian"""

    def __init__(
        self,
        opt: BaseOptions, recorder,
    ):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt_cam = opt.opt_cam
        self.lr = opt.lr

        self.vgg_importance = opt.vgg_importance
        self.eye_loss_importance = opt.eye_loss_importance
        self.face_loss_importance = opt.face_loss_importance
        self.gaze_loss_importance = opt.gaze_loss_importance
        self.use_vgg_loss = opt.use_vgg_loss
        self.use_ssim_loss = opt.use_ssim_loss
        self.use_l1_loss = opt.use_l1_loss
        self.use_angular_loss = opt.use_angular_loss
        self.use_patch_gan_loss = opt.use_patch_gan_loss
        self.vgg_loss_begin = opt.vgg_loss_begin
        self.angular_loss_begin = opt.angular_loss_begin
        self.is_gradual_loss = opt.is_gradual_loss

        self.step_decay = opt.step_decay

        self.view_num = opt.view_num
        self.duration = opt.duration
        self.fit_image = opt.fit_image


        self.iden_code_dims = opt.iden_code_dims
        self.expr_code_dims = opt.expr_code_dims

        self.recorder = recorder

        self.build_info()
        self.build_tool_funcs()

    def build_info(self):

        if self.use_patch_gan_loss:
            self.discriminator = PatchGAN(input_nc=3).to(self.device)
        else:
            self.discriminator = None

        if os.path.exists(self.opt.load_gazegaussian_checkpoint):
            self.state_dict = torch.load(self.opt.load_gazegaussian_checkpoint, map_location=self.device)

            net = GazeGaussianNet(self.opt,
                                load_state_dict=self.state_dict['gazegaussian']).to(self.device)

        else:
            meshhead_state_dict = torch.load(self.opt.load_meshhead_checkpoint, map_location=lambda storage, loc: storage)['meshhead']

            from configs.meshhead_options import BaseOptions as MeshHeadOptions
            mesh_head_options = MeshHeadOptions()
            mesh_head_options.is_rotate_eye = self.opt.is_rotate_eye
            meshhead = MeshHeadModule(mesh_head_options, meshhead_state_dict['landmarks_3d_neutral']).to(self.device)

            meshhead.load_state_dict(meshhead_state_dict)
            meshhead.subdivide()
            
            with torch.no_grad():
                data = meshhead.reconstruct_neutral()

            epsilon = 1e-5

            verts_feature = torch.clamp(data['verts_feature'].cpu(), min=-1.0 + epsilon, max=1.0 - epsilon)
            net = GazeGaussianNet(self.opt,
                                xyz=data['verts'].cpu(),
                                feature=torch.atanh(verts_feature), 
                                landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                add_mouth_points=False).to(self.device)
            net.initialize_with_meshhead(meshhead)

            self.recorder.print_info("Gazegaussian initialized with meshhead!")

        self.net = net.to(self.device)
        self.net.train()

    def build_tool_funcs(self):

        self.loss_utils = GazeNeRFLoss(
            gaze_loss_importance=self.gaze_loss_importance,
            eye_loss_importance=self.eye_loss_importance,
            face_loss_importance=self.face_loss_importance,
            vgg_importance=self.vgg_importance,
            use_vgg_loss=self.use_vgg_loss,
            use_ssim_loss=self.use_ssim_loss,
            use_l1_loss=self.use_l1_loss,
            use_angular_loss=self.use_angular_loss,
            use_patch_gan_loss=self.use_patch_gan_loss,
            vgg_loss_begin=self.vgg_loss_begin,
            angular_loss_begin=self.angular_loss_begin,
            device=self.device,
        )
        self.render_utils = RenderUtils(view_num=self.view_num, device=self.device, opt=self.opt)

    def get_optimizer(self, params_group):
        self.optimizer = torch.optim.Adam(params_group)
        self.lr_func = lambda epoch: 0.1 ** (epoch / self.step_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_func
        )

        if self.use_patch_gan_loss:
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-4)


    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""
        self.prepare_optimizer_opt(len(train_data_loader.dataset))

        for i in range(n_epochs):
            self.train_epoch(train_data_loader, i)
            if self.is_gradual_loss:
                self.loss_utils.increase_eye_importance()


    def prepare_data(self, data):
        batch_size = data['image'].shape[0]
        to_cuda = ['image', 'face_mask', 'left_eye_mask', 'right_eye_mask', 'ldms', 'ldms_3d', 'cam_ind', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']
        for data_item in to_cuda:
            data[data_item] = data[data_item].to(device=self.device)
        
        nl3dmm_to_cuda = ['code', 'intrinsics', "extrinsics", 'pitchyaw', 'head_pose', 'pose', 'scale', 'iden_code', 'expr_code']
        for data_item in nl3dmm_to_cuda:
            data['nl3dmm_para_dict'][data_item] = data['nl3dmm_para_dict'][data_item].to(device=self.device)
            if 'intrinsics' in data_item or 'extrinsics' in data_item:
                data['nl3dmm_para_dict'][data_item] = data['nl3dmm_para_dict'][data_item].unsqueeze(0)

        return data

    def build_code_and_cam(self, iter, data, skip_opt_cam=False):
        batch_size = data['image'].shape[0]

        if iter != -1:
            pos_start = iter * batch_size
            pos_end = pos_start + batch_size
            expr_offset = self.expr_offset[pos_start:pos_end]
            iden_offset = self.iden_offset[pos_start:pos_end]
        else:
            expr_offset = torch.zeros((batch_size, self.expr_code_dims), dtype=torch.float32).to(self.device)
            iden_offset = torch.zeros((batch_size, self.iden_code_dims), dtype=torch.float32).to(self.device)


        data['nl3dmm_para_dict']['shape_code'] = (
            torch.cat(
                [
                    data['nl3dmm_para_dict']['expr_code'] + expr_offset,
                    data['nl3dmm_para_dict']['iden_code'] + iden_offset,
                ],
                dim=-1,
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )

        data["opt_code_dict"] = {
            "iden": iden_offset,
            "expr": expr_offset,
        }

        if self.opt_cam and iter != -1 and not skip_opt_cam:
            opt_cam_dict = {
                "delta_eulur": self.delta_EulurAngles[pos_start:pos_end],
                "delta_tvec": self.delta_Tvecs[pos_start:pos_end],
            }

            data['nl3dmm_para_dict']['pose'][:,:3] += self.delta_EulurAngles[pos_start:pos_end]
            data['nl3dmm_para_dict']['pose'][:,3:] += self.delta_Tvecs[pos_start:pos_end].squeeze(-1)

        else:
            opt_cam_dict = None

        data["opt_cam_dict"] = opt_cam_dict

        return data

    def prepare_optimizer_opt(self, train_len):
        
        if self.opt.load_gazegaussian_checkpoint:
            self.iden_offset = self.state_dict['iden_offset'].to(self.device).requires_grad_(True)
            self.expr_offset = self.state_dict['expr_offset'].to(self.device).requires_grad_(True)
            self.delta_EulurAngles = self.state_dict['delta_EulurAngles'].to(self.device)
            self.delta_Tvecs = self.state_dict['delta_Tvecs'].to(self.device)
        else:
            self.iden_offset = torch.zeros(
                (train_len, self.iden_code_dims), dtype=torch.float32
            ).to(self.device).requires_grad_(True)
            self.expr_offset = torch.zeros(
                (train_len, self.expr_code_dims), dtype=torch.float32
            ).to(self.device).requires_grad_(True)


            self.delta_EulurAngles = torch.zeros(
                (train_len, 3), dtype=torch.float32
            ).to(self.device)
            self.delta_Tvecs = torch.zeros(
                (train_len, 3, 1), dtype=torch.float32
            ).to(self.device).requires_grad_(True)

        if self.opt_cam:
            self.delta_EulurAngles = self.delta_EulurAngles.requires_grad_(True)
            self.delta_Tvecs = self.delta_Tvecs.requires_grad_(True)

        init_learn_rate = self.lr


        params_group = [
            {"params": [self.iden_offset], "lr": init_learn_rate * 1.0},
            {"params": [self.expr_offset], "lr": init_learn_rate * 0.1},
        ]

        params_group.extend(self.net.get_optimization_group(init_learn_rate))

        if self.opt_cam:
            params_group += [
                {"params": [self.delta_EulurAngles], "lr": init_learn_rate * 0.1},
                {"params": [self.delta_Tvecs], "lr": init_learn_rate * 0.1},
            ]

        self.get_optimizer(params_group)

    def train_epoch(self, data_loader, epoch):
        """Train for one epoch"""

        self.net.train()

        batch_loop_bar = tqdm(data_loader, desc="Batch Progress")


        for batch_index, data in enumerate(batch_loop_bar):
            data = self.prepare_data(data)

            with torch.set_grad_enabled(True):
                data = self.build_code_and_cam(data["idx_input"][0], data)
                data["iter"] = batch_index + epoch * len(data_loader)
                data = self.net(data)

                batch_loss_dict = self.loss_utils.calc_total_loss(
                    data,
                    epoch = epoch,
                    batch_num = batch_index,
                    discriminator = self.discriminator,
                )

                log_data = {
                    'data': data,
                    'gazegaussian' : self.net,
                    "iden_offset": self.iden_offset,
                    "expr_offset": self.expr_offset,
                    "delta_EulurAngles": self.delta_EulurAngles,
                    "delta_Tvecs": self.delta_Tvecs,
                    'optimizer' : self.optimizer,
                    'loss_dict' : batch_loss_dict,
                    'epoch' : epoch,
                    'iter' : batch_index + epoch * len(data_loader)
                }

                self.recorder.log(log_data)

                tq_str = "Epoch: %d, Batch: %d, Loss: %.4f" % (epoch, batch_index, batch_loss_dict["total_loss"])
                for key in batch_loss_dict.keys():
                    if key == "total_loss":
                        continue
                    tq_str += ", %s: %.4f" % (key, batch_loss_dict[key])
                batch_loop_bar.set_description_str(tq_str)

            self.optimizer.zero_grad()
            if self.opt.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            batch_loss_dict["total_loss"].backward()
            self.optimizer.step()

    def train_single_image(
        self, dataloader, n_epochs, index, is_target=False
    ):
        self.net.train()
        self.prepare_optimizer_opt(len(dataloader.dataset))


        batch_loop_bar = range(0, n_epochs)

        dataloader.dataset.modify_index(index, is_target=is_target)
        for batch_index, data in enumerate(dataloader):
            data = self.prepare_data(data)
            data = self.build_code_and_cam(data["idx_input"][0], data, skip_opt_cam=True)
            break
        dataloader.dataset.modify_index(None, is_target=False)

        for epoch in batch_loop_bar:
            with torch.set_grad_enabled(True):
                data = self.net(data)
                batch_loss_dict = self.loss_utils.calc_total_loss(
                    data,
                    epoch = epoch,
                    batch_num = batch_index,
                    discriminator = self.discriminator,
                )
            self.optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            if self.opt.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()


    @torch.no_grad()
    def predict_single_image(self, data):
        """ "Evaluate the model"""
        self.net.eval()

        data = self.prepare_data(data)
        with torch.set_grad_enabled(False):
            data = self.build_code_and_cam(-1, data)
            data = self.net(data)

        return data['total_render_dict']["merge_img_pro"]

    @torch.no_grad()
    def generate_dual_image(self, data_loader, key, start_frame=0, gaze_estimate_label=None):
        """ "Evaluate the model"""

        self.net.eval()
        data_pre_list = []

        for i, data_pre in enumerate(data_loader):
            if start_frame < 18:
                data_pre = self.prepare_data(data_pre)
                with torch.set_grad_enabled(False):
                    data_pre = self.build_code_and_cam(-1, data_pre)
                    data_pre_list.append(data_pre)
            else:
                break
        
        for l_ind in range(len(gaze_estimate_label)):
            if l_ind == 18:
                break
            data = data_pre_list[int(l_ind % 18)].copy()
            concat_img = self.render_utils.render_dual_views(self.net, data, -gaze_estimate_label['gaze_x'][l_ind], gaze_estimate_label['gaze_y'][l_ind], rotate_mat=[-1., 1., 1.])
            ref_img = cv2.imread(gaze_estimate_label['image_path'][l_ind])
            ref_img = cv2.cvtColor(cv2.resize(ref_img, (512, 512)), cv2.COLOR_BGR2RGB)
            concat_img = np.concatenate([ref_img, concat_img], axis=1)
            cv2.imwrite(f"dual/dual_image_{key}_{l_ind}.png", cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))

    @torch.no_grad()
    def evaluate_single_image(self, data_loader, key, start_frame=0):
        """ "Evaluate the model"""

        self.net.eval()

        for i, data in enumerate(data_loader):
            if start_frame == i:
                data = self.prepare_data(data)
                with torch.set_grad_enabled(False):
                    data = self.build_code_and_cam(-1, data)
                    novel_gaze_head_set = self.render_utils.render_novel_views(
                        self.net, data, move_gaze=True
                    )

                    novel_gaze_set = self.render_utils.render_novel_views_gaze(
                        self.net, data
                    )

                    novel_head_set = self.render_utils.render_novel_views(
                        self.net, data, move_gaze=False
                    )

                    log_data = {
                        "novel_gaze_head_set": novel_gaze_head_set,
                        "novel_gaze_set": novel_gaze_set,
                        "novel_head_set": novel_head_set,
                        "subject": key,
                        "start_frame": start_frame,
                    }
                    self.recorder.visualize(log_data)
                    return

    @torch.no_grad()
    def evaluate_morphing(self, data_loader_1, data_loader_2, key_1, key_2, start_frame_1=0, start_frame_2=0):
        """ "Evaluate the model"""

        self.net.eval()

        for i, data_1 in enumerate(data_loader_1):
            if start_frame_1 == i:
                data_1 = self.prepare_data(data_1)
                break
                
        for j, data_2 in enumerate(data_loader_2):
            if start_frame_2 == j:
                data_2 = self.prepare_data(data_2)
                break
        
        with torch.set_grad_enabled(False):
            data_1 = self.build_code_and_cam(-1, data_1)
            data_2 = self.build_code_and_cam(-1, data_2)

            morphing_res = self.render_utils.render_morphing_res(
                self.net, data_1, data_2
            )

            log_data = {
                "morphing_res": morphing_res,
                "subject_1": key_1,
                "subject_2": key_2,
            }
            self.recorder.visualize_morphing(log_data)
            return

