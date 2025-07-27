import torch
import torch.nn.functional as F
from tqdm import tqdm
import kaolin
import os
from pytorch3d.transforms import so3_exponential_map
import logging
from models.mesh_head import MeshHeadModule
from models.camera_module import CameraModule

logger = logging.getLogger(__name__)

def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


class MeshHeadTrainer():
    def __init__(self, opt, recorder, init_landmarks_3d_neutral):
        self.opt = opt

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meshhead = MeshHeadModule(self.opt, init_landmarks_3d_neutral).to(self.device)
        if os.path.exists(self.opt.load_meshhead_checkpoint):
            self.state_dict = torch.load(self.opt.load_meshhead_checkpoint)
            meshhead.load_state_dict(self.state_dict['meshhead'])
        else:
            meshhead.pre_train_sphere(300, self.device)
        self.meshhead = meshhead

        self.camera = CameraModule()

        self.optimizer = torch.optim.Adam([
                                    {'params' : self.meshhead.landmarks_3d_neutral, 'lr' : self.opt.lr_lmk},
                                    {'params' : self.meshhead.geo_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.shape_color_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.pose_color_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.eye_color_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.shape_deform_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.pose_deform_mlp.parameters(), 'lr' : self.opt.lr_net},
                                    {'params' : self.meshhead.eye_deform_mlp.parameters(), 'lr' : self.opt.lr_net},
                        ])
        if os.path.exists(self.opt.load_meshhead_checkpoint):
            self.optimizer.load_state_dict(self.state_dict['optimizer'])
        self.recorder = recorder

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        for epoch in range(n_epochs):
            train_tq = tqdm(train_data_loader)
            for idx, data in enumerate(train_tq):
                to_cuda = ['image', 'face_mask', 'left_eye_mask', 'right_eye_mask', 'ldms', 'ldms_3d', 'cam_ind']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)
                    if 'image' in data_item or 'mask' in data_item:
                        data[data_item] = data[data_item].unsqueeze(1)
                
                nl3dmm_to_cuda = ['code', 'intrinsics', "extrinsics", 'pitchyaw', 'head_pose', 'pose', 'scale', \
                                  'iden_code', 'expr_code']
                for data_item in nl3dmm_to_cuda:
                    data['nl3dmm_para_dict'][data_item] = data['nl3dmm_para_dict'][data_item].to(device=self.device)
                    if 'intrinsics' in data_item or 'extrinsics' in data_item:
                        data['nl3dmm_para_dict'][data_item] = data['nl3dmm_para_dict'][data_item].unsqueeze(1)

                images = data['image'].permute(0, 1, 3, 4, 2)
                masks = data['face_mask'].permute(0, 1, 3, 4, 2)
                resolution = images.shape[2]

                R = so3_exponential_map(data['nl3dmm_para_dict']['pose'][:, :3])

                T = data['nl3dmm_para_dict']['pose'][:, 3:, None]

                S = data['nl3dmm_para_dict']['scale'][:, :, None]

                landmarks_3d_can = (torch.bmm(R.permute(0,2,1), (data['ldms_3d'].permute(0, 2, 1) - T)) / S).permute(0, 2, 1)

                landmarks_3d_neutral = self.meshhead.get_landmarks()[None].repeat(data['ldms_3d'].shape[0], 1, 1)
                data['landmarks_3d_neutral'] = landmarks_3d_neutral

                data['nl3dmm_para_dict']['shape_code'] = (
                    torch.cat(
                        [
                            data['nl3dmm_para_dict']['expr_code'],
                            data['nl3dmm_para_dict']['iden_code']
                        ],
                        dim=-1,
                    )
                    .type(torch.FloatTensor)
                    .to(self.device)
                )

                deform_data = {
                    'shape_code': data['nl3dmm_para_dict']['shape_code'],
                    'query_pts': landmarks_3d_neutral
                }
                deform_data = self.meshhead.deform(deform_data)
                pred_landmarks_3d_can = deform_data['deformed_pts']
                loss_def = F.mse_loss(pred_landmarks_3d_can, landmarks_3d_can)

                deform_data = self.meshhead.query_sdf(deform_data)
                sdf_landmarks_3d = deform_data['sdf']
                loss_lmk = torch.abs(sdf_landmarks_3d[:, :, 0]).mean()

                data = self.meshhead.reconstruct(data)
                data = self.camera.render(data, resolution)
                render_images = data['render_images']
                render_soft_masks = data['render_soft_masks']
                shape_deform = data['shape_deform']
                pose_deform = data['pose_deform']
                verts_list = data['verts_list']
                faces_list = data['faces_list']

                loss_rgb = F.l1_loss(render_images[:, :, :, :, 0:3], images * masks + 1-masks)
                loss_sil = kaolin.metrics.render.mask_iou((render_soft_masks).reshape(-1, resolution, resolution), (masks).squeeze().reshape(-1, resolution, resolution))
                loss_offset = (shape_deform ** 2).sum(-1).mean() + (pose_deform ** 2).sum(-1).mean()

                loss_lap = 0.0
                for b in range(len(verts_list)):
                    loss_lap += laplace_regularizer_const(verts_list[b], faces_list[b])
                
                loss = loss_rgb * 1e0 + loss_sil * 1e-1 + loss_def * 1e-1 + loss_offset * 1e-2 + loss_lmk * 1e-1 + loss_lap * 1e2

                self.optimizer.zero_grad()
                loss.backward()
                
                # 检查梯度是否有NaN
                def check_grad_nan(model, model_name):
                    has_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"❌ NaN gradient found in {model_name}.{name}")
                            print(f"   Shape: {param.grad.shape}")
                            print(f"   NaN count: {torch.isnan(param.grad).sum()}")
                            has_nan = True
                    if not has_nan:
                        print(f"✅ No NaN gradients in {model_name}")
                    return has_nan
                
                # 检查meshhead和camera的梯度
                meshhead_nan = check_grad_nan(self.meshhead, "meshhead")
                
                # 如果有NaN，可以选择停止训练或跳过这次更新
                if meshhead_nan:
                    print(f"⚠️  NaN gradients detected at epoch {epoch}, iter {idx}")
                    print(f"   Loss: {loss.item():.6f}")
                    
                    # 将NaN梯度置零
                    def zero_nan_gradients(model):
                        for param in model.parameters():
                            if param.grad is not None:
                                # 将NaN梯度置零
                                param.grad[torch.isnan(param.grad)] = 0.0
                                # 或者将整个梯度置零（更安全）
                                # param.grad.zero_()
                    
                    print("🔄 Zeroing NaN gradients...")
                    zero_nan_gradients(self.meshhead)
                    print("✅ NaN gradients zeroed, continuing training...")
                    
                    # breakpoint()  # 注释掉，让训练继续
                    # 可以选择跳过这次更新
                    # continue
                
                # 添加梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.meshhead.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                log = {
                    'data': data,
                    'meshhead' : self.meshhead,
                    'optimizer' : self.optimizer,
                    'loss_rgb' : loss_rgb,
                    'loss_sil' : loss_sil,
                    'loss_def' : loss_def,
                    'loss_offset' : loss_offset,
                    'loss_lmk' : loss_lmk,
                    'loss_lap' : loss_lap,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(train_data_loader)
                }

                train_tq.set_description(f'Epoch {epoch}, Iter {idx}, Loss {loss:.4f} RGB {loss_rgb:.4f} Sil {loss_sil:.4f} Def {loss_def:.4f} Offset {loss_offset:.4f} LMK {loss_lmk:.4f} Lap {loss_lap:.4f}')
                self.recorder.log(log)
        
        self.recorder.print_info('Training finished.')
