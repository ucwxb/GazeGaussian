import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from configs.gazegaussian_options import BaseOptions
from models.neural_renderer import NeuralRenderer
from models.gaussian_model import GaussianModel
from utils.model_utils import (CalcRayColor, Embedder, FineSample,
                               GenSamplePoints, rotate)
from models.camera_module import CameraModule

class GazeGaussianNet(nn.Module):
    def __init__(self, opt: BaseOptions,
                    xyz=None,
                    feature=None, 
                    landmarks_3d_neutral=None,
                    load_state_dict=None,
                    add_mouth_points=False
                ) -> None:
        super().__init__()
        self.opt = opt

        self.featmap_size = opt.featmap_size
        self.featmap_nc = opt.featmap_nc
        self.pred_img_size = opt.pred_img_size
        self.bg_type = opt.bg_type
        self.eye_lr_mult = opt.eye_lr_mult

        self.camera = CameraModule(self.bg_type, self.featmap_nc)
        if load_state_dict is not None:
            fg_CD_predictor_face_state_dict = {}
            fg_CD_predictor_eyes_state_dict = {}
            neural_render_state_dict = {}
            for key, value in load_state_dict.items():
                if 'fg_CD_predictor_face' in key:
                    fg_CD_predictor_face_state_dict[key.replace('fg_CD_predictor_face.', '')] = value
                
                if 'fg_CD_predictor_eyes' in key:
                    fg_CD_predictor_eyes_state_dict[key.replace('fg_CD_predictor_eyes.', '')] = value
                
                if 'neural_render' in key:
                    neural_render_state_dict[key.replace('neural_render.', '')] = value

            self.fg_CD_predictor_eyes = GaussianModel(
                self.opt,
                fg_CD_predictor_eyes_state_dict['xyz'],
                fg_CD_predictor_eyes_state_dict['feature'],
                fg_CD_predictor_eyes_state_dict['landmarks_3d_neutral'],
                add_mouth_points=False,
                is_eye=True,
                load_state_dict=True,
            )
            self.fg_CD_predictor_eyes.load_state_dict(fg_CD_predictor_eyes_state_dict)

            self.fg_CD_predictor_face = GaussianModel(
                self.opt,
                fg_CD_predictor_face_state_dict['xyz'],
                fg_CD_predictor_face_state_dict['feature'],
                fg_CD_predictor_face_state_dict['landmarks_3d_neutral'],
                add_mouth_points=False,
                load_state_dict=True,
            )
            self.fg_CD_predictor_face.load_state_dict(fg_CD_predictor_face_state_dict)
        else:
            left_eye_indices = list(range(36, 42))
            right_eye_indices = list(range(42, 48))

            def get_threshold_mask(dist, threshold=0.005):
                min_dist = dist.min()
                max_dist = dist.max()
                thr = min_dist + threshold * (max_dist - min_dist)
                mask = dist <= thr
                return mask
            eye_indices = left_eye_indices + right_eye_indices

            dists_left, _, _ = knn_points(xyz.unsqueeze(0), landmarks_3d_neutral[left_eye_indices, :].unsqueeze(0))
            dists_right, _, _ = knn_points(xyz.unsqueeze(0), landmarks_3d_neutral[right_eye_indices, :].unsqueeze(0))
            mask_left = get_threshold_mask(dists_left.squeeze())
            mask_right = get_threshold_mask(dists_right.squeeze())
            mask = mask_left | mask_right

            self.fg_CD_predictor_eyes = GaussianModel(
                self.opt,
                xyz[mask],
                feature[mask],
                landmarks_3d_neutral[eye_indices, :],
                add_mouth_points=False,
                is_eye=True,
            )

            not_eye_indices = [i for i in range(landmarks_3d_neutral.shape[0]) if i not in eye_indices]
            self.fg_CD_predictor_face = GaussianModel(
                self.opt,
                xyz[~mask],
                feature[~mask],
                landmarks_3d_neutral[not_eye_indices, :],
                add_mouth_points=add_mouth_points,
            )

        self.neural_render = NeuralRenderer(
            bg_type=self.opt.bg_type,
            feat_nc=self.featmap_nc,
            out_dim=3,
            final_actvn=True,
            min_feat=32,
            featmap_size=self.featmap_size,
            img_size=self.pred_img_size,
        )

        if load_state_dict is not None:
            self.neural_render.load_state_dict(neural_render_state_dict, strict=False)

    def get_optimization_group(self, lr):
        optimization_group = []
        fg_CD_predictor_face_optimization_group = self.fg_CD_predictor_face.get_optimization_group(lr)
        optimization_group.extend(fg_CD_predictor_face_optimization_group)
        fg_CD_predictor_eyes_optimization_group = self.fg_CD_predictor_eyes.get_optimization_group(lr * self.eye_lr_mult)
        optimization_group.extend(fg_CD_predictor_eyes_optimization_group)
        optimization_group.append(
            {"params": self.neural_render.parameters(), "lr": lr}
        )
        
        return optimization_group

    def initialize_with_meshhead(self, meshhead):
        self.fg_CD_predictor_face.shape_color_mlp.load_state_dict(meshhead.shape_color_mlp.state_dict())
        self.fg_CD_predictor_face.pose_color_mlp.load_state_dict(meshhead.pose_color_mlp.state_dict())

        self.fg_CD_predictor_face.shape_deform_mlp.load_state_dict(meshhead.shape_deform_mlp.state_dict())
        self.fg_CD_predictor_face.pose_deform_mlp.load_state_dict(meshhead.pose_deform_mlp.state_dict())

        self.fg_CD_predictor_eyes.shape_color_mlp.load_state_dict(meshhead.shape_color_mlp.state_dict())
        self.fg_CD_predictor_eyes.pose_color_mlp.load_state_dict(meshhead.pose_color_mlp.state_dict())
        self.fg_CD_predictor_eyes.eye_color_mlp.load_state_dict(meshhead.eye_color_mlp.state_dict())

        self.fg_CD_predictor_eyes.shape_deform_mlp.load_state_dict(meshhead.shape_deform_mlp.state_dict())
        self.fg_CD_predictor_eyes.pose_deform_mlp.load_state_dict(meshhead.pose_deform_mlp.state_dict())
        self.fg_CD_predictor_eyes.eye_deform_mlp.load_state_dict(meshhead.eye_deform_mlp.state_dict())


    def forward(self, data, full_pipe=True):
        data['gaussians_face'] = self.fg_CD_predictor_face.generate(data)
        data['gaussians_eye'] = self.fg_CD_predictor_eyes.generate(data)
        if full_pipe:
            data['face_render_dict'] = self.camera.render_gaussian(data['gaussians_face'], data)
            data['eyes_render_dict'] = self.camera.render_gaussian_eye(data['gaussians_eye'], data)
        data['gaussians_merge'] = {}
        for key in data['gaussians_face'].keys():
            data['gaussians_merge'][key] = torch.concat(
                [data['gaussians_face'][key], data['gaussians_eye'][key]], dim=1
            )
        data['merge_render_dict'] = self.camera.render_gaussian(data['gaussians_merge'], data)

        if self.opt.unet_atten:
            if full_pipe:
                data['face_render_dict']['render_images_pro'] = self.neural_render(data['face_render_dict']['render_images'], data['nl3dmm_para_dict']['shape_code'])
                data['eyes_render_dict']['render_images_pro'] = self.neural_render(data['eyes_render_dict']['render_images'], data['nl3dmm_para_dict']['shape_code'])
            data['merge_render_dict']['render_images_pro'] = self.neural_render(data['merge_render_dict']['render_images'], data['nl3dmm_para_dict']['shape_code'])
        else:
            if full_pipe:
                data['face_render_dict']['render_images_pro'] = self.neural_render(data['face_render_dict']['render_images'], None)
                data['eyes_render_dict']['render_images_pro'] = self.neural_render(data['eyes_render_dict']['render_images'], None)
            data['merge_render_dict']['render_images_pro'] = self.neural_render(data['merge_render_dict']['render_images'], None)

        data['total_render_dict'] = {
            "merge_img": data['merge_render_dict']['render_images'][:,:3],
            "merge_img_pro": data['merge_render_dict']['render_images_pro'][:,:3],
        }
        if full_pipe:
            data['total_render_dict']["merge_img_face"] = data['face_render_dict']['render_images'][:,:3]
            data['total_render_dict']["merge_img_eyes"] = data['eyes_render_dict']['render_images'][:,:3]
            data['total_render_dict']["merge_img_eyes_pro"] = data['eyes_render_dict']['render_images_pro'][:,:3]
            data['total_render_dict']["merge_img_face_pro"] = data['face_render_dict']['render_images_pro'][:,:3]

        if data["down_scale"][0] != 1:
            if full_pipe:
                data['total_render_dict']["merge_img_face"] = F.interpolate(data['total_render_dict']["merge_img_face"], scale_factor=int(data["down_scale"][0].numpy()), mode='bilinear')
                data['total_render_dict']["merge_img_eyes"] = F.interpolate(data['total_render_dict']["merge_img_eyes"], scale_factor=int(data["down_scale"][0].numpy()), mode='bilinear')
            data['total_render_dict']["merge_img"] = F.interpolate(data['total_render_dict']["merge_img"], scale_factor=int(data["down_scale"][0].numpy()), mode='bilinear')
        
        return data
