import torch
from torch import nn
from einops import rearrange
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from simple_knn._C import distCUDA2
from configs.gazegaussian_options import BaseOptions

from models.MLP import MLP
from models.PositionalEmbedding import get_embedder
from utils.model_utils import inverse_sigmoid
from utils.model_utils import rotation_matrix_2d
from utils.sh_utils import eval_sh

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def batch_rotation_matrix_between_vectors(vec1, vec2):
    vec1 = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2 = vec2 / vec2.norm(dim=1, keepdim=True)
    cross_product = torch.cross(vec1, vec2)
    dot_product = torch.bmm(vec1.unsqueeze(1), vec2.unsqueeze(2)).squeeze(1)
    sine_theta = cross_product.norm(dim=1, keepdim=True)
    K = torch.empty(vec1.size(0), 3, 3, device=vec1.device)
    K[:, 0, 1] = -cross_product[:, 2]
    K[:, 0, 2] = cross_product[:, 1]
    K[:, 1, 0] = cross_product[:, 2]
    K[:, 1, 2] = -cross_product[:, 0]
    K[:, 2, 0] = -cross_product[:, 1]
    K[:, 2, 1] = cross_product[:, 0]
    K += torch.eye(3, device=vec1.device).unsqueeze(0) 
    R = torch.eye(3, device=vec1.device).unsqueeze(0) + K + (K @ K) * (1 - dot_product) / (sine_theta ** 2)

    return R

def project_3d_to_2d(landmarks_3d, intrinsics, extrinsics, image_shape):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    landmarks_3d_homogeneous = np.hstack([landmarks_3d, np.ones((landmarks_3d.shape[0], 1))])
    landmarks_camera = np.dot(extrinsics, landmarks_3d_homogeneous.T).T
    x_2d = (landmarks_camera[:, 0] * fx) / landmarks_camera[:, 2] + cx
    y_2d = (landmarks_camera[:, 1] * fy) / landmarks_camera[:, 2] + cy
    x_2d = (x_2d + 1) * image_shape[1] / 2 
    y_2d = (y_2d + 1) * image_shape[0] / 2 

    return np.vstack([x_2d, y_2d]).T


def vector_to_pitchyaw(gaze_direction):
    pitch = torch.atan2(gaze_direction[:, 1], gaze_direction[:, 2])
    yaw = torch.atan2(gaze_direction[:, 0], gaze_direction[:, 2])
    return pitch, yaw

class GaussianModel(nn.Module):
    def __init__(self, opt: BaseOptions, xyz, feature, landmarks_3d_neutral, add_mouth_points=False, is_eye=False, load_state_dict=False):
        super(GaussianModel, self).__init__()
        self.is_eye = is_eye

        if add_mouth_points and opt.num_add_mouth_points > 0:
            mouth_keypoints = landmarks_3d_neutral[48:68]
            mouth_center = torch.mean(mouth_keypoints, dim=0, keepdim=True)
            mouth_center[:, 2] = mouth_keypoints[:, 2].min()
            max_dist = (mouth_keypoints - mouth_center).abs().max(0)[0]
            points_add = (torch.rand([opt.num_add_mouth_points, 3]) - 0.5) * 1.6 * max_dist + mouth_center
        
            xyz = torch.cat([xyz, points_add])
            feature = torch.cat([feature, torch.zeros([opt.num_add_mouth_points, feature.shape[1]])])
            
        self.xyz = nn.Parameter(xyz)

        self.feature = nn.Parameter(feature)
        
        self.register_buffer('landmarks_3d_neutral', landmarks_3d_neutral)


        dist2 = torch.clamp_min(distCUDA2(self.xyz.cuda()), 0.0000001).cpu()
        if self.is_eye:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 1)
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.scales = nn.Parameter(scales)

        rots = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        rots[:, 0] = 1
        self.rotation = nn.Parameter(rots)

        if self.is_eye:
            self.opacity = nn.Parameter(inverse_sigmoid(0.9 * torch.ones((xyz.shape[0], 1))))
        else:
            self.opacity = nn.Parameter(inverse_sigmoid(0.3 * torch.ones((xyz.shape[0], 1))))

        self.shape_color_mlp = MLP(opt.shape_color_mlp, last_op=None)
        self.pose_color_mlp = MLP(opt.pose_color_mlp, last_op=None)
        self.appea_color_mlp = MLP(opt.appea_color_mlp, last_op=None)
        
        if is_eye:
            self.eye_color_mlp = MLP(opt.eye_color_mlp, last_op=None)

        self.shape_attributes_mlp = MLP(opt.shape_attributes_mlp, last_op=None)
        self.pose_attributes_mlp = MLP(opt.pose_attributes_mlp, last_op=None)
        self.appea_attributes_opacity_mlp = MLP(opt.appea_attributes_opacity_mlp, last_op=None)
        if is_eye:
            self.eye_attributes_mlp = MLP(opt.eye_attributes_mlp, last_op=None)
        
        self.shape_deform_mlp = MLP(opt.shape_deform_mlp, last_op=nn.Tanh())
        self.pose_deform_mlp = MLP(opt.pose_deform_mlp, last_op=nn.Tanh())
        if is_eye:
            self.is_rotate_eye = opt.is_rotate_eye
            if opt.is_rotate_eye:
                self.eye_deform_mlp = MLP(opt.eye_deform_mlp_rotate, last_op=nn.functional.normalize)
            else:
                self.eye_deform_mlp = MLP(opt.eye_deform_mlp, last_op=nn.Tanh())

        self.pos_embedding, _ = get_embedder(opt.pos_freq)
        if is_eye:
            self.gaze_embedding, _ = get_embedder(opt.gaze_freq)
            self.eye_offset = nn.Parameter(torch.zeros(3))
        
        self.shape_coeffs_dim = opt.shape_coeffs_dim
        self.dist_threshold_near = opt.dist_threshold_near
        self.dist_threshold_far = opt.dist_threshold_far
        self.deform_scale = opt.deform_scale
        self.attributes_scale = opt.attributes_scale
    
    def generate(self, data):
        B = data['nl3dmm_para_dict']['shape_code'].shape[0]

        xyz = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        feature = torch.tanh(self.feature).unsqueeze(0).repeat(B, 1, 1)

        color = torch.zeros([B, xyz.shape[1], self.shape_color_mlp.dims[-1]], device=xyz.device)
        delta_xyz = torch.zeros_like(xyz, device=xyz.device)
        delta_attributes = torch.zeros([B, xyz.shape[1], self.scales.shape[1] + self.rotation.shape[1] + self.opacity.shape[1]], device=xyz.device)

        try:
            for b in range(B):

                if self.is_eye:
                    
                    feature_eye_controlled = feature[b, ...]
                    eye_color_input = torch.cat([feature_eye_controlled.t(), \
                                                self.gaze_embedding(data['nl3dmm_para_dict']['pitchyaw'][b]).unsqueeze(-1).repeat(1, feature_eye_controlled.shape[0])], 0)[None]
                    eye_color = self.eye_color_mlp(eye_color_input)[0].t()
                    color[b, :, :] += eye_color

                    feature_shape_controlled = feature[b, ...]
                    shape_color_input = torch.cat([feature_shape_controlled.t(), 
                                                data['nl3dmm_para_dict']['shape_code'][b].unsqueeze(-1).repeat(1, feature_shape_controlled.shape[0])], 0)[None]
                    shape_color = self.shape_color_mlp(shape_color_input)[0].t()
                    color[b, :, :] += shape_color

                    eye_attributes_input = eye_color_input
                    eye_attributes = self.eye_attributes_mlp(eye_attributes_input)[0].t()
                    delta_attributes[b, :, :] += eye_attributes

                    xyz_shape_controlled = xyz[b, ...]
                    shape_deform_input = torch.cat([self.pos_embedding(xyz_shape_controlled).t(), 
                                                data['nl3dmm_para_dict']['shape_code'][b].unsqueeze(-1).repeat(1, xyz_shape_controlled.shape[0])], 0)[None]
                    shape_deform = self.shape_deform_mlp(shape_deform_input)[0].t()
                    delta_xyz[b, :, :] += shape_deform
                    
                    if self.is_rotate_eye:
                        left_dists, left_idx, _ = knn_points(xyz, self.landmarks_3d_neutral[:6].unsqueeze(0).repeat(B, 1, 1))
                        right_dists, right_idx, _ = knn_points(xyz, self.landmarks_3d_neutral[6:12].unsqueeze(0).repeat(B, 1, 1))
                        left_eye_weights = torch.clamp((self.dist_threshold_far - left_dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
                        right_eye_weights = torch.clamp((self.dist_threshold_far - right_dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)

                        left_mask = left_eye_weights.squeeze(-1) > 0.00
                        left_eye_deform_input = torch.cat([self.pos_embedding(xyz[left_mask].mean(dim=0, keepdim=True)), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw'])], 1).unsqueeze(-1)
                        left_eye_deform = self.eye_deform_mlp(left_eye_deform_input).permute(0, 2, 1)
                        left_rotation_matrices = quaternion_to_matrix(left_eye_deform).expand(-1, left_mask.sum(), -1, -1)

                        verts_batch_expanded = xyz.unsqueeze(-1)
                        left_eye_deform = torch.matmul(left_rotation_matrices, verts_batch_expanded[left_mask]).squeeze(-1)
                        xyz[left_mask] = left_eye_deform

                        right_mask = right_eye_weights.squeeze(-1) > 0.0
                        right_eye_deform_input = torch.cat([self.pos_embedding(xyz[right_mask].mean(dim=0, keepdim=True)), self.pos_embedding(data['nl3dmm_para_dict']['pitchyaw'])], 1).unsqueeze(-1)
                        right_eye_deform = self.eye_deform_mlp(right_eye_deform_input).permute(0, 2, 1)
                        right_rotation_matrices = quaternion_to_matrix(right_eye_deform).expand(-1, right_mask.sum(), -1, -1)
                        verts_batch_expanded = xyz.unsqueeze(-1)
                        right_eye_deform = torch.matmul(right_rotation_matrices, verts_batch_expanded[right_mask]).squeeze(-1)
                        xyz[right_mask] = right_eye_deform

                    else:
                        xyz_eye_controlled = xyz[b, ...]
                        eye_deform_input = torch.cat([self.pos_embedding(xyz_eye_controlled).t(), 
                                                    self.gaze_embedding(data['nl3dmm_para_dict']['pitchyaw'][b]).unsqueeze(-1).repeat(1, xyz_eye_controlled.shape[0])], 0)[None]
                        eye_deform = self.eye_deform_mlp(eye_deform_input)[0].t()
                        delta_xyz[b, :, :] += eye_deform

                else:

                    dists, _, _ = knn_points(xyz, self.landmarks_3d_neutral.unsqueeze(0).repeat(B, 1, 1))
                    shape_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
                    pose_weights = 1 - shape_weights
                    appea_weights = shape_weights
                    shape_controlled = (dists < self.dist_threshold_far).squeeze(-1)
                    pose_controlled = (dists > self.dist_threshold_near).squeeze(-1)
                    appea_controlled = (dists < self.dist_threshold_far).squeeze(-1)

                    feature_shape_controlled = feature[b, shape_controlled[b], :]
                    shape_color_input = torch.cat([feature_shape_controlled.t(), 
                                                data['nl3dmm_para_dict']['shape_code'][b].unsqueeze(-1).repeat(1, feature_shape_controlled.shape[0])], 0)[None]
                    shape_color = self.shape_color_mlp(shape_color_input)[0].t()
                    color[b, shape_controlled[b], :] += shape_color * shape_weights[b, shape_controlled[b], :]

                    feature_pose_controlled = feature[b, pose_controlled[b], :]
                    pose_color_input = torch.cat([feature_pose_controlled.t(), 
                                                    self.pos_embedding(data['nl3dmm_para_dict']['pose'][b]).unsqueeze(-1).repeat(1, feature_pose_controlled.shape[0])], 0)[None]
                    pose_color = self.pose_color_mlp(pose_color_input)[0].t()
                    color[b, pose_controlled[b], :] += pose_color * pose_weights[b, pose_controlled[b], :]


                    shape_attributes_input = shape_color_input
                    shape_delta_attributes = self.shape_attributes_mlp(shape_attributes_input)[0].t()
                    delta_attributes[b, shape_controlled[b], :] += shape_delta_attributes * shape_weights[b, shape_controlled[b], :]

                    pose_attributes_input = pose_color_input
                    pose_attributes = self.pose_attributes_mlp(pose_attributes_input)[0].t()
                    delta_attributes[b, pose_controlled[b], :] += pose_attributes * pose_weights[b, pose_controlled[b], :]


                    xyz_shape_controlled = xyz[b, shape_controlled[b], :]
                    shape_deform_input = torch.cat([self.pos_embedding(xyz_shape_controlled).t(), 
                                                data['nl3dmm_para_dict']['shape_code'][b].unsqueeze(-1).repeat(1, xyz_shape_controlled.shape[0])], 0)[None]
                    shape_deform = self.shape_deform_mlp(shape_deform_input)[0].t()
                    delta_xyz[b, shape_controlled[b], :] += shape_deform * shape_weights[b, shape_controlled[b], :]

                    xyz_pose_controlled = xyz[b, pose_controlled[b], :]
                    pose_deform_input = torch.cat([self.pos_embedding(xyz_pose_controlled).t(), 
                                                self.pos_embedding(data['nl3dmm_para_dict']['pose'][b]).unsqueeze(-1).repeat(1, xyz_pose_controlled.shape[0])], 0)[None]
                    pose_deform = self.pose_deform_mlp(pose_deform_input)[0].t()
                    delta_xyz[b, pose_controlled[b], :] += pose_deform * pose_weights[b, pose_controlled[b], :]
        except Exception as e:
            print(e)
            breakpoint()

        xyz = xyz + delta_xyz * self.deform_scale
        if self.is_eye:
            delta_scales = delta_attributes[:, :, 0:1].repeat(1, 1, 3)
            scales = self.scales.unsqueeze(0).repeat(B, 1, 3) + delta_scales * self.attributes_scale
            scales = torch.exp(scales)

            delta_rotation = delta_attributes[:, :, 1:5]
            rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation * self.attributes_scale
            rotation = torch.nn.functional.normalize(rotation)

            delta_opacity = delta_attributes[:, :, 5:6]
            opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity * self.attributes_scale
            opacity = torch.sigmoid(opacity)

        else:
            delta_scales = delta_attributes[:, :, 0:3]
            scales = self.scales.unsqueeze(0).repeat(B, 1, 1) + delta_scales * self.attributes_scale
            scales = torch.exp(scales)

            delta_rotation = delta_attributes[:, :, 3:7]
            rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation * self.attributes_scale
            rotation = torch.nn.functional.normalize(rotation)

            delta_opacity = delta_attributes[:, :, 7:8]
            opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity * self.attributes_scale
            opacity = torch.sigmoid(opacity)
            

        R = so3_exponential_map(data['nl3dmm_para_dict']['pose'][:, :3])
        T = data['nl3dmm_para_dict']['pose'][:, None, 3:]
        S = data['nl3dmm_para_dict']['scale'][:, :, None]

        if self.is_eye:
            pass

        xyz = torch.bmm(xyz * S, R.permute(0, 2, 1)) + T

        rotation_matrix = quaternion_to_matrix(rotation)
        rotation_matrix = rearrange(rotation_matrix, 'b n x y -> (b n) x y')
        R = rearrange(R.unsqueeze(1).repeat(1, rotation.shape[1], 1, 1), 'b n x y -> (b n) x y')


        rotation_matrix = rearrange(torch.bmm(R, rotation_matrix), '(b n) x y -> b n x y', b=B)
        rotation = matrix_to_quaternion(rotation_matrix)

        scales = scales * S

        gaussians = {}
        gaussians['xyz'] = xyz
        gaussians['color'] = color
        gaussians['scales'] = scales
        gaussians['rotation'] = rotation
        gaussians['opacity'] = opacity

        return gaussians

    def get_optimization_group(self, lr):
        optimization_group = [
            {'params' : self.xyz, 'lr' : lr * 0.1},
            {'params' : self.feature, 'lr' : lr * 0.1},
            {'params' : self.shape_color_mlp.parameters(), 'lr' : lr},
            {'params' : self.pose_color_mlp.parameters(), 'lr' : lr},
            {'params' : self.appea_color_mlp.parameters(), 'lr' : lr},
            {'params' : self.shape_deform_mlp.parameters(), 'lr' : lr},
            {'params' : self.pose_deform_mlp.parameters(), 'lr' : lr},
            {'params' : self.shape_attributes_mlp.parameters(), 'lr' : lr},
            {'params' : self.pose_attributes_mlp.parameters(), 'lr' : lr},
            {'params' : self.scales, 'lr' : lr * 0.3},
            {'params' : self.rotation, 'lr' : lr * 0.1},
            {'params' : self.opacity, 'lr' : lr}
        ]
        if self.is_eye:
            optimization_group += [
                {'params' : self.eye_color_mlp.parameters(), 'lr' : lr},
                {'params' : self.eye_deform_mlp.parameters(), 'lr' : lr},
                {'params' : self.eye_attributes_mlp.parameters(), 'lr' : lr},
                {'params' : self.eye_offset, 'lr' : lr * 0.1}
            ]
        return optimization_group
    
