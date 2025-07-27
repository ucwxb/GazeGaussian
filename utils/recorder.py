from tensorboardX import SummaryWriter
import torch
import logging
import os
import numpy as np
import cv2
import time
import trimesh
import imageio
from plyfile import PlyData, PlyElement
from utils.logging import config_logging

def get_time_prefix():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

class MeshHeadTrainRecorder():
    def __init__(self, cfg):
        self.name = cfg.name

        os.makedirs(cfg.logdir, exist_ok=True)
        self.logdir = os.path.join(cfg.logdir, self.name + "_" + get_time_prefix())
        os.makedirs(self.logdir, exist_ok=True)
        self.logger = SummaryWriter(self.logdir)

        log_file = os.path.join(self.logdir, "log.txt")
        config_logging(verbose=cfg.verbose, log_file=log_file, append=cfg.resume)
        logging.getLogger('gaze').info(cfg)

        self.checkpoint_path = os.path.join(self.logdir, cfg.checkpoint_path)
        self.result_path = os.path.join(self.logdir, cfg.result_path)
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])
        self.logger.add_scalar('loss_sil', log_data['loss_sil'], log_data['iter'])
        self.logger.add_scalar('loss_def', log_data['loss_def'], log_data['iter'])
        self.logger.add_scalar('loss_offset', log_data['loss_offset'], log_data['iter'])
        self.logger.add_scalar('loss_lmk', log_data['loss_lmk'], log_data['iter'])
        self.logger.add_scalar('loss_lap', log_data['loss_lap'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:

            state_dict = {
                "meshhead": log_data['meshhead'].state_dict(),
                "optimizer": log_data['optimizer'].state_dict(),
            }
            torch.save(state_dict, '%s/meshhead_epoch_%d.pth' % (self.checkpoint_path, log_data['epoch']))

        if log_data['iter'] % self.show_freq == 0:
            image = log_data['data']['image'][0, 0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]


            render_image = log_data['data']['render_images'][0, 0, :, :, 0:3]
            render_image = render_image.detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            render_normal = log_data['data']['render_normals'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
            render_normal = (render_normal * 255).astype(np.uint8)[:,:,::-1]

            render_depths = log_data['data']['render_depths'][0, 0].detach().cpu().numpy()
            valid_depth = render_depths > 1e-3
            render_depths[valid_depth] = (render_depths[valid_depth].max() - render_depths[valid_depth]) / (render_depths[valid_depth].max() - render_depths[valid_depth].min()) * 255.0
            render_depths = render_depths.astype(np.uint8)

            render_depths = cv2.merge([render_depths, render_depths, render_depths])

            render_image = cv2.resize(render_image, (render_image.shape[0], render_image.shape[1]))
            render_normal = cv2.resize(render_normal, (render_image.shape[0], render_image.shape[1]))
            result = np.hstack((image, render_image, render_normal, render_depths))
            cv2.imwrite('%s/%06d.jpg' % (self.result_path, log_data['iter']), result)


    def print_info(self, info):
        logging.getLogger('gaze').info(info)


class GazeGaussianTrainRecorder():
    def __init__(self, cfg):
        self.name = cfg.name

        os.makedirs(cfg.logdir, exist_ok=True)
        self.logdir = os.path.join(cfg.logdir, self.name + "_" + get_time_prefix())
        os.makedirs(self.logdir, exist_ok=True)
        self.logger = SummaryWriter(self.logdir)
        
        log_file = os.path.join(self.logdir, "log.txt")
        config_logging(verbose=cfg.verbose, log_file=log_file, append=cfg.resume)
        logging.getLogger('gaze').info(cfg)
        
        self.checkpoint_path = os.path.join(self.logdir, cfg.checkpoint_path)
        self.result_path = os.path.join(self.logdir, cfg.result_path)
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    
    def log(self, log_data):
        loss_dict = log_data['loss_dict']
        for key in loss_dict:
            self.logger.add_scalar(key, loss_dict[key], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:

            state_dict = {
                "gazegaussian": log_data['gazegaussian'].state_dict(),
                "iden_offset": log_data['iden_offset'],
                "expr_offset": log_data['expr_offset'],
                "delta_EulurAngles": log_data['delta_EulurAngles'],
                "delta_Tvecs": log_data['delta_Tvecs'],
            }

            torch.save(state_dict, '%s/gazegaussian_epoch_%d.pth' % (self.checkpoint_path, log_data['epoch']))

        if log_data['iter'] % self.show_freq == 0:

            head_mask =  log_data['data']['face_mask'] >= 0.5
            face_mask = torch.logical_and(log_data['data']['face_mask'] >= 0.5, torch.logical_and(log_data['data']['left_eye_mask'] < 0.5, log_data['data']['right_eye_mask'] < 0.5))
            eyes_mask = torch.logical_or(log_data['data']['left_eye_mask'] >= 0.5, log_data['data']['right_eye_mask'] >= 0.5)
            nonhead_mask = log_data['data']['face_mask'] < 0.5
            head_mask = head_mask.expand(-1, 3, -1, -1).float()
            face_mask = face_mask.expand(-1, 3, -1, -1).float()
            eyes_mask = eyes_mask.expand(-1, 3, -1, -1).float()
            log_data['data']['total_render_dict']['merge_img_face'] = log_data['data']['total_render_dict']['merge_img'] * face_mask +  1 - face_mask
            log_data['data']['total_render_dict']['merge_img_eyes'] = log_data['data']['total_render_dict']['merge_img'] * eyes_mask +  1 - eyes_mask
            log_data['data']['total_render_dict']['merge_img'] = log_data['data']['total_render_dict']['merge_img'] * head_mask +  1 - head_mask
            log_data['data']['image'] = log_data['data']['image'] * head_mask +  1 - head_mask


            image = log_data['data']['image'][0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            merge_img_face = log_data['data']['total_render_dict']['merge_img_face'][0].permute(1, 2, 0).detach().cpu().numpy()
            merge_img_face = (merge_img_face * 255).astype(np.uint8)[:,:,::-1]

            merge_img_eyes = log_data['data']['total_render_dict']['merge_img_eyes'][0].permute(1, 2, 0).detach().cpu().numpy()
            merge_img_eyes = (merge_img_eyes * 255).astype(np.uint8)[:,:,::-1]
        
            merge_img = log_data['data']['total_render_dict']['merge_img'][0].permute(1, 2, 0).detach().cpu().numpy()
            merge_img = (merge_img * 255).astype(np.uint8)[:,:,::-1]

            img_stack = [image, merge_img, merge_img_face, merge_img_eyes]

            if 'merge_img_pro' in log_data['data']['total_render_dict']:
                merge_img_pro = log_data['data']['total_render_dict']['merge_img_pro'][0].permute(1, 2, 0).detach().cpu().numpy()
                merge_img_pro = (merge_img_pro * 255).astype(np.uint8)[:,:,::-1]
                img_stack.append(merge_img_pro)
            
            if 'merge_img_eyes_pro' in log_data['data']['total_render_dict']:
                merge_img_eyes_pro = log_data['data']['total_render_dict']['merge_img_eyes_pro'][0].permute(1, 2, 0).detach().cpu().numpy()
                merge_img_eyes_pro = (merge_img_eyes_pro * 255).astype(np.uint8)[:,:,::-1]
                img_stack.append(merge_img_eyes_pro)

            if 'merge_img_face_pro' in log_data['data']['total_render_dict']:
                merge_img_face_pro = log_data['data']['total_render_dict']['merge_img_face_pro'][0].permute(1, 2, 0).detach().cpu().numpy()
                merge_img_face_pro = (merge_img_face_pro * 255).astype(np.uint8)[:,:,::-1]
                img_stack.append(merge_img_face_pro)
            
            result = np.hstack(img_stack)
            cv2.imwrite('%s/%06d.jpg' % (self.result_path, log_data['iter']), result)


            # eye_color = log_data['data']['gaussians_eye']['color'][0][:,:3].detach().cpu().numpy()
            # eye_color = (eye_color - eye_color.min()) / (eye_color.max() - eye_color.min())
            # eye_opacity = log_data['data']['gaussians_eye']['opacity'][0][:,:1].detach().cpu().numpy()

            # eye_xyz = log_data['data']['gaussians_eye']['xyz'][0].detach().cpu().numpy()

            # eye_xyz = np.concatenate([eye_xyz, eye_color, eye_opacity], axis=1)


            # eye_scales = log_data['data']['gaussians_eye']['scales'][0].detach().cpu().numpy()
            # eye_rotations = log_data['data']['gaussians_eye']['rotation'][0].detach().cpu().numpy()
            # eye_data = {
            #     'vertices': eye_xyz[:,:3],
            #     'colors': eye_xyz[:,3:6],
            #     'opacities': eye_xyz[:,6],
            # }

            # np.save('%s/eye_%06d.npy' % (self.result_path, log_data['iter']), eye_data)


            # face_color = log_data['data']['gaussians_face']['color'][0][:,:3].detach().cpu().numpy()
            # face_color = (face_color - face_color.min()) / (face_color.max() - face_color.min())
            # face_opacity = log_data['data']['gaussians_face']['opacity'][0][:,:1].detach().cpu().numpy()
            # face_scales = log_data['data']['gaussians_face']['scales'][0].detach().cpu().numpy()
            # face_rotations = log_data['data']['gaussians_face']['rotation'][0].detach().cpu().numpy()

            # face_xyz = log_data['data']['gaussians_face']['xyz'][0].detach().cpu().numpy()
            # face_xyz = np.concatenate([face_xyz, face_color, face_opacity], axis=1)


            # np.save('%s/face_%06d.npy' % (self.result_path, log_data['iter']), {
            #     'vertices': face_xyz[:,:3],
            #     'colors': face_xyz[:,3:6],
            #     'opacities': face_xyz[:,6],
            # })


    def print_info(self, info):
        logging.getLogger('gaze').info(info)
