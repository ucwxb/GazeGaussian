import json
import math
import time
import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.transforms import so3_log_map, so3_exp_map


class RenderUtils(object):
    def __init__(self, view_num, device, opt) -> None:
        super().__init__()
        self.view_num = view_num
        self.device = device
        self.opt = opt
        self.build_base_info()
        self.build_cam_info()

    def build_base_info(self):
        mini_h = self.opt.featmap_size
        mini_w = self.opt.featmap_size

        indexs = torch.arange(mini_h * mini_w)
        x_coor = (indexs % mini_w).view(-1)
        y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)

        xy = torch.stack([x_coor, y_coor], dim=0).float()
        uv = torch.stack(
            [x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1
        )

        self.ray_xy = xy.unsqueeze(0).to(self.device)
        self.ray_uv = uv.unsqueeze(0).to(self.device)

        with open("configs/config_files/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        temp_inv_inmat = torch.as_tensor(temp_dict["inv_inmat"])
        temp_inv_inmat[:2, :2] /= self.opt.featmap_size / 32.0
        self.inv_inmat = temp_inv_inmat.view(1, 3, 3).to(self.device)

    def build_cam_info(self):
        tv_z = 0.5 + 11.5
        tv_x = 5.3

        center_ = np.array([0, 0.0, 0.0]).reshape(3)
        temp_center = np.array([0.0, 0.0, tv_z]).reshape(3)
        temp_cam_center = np.array([[tv_x, 0.0, tv_z]]).reshape(3)

        radius_ = math.sqrt(
            np.sum((temp_cam_center - center_) ** 2)
            - np.sum((temp_center - center_) ** 2)
        )
        temp_d2 = np.array([[0.0, -1.0, 0.0]]).reshape(3)

        cam_info_list = []

        angles = np.linspace(0, 360.0, self.view_num)
        for angle in angles:
            theta_ = angle / 180.0 * 3.1415926535
            x_ = math.cos(theta_) * radius_
            y_ = math.sin(theta_) * radius_

            temp_vp = np.array([x_, y_, tv_z]).reshape(3)
            d_1 = (center_ - temp_vp).reshape(3)

            d_2 = np.cross(temp_d2, d_1)
            d_3 = np.cross(d_1, d_2)

            d_1 = d_1 / np.linalg.norm(d_1)
            d_2 = d_2 / np.linalg.norm(d_2)
            d_3 = d_3 / np.linalg.norm(d_3)

            rmat = np.zeros((3, 3), dtype=np.float32)
            rmat[:, 0] = d_2
            rmat[:, 1] = d_3
            rmat[:, 2] = d_1
            rmat = torch.from_numpy(rmat).view(1, 3, 3).to(self.device)
            tvec = torch.from_numpy(temp_vp).view(1, 3, 1).float().to(self.device)

            cam_info = {
                "batch_Rmats": rmat,
                "batch_Tvecs": tvec,
                "batch_inv_inmats": self.inv_inmat,
            }
            cam_info_list.append(cam_info)

        base_rmat = torch.eye(3).float().view(1, 3, 3).to(self.device)
        base_rmat[0, 1:, :] *= -1
        base_tvec = torch.zeros(3).float().view(1, 3, 1).float().to(self.device)
        base_tvec[0, 2, 0] = tv_z

        self.base_cam_info = {
            "batch_Rmats": base_rmat,
            "batch_Tvecs": base_tvec,
            "batch_inv_inmats": self.inv_inmat,
        }

        self.cam_info_list = cam_info_list

    def render_dual_views(self, net, data, horizontal, vertical, rotate_mat=[1., 1., 1.]):

        data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
            [horizontal, vertical]
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            render_img_ori_pose = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
        render_img_ori_pose = (render_img_ori_pose[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


        pose_ori = data["nl3dmm_para_dict"]["pose"][:, :3]  # (B, 3)
        R_ori = so3_exp_map(pose_ori)  # (B, 3, 3)
        mirror_mat = torch.diag(torch.tensor(rotate_mat, device=R_ori.device)).view(1, 3, 3)
        R_mirror = torch.bmm(torch.bmm(mirror_mat, R_ori), mirror_mat)
        pose_mirror = so3_log_map(R_mirror)


        data["nl3dmm_para_dict"]["pose"][:, :3] = pose_mirror
        with torch.no_grad():
            render_img_mirror_pose = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
        render_img_mirror_pose = (render_img_mirror_pose[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        concat_img = np.concatenate([render_img_ori_pose, render_img_mirror_pose], axis=1)
        return concat_img

    def render_novel_views(self, net, data, move_gaze=True):
        res_img_list = []

        horizontal = [
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.2,
            -0.2,
            -0.2,
            -0.1,
            -0.1,
            -0.1,
            0.0,
            0.1,
            0.1,
            0.1,
            0.2,
            0.2,
            0.2,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.2,
            0.2,
            0.2,
            0.1,
            0.0,
            -0.1,
            -0.2,
            -0.2,
            -0.2,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
            -0.3,
        ]
        vertical = [
            0.0,
            -0.1,
            -0.2,
            -0.2,
            -0.3,
            -0.3,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.3,
            -0.3,
            -0.2,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.2,
            0.3,
            0.3,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.3,
            0.3,
            0.2,
            0.2,
            0.1,
            0.0,
        ]

        time_list = []
        for i in range(self.view_num):
            R_novel = self.cam_info_list[i]["batch_Rmats"]
            R_novel = torch.bmm(R_novel, self.base_cam_info["batch_Rmats"]).transpose(1, 2)
            pose_novel = so3_log_map(R_novel)
            data["nl3dmm_para_dict"]["pose"][:, :3] = pose_novel



            if move_gaze:
                data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
                    [horizontal[i], vertical[i]]
                ).unsqueeze(0).to(self.device)
            else:
                data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor([0.0 , -0.5]).unsqueeze(0).to(self.device)
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    s_time = time.time()
                    merge_img_pro = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
                    e_time = time.time()
                    time_list.append(e_time - s_time)
            merge_img_pro = (merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            res_img_list.append(merge_img_pro)

        fps = 1.0 / np.mean(time_list)
        if move_gaze:
            print("Generate Novel Views with Gaze at {:.5f} fps".format(fps))
        else:
            print("Generate Novel Views without Gaze at {:.5f} fps".format(fps))

        return res_img_list

    def render_novel_views_gaze(self, net, data):
        horizontal = [-20, 20]
        vertical = [-50, 50]
        range_x = 4
        range_y = 10

        res_img_list = []

        i = horizontal[0]

        time_list = []

        for j in range(vertical[0], vertical[1]+1, range_y):
            data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
                [i/100.0 , j/100.0 ]
            ).unsqueeze(0).to(self.device)
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    s_time = time.time()
                    merge_img_pro = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
                    e_time = time.time()
                    time_list.append(e_time - s_time)

            merge_img_pro = (
                merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            res_img_list.append(merge_img_pro)

        i = vertical[1]
        for j in range(horizontal[0], horizontal[1]+1, range_x):
            data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
                [j/100.0 , i/100.0 ]
            ).unsqueeze(0).to(self.device)
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    s_time = time.time()
                    merge_img_pro = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
                    e_time = time.time()
                    time_list.append(e_time - s_time)
                
            merge_img_pro = (
                merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            res_img_list.append(merge_img_pro)

        i = horizontal[1]
        for j in range(vertical[1], vertical[0]+1, -range_y):
            data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
                [i/100.0 , j/100.0 ]
            ).unsqueeze(0).to(self.device)
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    s_time = time.time()
                    merge_img_pro = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
                    e_time = time.time()
                    time_list.append(e_time - s_time)

            merge_img_pro = (
                merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            res_img_list.append(merge_img_pro)

        i = vertical[0]
        for j in range(horizontal[1], horizontal[0]+1, -range_x):
            data["nl3dmm_para_dict"]["pitchyaw"] = torch.FloatTensor(
                [j/100.0 , i/100.0 ]
            ).unsqueeze(0).to(self.device)
            with torch.set_grad_enabled(False):
                with torch.no_grad():
                    s_time = time.time()
                    merge_img_pro = net(data, full_pipe=False)['total_render_dict']["merge_img_pro"]
                    e_time = time.time()
                    time_list.append(e_time - s_time)

            merge_img_pro = (
                merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            res_img_list.append(merge_img_pro)

        print("Generate Novel Gaze Views at {:.5f} fps".format(1.0 / np.mean(time_list)))
        return res_img_list

    def render_morphing_res(self, net, data_1, data_2, nums=10):
        res_img_list = []

        data_1_shape_code = data_1['nl3dmm_para_dict']['shape_code'].clone()
        data_2_shape_code = data_2['nl3dmm_para_dict']['shape_code'].clone()

        loop_bar = tqdm(range(nums), leave=True)
        for i in loop_bar:
            loop_bar.set_description("Generate Morphing Res")
            tv = 1.0 - (i / (nums - 1))
            shape_code_new = data_1_shape_code * tv + data_2_shape_code * (
                1 - tv
            )
            data_1['nl3dmm_para_dict']['shape_code'] = shape_code_new

            with torch.set_grad_enabled(False):
                merge_img_pro = net(data_1, full_pipe=False)['total_render_dict']["merge_img_pro"]

            coarse_fg_rgb = (
                merge_img_pro[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            res_img_list.append(coarse_fg_rgb)

        return res_img_list
