import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision
from torchvision import transforms
import torch.nn as nn
import lpips

from gaze_estimation.xgaze_baseline_vgg import gaze_network

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trans_eval = transforms.Compose([transforms.Resize(size=(224, 224))])

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def discriminator_loss(real, fake, device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        real_size = list(real.size())
        fake_size = list(fake.size())
        real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
        fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)

        discriminator_loss = (GANLoss(fake, fake_label) + GANLoss(real, real_label)) / 2

        return discriminator_loss

def generator_loss(fake,device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        fake_size = list(fake.size())
        fake_label = torch.zeros(fake_size, dtype=torch.float32).to(device)
        return GANLoss(fake, fake_label)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        """
        Init function for the perceptual loss using VGG16 model.
        :resize: Boolean value that indicates if to resize the images
        """

        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        """
        Forward function that calculate a perceptual loss using the VGG16 model.
        :input: Generated image
        :target: Groundtruth image
        :feature_layers: Which layers to use from the VGG16 model
        :style_layers: Style layers to use
        :return: Returns a perceptual loss between the groundtruth and generated images
        """

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class GazePerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device=None):
        """
        Init function for the gaze perceptual loss using VGG16 model.
        :resize: Boolean value that indicates if to resize the images
        """

        super(GazePerceptualLoss, self).__init__()
        self.model = gaze_network().to(device)
        self.device = device
        path = "configs/config_models/epoch_60_512_ckpt.pth.tar"
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict=state_dict["model_state"])
        self.model.eval()
        self.img_dim = 224
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.cam_matrix = []
        self.cam_distortion = []
        self.face_model_load = np.loadtxt(
            "configs/dataset/eth_xgaze/face_model.txt"
        )  # Generic face model with 3D facial landmarks

        for cam_id in range(18):
            cam_file_name = "configs/dataset/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
            fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
            self.cam_matrix.append(fs.getNode("Camera_Matrix").mat())
            self.cam_distortion.append(fs.getNode("Distortion_Coefficients").mat())
            fs.release()

    def nn_angular_distance(self, a, b):
        sim = F.cosine_similarity(a, b, eps=1e-6)
        sim = F.hardtanh(sim, -1.0, 1.0)
        return torch.acos(sim) * (180 / np.pi)

    def pitchyaw_to_vector(self, pitchyaws):
        sin = torch.sin(pitchyaws)
        cos = torch.cos(pitchyaws)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)

    def gaze_angular_loss(self, y, y_hat):
        y = self.pitchyaw_to_vector(y)
        y_hat = self.pitchyaw_to_vector(y_hat)
        loss = self.nn_angular_distance(y, y_hat)
        return torch.mean(loss)

    def forward(self, input, target, cam_ind, ldms):
        """
        Forward function that calculate a perceptual loss using the VGG16 model.
        :input: Generated image
        :target: Groundtruth image
        :feature_layers: Which layers to use from the VGG16 model
        :style_layers: Style layers to use
        :return: Returns a perceptual loss between the groundtruth and generated images
        """
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = trans_eval(input)
            target = trans_eval(target)

        ldms = ldms[0]
        x = input
        y = target

        gaze_x, head_x = self.model(x)

        gaze_y, head_y = self.model(y)

        angular_loss = self.gaze_angular_loss(gaze_y.detach(), gaze_x)

        head_loss = 0.0

        return angular_loss + head_loss


class GazeNeRFLoss(object):
    def __init__(
        self,
        gaze_loss_importance,
        eye_loss_importance,
        face_loss_importance,
        vgg_importance,
        bg_type="white",
        use_vgg_loss=True,
        use_ssim_loss=True,
        use_l1_loss=False,
        use_angular_loss=False,
        use_patch_gan_loss=False,
        vgg_loss_begin=2,
        angular_loss_begin=2,
        device=None,
    ) -> None:
        """
        Init function for GazeNeRFLoss.
        :eye_loss_importance: Weight to give to the mse loss of the eye region
        :bg_type:  Backgorund color, can be black or white
        :use_vgg_loss: Boolean value that indicate if to use the perceptual loss
        :device: Device to use
        """

        super().__init__()

        self.vgg_importance = vgg_importance
        self.eye_loss_importance = eye_loss_importance
        self.face_loss_importance = face_loss_importance
        self.gaze_loss_importance = gaze_loss_importance
        self.eye_region_importance = 1.0

        if bg_type == "white":
            self.bg_value = 1.0
        elif bg_type == "black":
            self.bg_value = 0.0
        else:
            self.bg_type = None
            print("Error BG type. ")
            exit(0)

        self.use_vgg_loss = use_vgg_loss
        self.use_ssim_loss = use_ssim_loss
        self.use_l1_loss = use_l1_loss
        self.use_angular_loss = use_angular_loss
        self.use_patch_gan_loss = use_patch_gan_loss
        self.vgg_loss_begin = vgg_loss_begin
        self.angular_loss_begin = angular_loss_begin
        if self.use_vgg_loss:
            assert device is not None
            self.device = device
            self.vgg_loss_func = lpips.LPIPS(net='vgg', verbose=False).to(self.device)

        if self.use_angular_loss:
            self.gaze_loss_func = GazePerceptualLoss(resize=True, device=device).to(
                self.device
            )

    @staticmethod
    def calc_cam_loss(delta_cam_info):
        """
        Function that calculate the camera loss, if camera parameters is enabled.
        :delta_cam_info: Camera parameters
        :return: Returns the camera loss
        """

        delta_eulur_loss = torch.mean(
            delta_cam_info["delta_eulur"] * delta_cam_info["delta_eulur"]
        )
        delta_tvec_loss = torch.mean(
            delta_cam_info["delta_tvec"] * delta_cam_info["delta_tvec"]
        )

        return {"delta_eular": delta_eulur_loss, "delta_tvec": delta_tvec_loss}

    def increase_eye_importance(self):
        pass

    def calc_code_loss(self, opt_code_dict):
        """
        Function that calculate the disentanglement loss for all the latent codes.
        :opt_code_dict: Dictionary with all the latent code to optmize
        :return: Returns a dictionary with the losses calculated
        """

        iden_code_opt = opt_code_dict["iden"]
        expr_code_opt = opt_code_dict["expr"]
        iden_loss = torch.mean(iden_code_opt * iden_code_opt)
        expr_loss = torch.mean(expr_code_opt * expr_code_opt)


        res_dict = {
            "iden": iden_loss,
            "expr": expr_loss,
        }

        return res_dict

    def calc_data_loss(
        self,
        data_dict,
        gt_rgb,
        head_mask_c1b,
        face_mask_c1b,
        nonhead_mask_c1b,
        eyes_mask_c1b,
        cam_ind,
        ldms,
        epoch,
        batch_num,
        discriminator,
    ):
        """
        Function that calculate the L2 loss for head, eye and non-head regions. It also calculate the perceptual loss.
        :data_dict: Dictionary with gazenerf prediction
        :gt_rgb: Ground truth image
        :head_mask_c1b: Face mask
        :nonhead_mask_c1b: Non-head region mask
        :eye_mask_c1b: Eye region mask
        :return: Returns a dictionary with the losses calculated
        """

        bg_value = self.bg_value

        res_img_face = data_dict["merge_img_face"]
        res_img_face_pro = data_dict["merge_img_face_pro"]
        res_img_eyes = data_dict["merge_img_eyes"]
        res_img_eyes_pro = data_dict["merge_img_eyes_pro"]
        res_img = data_dict["merge_img"]
        res_img_pro = data_dict["merge_img_pro"]

        head_mask_c3b = head_mask_c1b.expand(-1, 3, -1, -1).float()
        face_mask_c3b = face_mask_c1b.expand(-1, 3, -1, -1).float()
        eyes_mask_c3b = eyes_mask_c1b.expand(-1, 3, -1, -1).float()


        masked_res_img = res_img * head_mask_c3b + 1 - head_mask_c3b
        masked_gt_rgb = gt_rgb * head_mask_c3b + 1 - head_mask_c3b    # 使用掩码保留头部区域


        masked_res_img_eyes = res_img_eyes * eyes_mask_c3b + 1 - eyes_mask_c3b

        masked_gt_rgb_eyes = gt_rgb * eyes_mask_c3b +  1 - eyes_mask_c3b


        masked_res_img_face = res_img_face * face_mask_c3b + 1 - face_mask_c3b
        masked_gt_rgb_face = gt_rgb * face_mask_c3b + 1 - face_mask_c3b

        if self.use_l1_loss:
            head_loss = F.l1_loss(masked_res_img, masked_gt_rgb) 
            head_pro_loss = F.l1_loss(res_img_pro, masked_gt_rgb)
            eyes_loss = F.l1_loss(
                masked_res_img_eyes, masked_gt_rgb_eyes
            )
            eye_pro_loss = F.l1_loss(
                res_img_eyes_pro, masked_gt_rgb_eyes
            )
            face_loss = F.l1_loss(
                masked_res_img_face, masked_gt_rgb_face
            )
            face_pro_loss = F.l1_loss(
                res_img_face_pro, masked_gt_rgb_face
            )
        else:
            head_loss = F.mse_loss(masked_res_img, masked_gt_rgb)
            eyes_loss = F.mse_loss(masked_res_img_eyes, masked_gt_rgb_eyes)
            face_loss = F.mse_loss(masked_res_img_face, masked_gt_rgb_face)

        ssim_head_loss = ssim_eyes_loss = ssim_face_loss = ssim_eyes_pro_loss = ssim_head_pro_loss = ssim_face_pro_loss = 0.0
        if self.use_ssim_loss:

            ssim_head_loss = 0.1 * (1 - ssim(masked_res_img, masked_gt_rgb))
            ssim_head_pro_loss = 0.1 * (1 - ssim(res_img_pro, masked_gt_rgb))
            ssim_eyes_loss = 0.1 * (1 - ssim(masked_res_img_eyes, masked_gt_rgb_eyes))
            ssim_eyes_pro_loss = 0.1 * (1 - ssim(res_img_eyes_pro, masked_gt_rgb_eyes))
            ssim_face_loss = 0.1 * (1 - ssim(masked_res_img_face, masked_gt_rgb_face))
            ssim_face_pro_loss = 0.1 * (1 - ssim(res_img_face_pro, masked_gt_rgb_face))
        
        vgg_face_loss = vgg_eyes_loss = vgg_head_loss = vgg_eyes_pro_loss = vgg_face_pro_loss = vgg_head_pro_loss = 0.0
        if self.use_vgg_loss and epoch >= self.vgg_loss_begin:
            vgg_face_loss = self.vgg_loss_func(masked_res_img_face, masked_gt_rgb_face, normalize=True).mean() * self.vgg_importance
            vgg_face_pro_loss = self.vgg_loss_func(res_img_face_pro, masked_gt_rgb_face, normalize=True).mean() * self.vgg_importance
            vgg_eyes_loss = self.vgg_loss_func(masked_res_img_eyes, masked_gt_rgb_eyes, normalize=True).mean() * self.vgg_importance
            vgg_eyes_pro_loss = self.vgg_loss_func(res_img_eyes_pro, masked_gt_rgb_eyes, normalize=True).mean() * self.vgg_importance
            vgg_head_loss = self.vgg_loss_func(masked_res_img, masked_gt_rgb, normalize=True).mean() * self.vgg_importance
            vgg_head_pro_loss = self.vgg_loss_func(res_img_pro, masked_gt_rgb, normalize=True).mean() * self.vgg_importance

        with torch.no_grad():
            psnr_head = psnr(masked_res_img, masked_gt_rgb)

        res = {
            "eyes": eyes_loss + ssim_eyes_loss + vgg_eyes_loss,
            "face": face_loss + ssim_face_loss + vgg_face_loss,
            "head": head_loss + ssim_head_loss + vgg_head_loss,
            "eyes_p": eye_pro_loss + ssim_eyes_pro_loss + vgg_eyes_pro_loss,
            "face_p": face_pro_loss + ssim_face_pro_loss + vgg_face_pro_loss,
            "head_p": head_pro_loss + ssim_head_pro_loss + vgg_head_pro_loss,
            "psnr": psnr_head,
        }
        res["eyes"] = res["eyes"] * self.eye_loss_importance
        res["eyes_p"] = res["eyes_p"] * self.eye_loss_importance

        res["face"] = res["face"] * self.face_loss_importance
        res["face_p"] = res["face_p"] * self.face_loss_importance

        if self.use_angular_loss and epoch >= self.angular_loss_begin:
            angular = self.gaze_loss_func(
                masked_res_img, masked_gt_rgb, cam_ind, ldms
            )

            res["angular"] = angular * self.gaze_loss_importance

        if self.use_patch_gan_loss:
           warm_up_coeff = torch.tensor(max(min(1.0 / 10.0, (200000 * epoch + batch_num) / 200000), 0.0))
           patch_pred = data_dict["merge_img"]
           fake = discriminator(trans_eval(patch_pred))
           res["gen_patch_gan_loss"] = generator_loss(fake=fake, device = self.device) * warm_up_coeff
        
        return res

    def calc_total_loss(
        self,
        data,
        epoch,
        batch_num,
        discriminator = None
    ):
        """
        Function that calcluate all the losses of gaze nerf.
        :delta_cam_info: Camera parameters
        :opt_code_dict: Dictionary with all the latent code to optmize
        :pred_dict: Dictionary with gazenerf prediction
        :gt_rgb: Ground truth image
        :face_mask_tensor: Face mask
        :eye_mask_tensor: Eye region mask
        :return: Returns a dictionary with the weighted losses calculated
        """

        head_mask =  data['face_mask'] >= 0.5
        face_mask = torch.logical_and(data['face_mask'] >= 0.5, torch.logical_and(data['left_eye_mask'] < 0.5, data['right_eye_mask'] < 0.5))
        eyes_mask = torch.logical_or(data['left_eye_mask'] >= 0.5, data['right_eye_mask'] >= 0.5)
        nonhead_mask = data['face_mask'] < 0.5

        loss_dict = self.calc_data_loss(
            data['total_render_dict'],
            data['image'],
            head_mask,
            face_mask,
            nonhead_mask,
            eyes_mask,
            data['cam_ind'],
            data['ldms'],
            epoch,
            batch_num,
            discriminator,
        )

        total_loss = 0.0
        for k in loss_dict:
            if k == "psnr":
                continue
            total_loss += loss_dict[k]


        if data["opt_cam_dict"] is not None:
            loss_dict.update(self.calc_cam_loss(data["opt_cam_dict"]))
            total_loss += (
                0.001 * loss_dict["delta_eular"] + 0.001 * loss_dict["delta_tvec"]
            )

        loss_dict.update(self.calc_code_loss(data["opt_code_dict"]))
        total_loss += (
            0.001 * loss_dict["iden"]
            + 1.0 * loss_dict["expr"]
        )
        loss_dict["total_loss"] = total_loss
        return loss_dict