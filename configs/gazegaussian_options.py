class BaseOptions(object):
    def __init__(self) -> None:
        """
        Init function for BaseOptions class, defines all the training and test options.
        """

        super().__init__()

        self.iden_code_dims = 80
        self.expr_code_dims = 64

        self.opt_cam = True
        self.lr = 0.0001
        self.step_decay = 1000
        self.batch_size = 1
        self.num_epochs = 100
        self.num_workers = 2
        self.eye_lr_mult = 1.0
        

        self.bg_type = "white"

        self.featmap_size = 128
        self.featmap_nc = 32


        self.num_add_mouth_points = 3000
        self.shape_color_mlp = [272, 256, 256, 32]
        self.pose_color_mlp = [182, 128, 32]
        self.appea_color_mlp = [235, 256, 256, 32]

        self.eye_color_mlp = [146, 128, 32]

        
        self.shape_deform_mlp = [171, 256, 256, 256, 256, 256, 3]
        self.pose_deform_mlp = [81, 256, 256, 3]
        self.eye_deform_mlp = [45, 128, 128, 3]
        self.eye_deform_mlp_rotate = [45, 128, 128, 4]
        self.is_rotate_eye = True

        self.shape_attributes_mlp = [272, 256, 256, 256, 8]
        self.pose_attributes_mlp = [182, 128, 128, 8]
        self.appea_attributes_opacity_mlp = [235, 256, 256, 256, 1]

        self.eye_attributes_mlp = [146, 128, 128, 6]

        self.unet_atten = True

        self.shape_coeffs_dim = 64
        self.pos_freq = 4
        self.gaze_freq = 4
        self.dist_threshold_near = 0.1
        self.dist_threshold_far = 0.25
        self.deform_scale = 0.3
        self.attributes_scale = 0.2

        self.pred_img_size = 512

        self.vgg_importance = 0.1
        self.eye_loss_importance = 1.0
        self.face_loss_importance = 1.0
        self.gaze_loss_importance = 0.1
        self.use_vgg_loss = True
        self.vgg_loss_begin = 5
        self.use_ssim_loss = True
        self.use_l1_loss = True
        self.use_angular_loss = True
        self.angular_loss_begin = 5
        self.use_patch_gan_loss = False
        self.is_gradual_loss = False
        self.fit_image = False
        self.clip_grad = False

        self.logdir = 'work_dirs'
        self.name = 'gaussiangaze_debug'
        self.checkpoint_path = 'checkpoints'
        self.result_path = 'results'
        self.save_freq = 1000
        self.show_freq = 1000
        self.resume = False
        self.verbose = True
        self.img_dir = "./data/ETH-XGaze"
        self.dataset_name = "eth_xgaze"
        self.down_scale = 4.0
        
        self.base_expr_fix = "configs/config_files/tensor.pt"
        self.load_gazegaussian_checkpoint = ""
        self.load_meshhead_checkpoint = ""


        self.view_num = 45
        self.duration = 3.0 / self.view_num
