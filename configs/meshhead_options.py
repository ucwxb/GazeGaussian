class BaseOptions(object):
    def __init__(self) -> None:
        """
        Init function for BaseOptions class, defines all the training and test options.
        """
        self.iden_code_dims = 80
        self.expr_code_dims = 64

        self.geo_mlp = [27, 256, 256, 256, 256, 256, 132]                    # dimensions of geometry MLP

        self.shape_color_mlp = [272, 256, 256, 32]              # dimensions of expression color MLP
        self.pose_color_mlp = [182, 128, 32]             # dimensions of pose color MLP

        self.appea_color_mlp = [235, 256, 256, 32]            # dimensions of appearance color MLP
        self.eye_color_mlp = [146, 128, 32]


        self.shape_deform_mlp = [171, 256, 256, 256, 256, 256, 3]             # dimensions of expression deformation MLP
        self.pose_deform_mlp = [81, 256, 256, 3]            # dimensions of pose deformation MLP
        self.eye_deform_mlp = [45, 128, 128, 3]
        self.eye_deform_mlp_rotate = [45, 128, 128, 4]
        self.pos_freq = 4                    # frequency of positional encoding
        self.gaze_freq = 4                   # frequency of gaze encoding
        self.is_rotate_eye = True
        self.model_bbox = [[-1.6, 1.6], [-1.7, 1.8], [-2.5, 1.0]]                 # bounding box of the head model
        self.dist_threshold_near = 0.1       # threshold t1
        self.dist_threshold_far = 0.25        # thresgold t2
        self.deform_scale = 0.3              # scale factor for deformation
        self.subdivide = False               # subdivide the tetmesh (resolution: 128 --> 256) or not3

        self.lr_net = 1e-3                                   # learning rate for models and networks
        self.lr_lmk = 1e-4                                   # learning rate for 3D landmarks
        self.num_epochs = 10

        self.load_meshhead_checkpoint = ""     # load mesh head checkpoint or not

        self.logdir = 'work_dirs'
        self.name = 'meshhead_debug'
        self.checkpoint_path = 'checkpoints'
        self.result_path = 'results'
        self.save_freq = 1000
        self.show_freq = 1000
        self.resume = False
        self.verbose = True
        self.down_scale = 1.0

        self.img_dir = "./data/ETH-XGaze"
        self.num_workers = 2
        self.batch_size = 1
        self.dataset_name = "eth_xgaze"
