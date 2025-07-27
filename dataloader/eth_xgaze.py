import json
import os
import random
from typing import List
import math
import torch
import cv2
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch3d.transforms import so3_exponential_map, so3_log_map
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

def dataset_name_parser(dataset_name):
    if dataset_name == "eth_xgaze":
        return "xgaze_"
    elif dataset_name == "columbia":
        return "columbia_"
    elif dataset_name == "gaze_capture":
        return "gaze_capture_"
    elif dataset_name == "mpii_face_gaze":
        return "mpii_"

def gen_eye_mask(ldm, cam_id):
        """
        Calculate bounding box for eye region and save the image.
        :ldm: 2d face landamarks
        :image_gt: Image to analyse
        :save_path: Path used to save the generated eye mask
        """

        mask = np.zeros((512, 512))
        min_point = 0
        if ldm[24][1] < ldm[19][1]:
            min_point = 24
        else:
            min_point = 19
        max_point = 29
        if cam_id in [3, 4, 5, 6]:
            max_point = 33
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 51
        elif cam_id == 13:
            max_point = 62
        else:
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 30
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 33
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 51
        mask[
            int(ldm[min_point][1]) : int(ldm[max_point][1]),
            int(ldm[17][0]) : int(ldm[26][0]),
        ] = 255

        return mask

def get_train_loader(
    args,
    data_dir,
    batch_size,
    dataset_name="eth_xgaze",
    num_workers=0,
    is_shuffle=True,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the train dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """


    refer_list_file = os.path.join("configs/dataset/%s"%(dataset_name), "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)





    sub_folder_use = "train"
    train_set = GazeDataset(
        args,
        dataset_name=dataset_name_parser(dataset_name),
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle)

    return train_loader


def get_val_loader(
    args,
    data_dir,
    batch_size,
    num_val_images,
    dataset_name="eth_xgaze",
    num_workers=0,
    is_shuffle=False,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the validation dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """


    refer_list_file = os.path.join("configs/dataset/%s"%(dataset_name), "train_test_split.json")


    with open(refer_list_file, "r") as f:
        datastore = json.load(f)





    sub_folder_use = "val"
    val_set = GazeDataset(
        args,
        dataset_name=dataset_name_parser(dataset_name),
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        num_val_images=num_val_images,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return val_loader


def get_test_loader(
    args,
    data_dir,
    batch_size,
    dataset_name="eth_xgaze",
    num_workers=0,
    is_shuffle=False,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the test dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """


    refer_list_file = os.path.join("configs/dataset/%s"%(dataset_name), "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)





    sub_folder_use = "test"
    test_set = GazeDataset(
        args,
        dataset_name=dataset_name_parser(dataset_name),
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(
        self,
        args,
        dataset_name: str,
        dataset_path: str,
        keys_to_use: List[str] = None,
        sub_folder="",
        num_val_images=50,
        transform=None,
        is_shuffle=True,
        index_file=None,
        subject=None,
        evaluate=None,
    ):
        """
        Init function for the ETH-XGaze dataset, create key pairs to shuffle the dataset.

        :dataset_path: Path to the subjects files with all the information
        :keys_to_use: The subjects ID to use for the dataset
        :sub_folder: Indicate if it has to create the train,validation or test dataset
        :num_val_images: Used only for the validation dataset, indicate how many images to include in the validation dataset
        :transform: All the transformations to apply to the images
        :is_shuffle: Boolean value that indicates if images will be shuffled
        :index_file: Path to a specific key pairs file
        """
        self.args = args
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.evaluate = evaluate
        self.index_fix = None
        self.dataset_name = dataset_name


        if subject is None:
            self.selected_keys = [k for k in keys_to_use]
        else:
            self.selected_keys = [subject]
        assert len(self.selected_keys) > 0



        self.file_path_list = []
        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.dataset_name + self.selected_keys[num_i])
            self.file_path_list.append(file_path)


        for num_i, file_path in enumerate(self.file_path_list):

            self.hdfs[num_i] = h5py.File(file_path, "r", swmr=True)
            assert self.hdfs[num_i].swmr_mode


        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.file_path_list)):
                if sub_folder == "val":
                    n = num_val_images
                else:
                    n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [
                    (num_i, i) for i in range(n)
                ]
        else:
            print("load the file: ", index_file[0])
            self.idx_to_kv = np.loadtxt(index_file[0], dtype=np.int)


        init_sel = 0
        init_landmarks_3d = torch.from_numpy(self.hdfs[0]["facial_landmarks_3d"][init_sel]).float()
        init_vertices = torch.from_numpy(self.hdfs[0]["vertice"][init_sel]).float()
        init_landmarks_3d = torch.cat([init_landmarks_3d, init_vertices[::100]], 0)
        
        
        wr = torch.from_numpy(self.hdfs[0]["w2c_Rmat"][init_sel, :]).float()
        wt = torch.from_numpy(self.hdfs[0]["w2c_Tvec"][init_sel, :]).float()

        init_landmarks_3d = torch.matmul(init_landmarks_3d - wt, wr.permute(1,0))
        pose = torch.from_numpy(self.hdfs[0]["pose"][init_sel]).float()


        R = so3_exponential_map(pose[None, :3])[0]
        T = pose[None, 3:]
        S = torch.from_numpy(self.hdfs[0]["scale"][init_sel]).float()


        self.init_landmarks_3d_neutral = (torch.matmul(init_landmarks_3d - T, R)) / S

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None
        
        self.target_idx = np.loadtxt(f"configs/config_files/{self.dataset_name}evaluation_target_single_subject.txt", dtype=np.int)
        if sub_folder == "val":
            new_idx_to_kv = []
            self.target_dict = {}
            for idx_to_kv, target_idx in zip(self.idx_to_kv, self.target_idx):
                new_idx_to_kv.append((idx_to_kv[0], target_idx))
                self.target_dict[(idx_to_kv[0], target_idx)] = idx_to_kv[1]
            self.idx_to_kv = new_idx_to_kv
        self.is_target = False

        self.hdf = None
        self.transform = transform

        self.iden_code_dims = args.iden_code_dims
        self.expr_code_dims = args.expr_code_dims

    def __len__(self):
        """
        Function that returns the length of the dataset.

        :return: Returns the length of the dataset
        """

        return len(self.idx_to_kv)

    def __del__(self):
        """
        Close all the hdfs files of the subjects.

        """

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def modify_index(self, index, is_target=False):
        self.index_fix = index
        self.is_target = is_target

    def __getitem__(self, idx_input):
        """
        Return one sample from the dataset, the on in poistiongiven by idx.

        :idx: Indicate the position of the data sample to return
        :return: Returns one sample from the dataset
        """
        

        if self.index_fix is not None:
            idx_input = self.index_fix
        

        if self.sub_folder != "val":
            flag = True
            while flag:
                key, idx = self.idx_to_kv[idx_input]
                if self.is_target:
                    idx = self.target_dict[(key, idx)]

                self.hdf = h5py.File(
                    os.path.join(self.file_path_list[key]),
                    "r",
                    swmr=True,
                )
                assert self.hdf.swmr_mode
                if (
                    len(np.unique(self.hdf["head_mask"][idx, :])) == 1 or  \
                    (len(np.unique(self.hdf["left_eye_mask"][idx, :])) == 1 and \
                    len(np.unique(self.hdf["right_eye_mask"][idx, :])) == 1 )
                ):
                    idx_input = random.randint(0, len(self.idx_to_kv) - 1)
                else:
                    flag = False
        else:
            key, idx = self.idx_to_kv[idx_input]
            target_idx = self.target_dict[(key, idx)]
            if self.is_target:
                idx = self.target_dict[(key, idx)]
            self.hdf = h5py.File(
                os.path.join(self.file_path_list[key]),
                "r",
                swmr=True,
            )

            assert self.hdf.swmr_mode


        image = self.hdf["face_patch"][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        face_mask = self.hdf["head_mask"][idx, :]
        kernel_2 = np.ones((3, 3), dtype=np.uint8)
        face_mask = cv2.erode(face_mask, kernel_2, iterations=2)
        face_mask = face_mask.astype(np.float32) / 255.0
        face_mask = face_mask[None, ...]
        left_eye_mask = self.hdf["left_eye_mask"][idx, :]
        left_eye_mask = left_eye_mask.astype(np.float32) / 255.0
        left_eye_mask = left_eye_mask[None, ...]
        right_eye_mask = self.hdf["right_eye_mask"][idx, :]
        right_eye_mask = right_eye_mask.astype(np.float32) / 255.0
        right_eye_mask = right_eye_mask[None, ...]

        ldms = self.hdf["facial_landmarks"][idx, :]

        ldms_3d = self.hdf["facial_landmarks_3d"][idx, :]

        wr = self.hdf["w2c_Rmat"][idx, :].astype(np.float32)
        wt = self.hdf["w2c_Tvec"][idx, :].astype(np.float32)

        vertices = self.hdf["vertice"][idx, :]
        
        ldms_3d = np.concatenate([ldms_3d, vertices[::100]], axis=0)
        ldms_3d = (ldms_3d - wt) @ wr.T

        cam_ind = self.hdf["cam_index"][idx, :]

        nl3dmm_para_dict = {}

        nl3dmm_para_dict["code"] = self.hdf["latent_codes"][idx, :]
        nl3dmm_para_dict["iden_code"] = self.hdf["latent_codes"][idx, :self.iden_code_dims]
        nl3dmm_para_dict["expr_code"] = self.hdf["latent_codes"][idx, self.iden_code_dims : self.iden_code_dims + self.expr_code_dims]

        nl3dmm_para_dict["extrinsics"] = np.concatenate([self.hdf["w2c_Rmat"][idx, :], self.hdf["w2c_Tvec"][idx, :][:, None]], axis=1)
        
        intrinsics = self.hdf["inmat"][idx, :]
        down_scale = self.args.down_scale

        intrinsics[0, 0] = intrinsics[0, 0] * 2 / image.shape[2]
        intrinsics[0, 2] = intrinsics[0, 2] * 2 / image.shape[2] - 1
        intrinsics[1, 1] = intrinsics[1, 1] * 2 / image.shape[1]
        intrinsics[1, 2] = intrinsics[1, 2] * 2 / image.shape[1] - 1
        nl3dmm_para_dict["intrinsics"] = intrinsics

        fovx = 2 * math.atan(1 / intrinsics[0, 0])
        fovy = 2 * math.atan(1 / intrinsics[1, 1])

        R = self.hdf["w2c_Rmat"][idx, :].transpose()



        T = self.hdf["w2c_Tvec"][idx, :]



        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        nl3dmm_para_dict["pitchyaw"] = self.hdf["pitchyaw_head"][idx, :]
        nl3dmm_para_dict["head_pose"] = self.hdf["face_head_pose"][idx, :]
        nl3dmm_para_dict["pose"] = self.hdf["pose"][idx, :]
        nl3dmm_para_dict["scale"] = self.hdf["scale"][idx, :]


        return {
            "idx_input": idx_input,
            "idx": idx,
            "key": key,
            "down_scale": down_scale,
            "image": image,
            "face_mask": face_mask,
            "left_eye_mask": left_eye_mask,
            "right_eye_mask": right_eye_mask,
            "nl3dmm_para_dict": nl3dmm_para_dict,
            "ldms": ldms,
            "ldms_3d": ldms_3d,
            "cam_ind": cam_ind,
            "fovx": fovx,
            "fovy": fovy,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "world_view_transform": world_view_transform,
            "projection_matrix": projection_matrix,
        }
