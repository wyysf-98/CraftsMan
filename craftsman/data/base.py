import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from craftsman.utils.typing import *

def fit_bounding_box(img, mask, marign_pix_dis, background_color):
    # alpha_channel = img[:, :, 3]
    alpha_channel = mask.numpy().squeeze()
    height = np.any(alpha_channel, axis=1)
    width = np.any(alpha_channel, axis=0)
    h_min, h_max = np.where(height)[0][[0, -1]]
    w_min, w_max = np.where(width)[0][[0, -1]]
    box_height = h_max - h_min
    box_width = w_max - w_min
    cropped_image = img[h_min:h_max, w_min:w_max]
    if box_height > box_width:
        new_hight = 512 - 2 * marign_pix_dis
        new_width = int((512 - 2 * marign_pix_dis) / (box_height) * box_width) + 1
    else:
        new_hight = int((512 - 2 * marign_pix_dis) / (box_width) * box_height) + 1
        new_width = 512 - 2 * marign_pix_dis 
    new_h_min_pos = int((512 - new_hight) / 2 + 1)
    new_h_max_pos = new_hight + new_h_min_pos

    new_w_min_pos = int((512 - new_width) / 2 + 1)
    new_w_max_pos = new_width + new_w_min_pos
    # extend of the bbox
    new_image = np.full((512, 512, 3), background_color)
    new_image[new_h_min_pos:new_h_max_pos, new_w_min_pos:new_w_max_pos, :] = cv2.resize(cropped_image.numpy(), (new_width, new_hight))
    
    return torch.from_numpy(new_image)
        
@dataclass
class BaseDataModuleConfig:
    local_dir: str = None

    ################################# Geometry part #################################
    load_geometry: bool = True           # whether to load geometry data
    geo_data_type: str = "occupancy"     # occupancy, sdf
    geo_data_path: str = ""              # path to the geometry data
    # for occupancy and sdf data
    n_samples: int = 4096                # number of points in input point cloud
    upsample_ratio: int = 1              # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = None    # sampling strategy for input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf
    tsdf_threshold: float = 0.05         # threshold for truncating sdf values, used when input is sdf
    n_supervision: int = 10000           # number of points in supervision

    ################################# Image part #################################
    load_image: bool = False             # whether to load images 
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (0.5, 0.5, 0.5)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    marign_pix_dis: int = 30             # margin of the bounding box
    batch_size: int = 32
    num_workers: int = 8


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split

        self.uids = json.load(open(f'{cfg.local_dir}/{split}.json'))
        print(f"Loaded {len(self.uids)} {split} uids")
    
    def __len__(self):
        return len(self.uids)


    def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        if self.cfg.geo_data_type == "occupancy":
            # for input point cloud, using Objaverse-MIX data
            pointcloud = np.load(f'{self.cfg.geo_data_path}/{self.uids[index]}/pointcloud.npz')
            surface = np.asarray(pointcloud['points']) * 2 # range from -1 to 1
            normal = np.asarray(pointcloud['normals'])
            surface = np.concatenate([surface, normal], axis=1)
        elif self.cfg.geo_data_type == "sdf":
            # for sdf data with our own format
            data = np.load(f'{self.cfg.geo_data_path}/{self.uids[index]}.npz')
            # for input point cloud
            surface = data["surface"]
        else:
            raise NotImplementedError(f"Data type {self.cfg.geo_data_type} not implemented")
        
        # random sampling
        if self.cfg.sampling_strategy == "random":
            rng = np.random.default_rng()
            ind = rng.choice(surface.shape[0], self.cfg.upsample_ratio * self.cfg.n_samples, replace=False)
            surface = surface[ind]
        elif self.cfg.sampling_strategy == "fps":
            import fpsample
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], self.cfg.n_samples, h=5)
            surface = surface[kdline_fps_samples_idx]
        elif self.cfg.sampling_strategy is None:
            pass
        else:
            raise NotImplementedError(f"sampling strategy {self.cfg.sampling_strategy} not implemented")
        # rescale data
        surface[:, :3] = surface[:, :3] * self.cfg.scale # target scale
        ret = {
            "uid": self.uids[index].split('/')[-1],
            "surface": surface.astype(np.float32),
        }

        return ret

    def _load_shape_supervision_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.data_type == "occupancy":
            points = np.load(f'{self.cfg.geo_data_path}/{self.uids[index]}/points.npz')
            rand_points = np.asarray(points['points']) * 2 # range from -1.1 to 1.1
            occupancies = np.asarray(points['occupancies'])
            occupancies = np.unpackbits(occupancies)
        elif self.cfg.data_type == "sdf":
            data = np.load(f'{self.cfg.geo_data_path}/{self.uids[index]}.npz')
            rand_points = data['rand_points']
            sdfs = data['sdfs']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        ret["rand_points"] = rand_points.astype(np.float32)

        if self.cfg.data_type == "occupancy":
            assert self.cfg.supervision_type == "occupancy", "Only occupancy supervision is supported for occupancy data"
            occupancies = occupancies[ind]
            ret["occupancies"] = occupancies.astype(np.float32)
        elif self.cfg.data_type == "sdf":
            if self.cfg.supervision_type == "sdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32)
            elif self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold) / self.cfg.tsdf_threshold
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret


    def _load_image(self, index: int) -> Dict[str, Any]:
        def _load_single_image(img_path, background_color, marign_pix_dis=None):
            img = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                )
                / 255.0
            ).float()
            mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
            image: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + background_color[
                None, None, :
            ] * (1 - mask)
            if marign_pix_dis is not None:
                image = fit_bounding_box(image, mask, marign_pix_dis, background_color)
            return image, mask
        
        if self.cfg.background_color == [-1, -1, -1]:
            background_color = torch.randint(0, 256, (3,))
        else:
            background_color = torch.as_tensor(self.cfg.background_color)
        ret = {}
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert self.cfg.n_views == 1, "Only single view is supported for single image"
            sel_idx = random.choice(self.cfg.idx)
            ret["sel_image_idx"] = sel_idx
            if self.cfg.image_type == "rgb":
                img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{'{:04d}'.format(sel_idx)}_rgb.jpeg"
            elif self.cfg.image_type == "normal":
                img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{'{:04d}'.format(sel_idx)}_normal.jpeg"
            ret["image"], ret["mask"] = _load_single_image(img_path, background_color, self.cfg.marign_pix_dis)

        else:
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        
        return ret

    def _get_data(self, index):
        ret = {"uid": self.uids[index]}
        # load geometry
        if self.cfg.load_geometry:
            if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                # load shape
                ret = self._load_shape_from_occupancy_or_sdf(index)
                # load supervision for shape
                if self.cfg.load_supervision:
                    ret.update(self._load_shape_supervision_occupancy_or_sdf(index))
            else:
                raise NotImplementedError(f"Geo data type {self.cfg.geo_data_type} not implemented")

        # load image
        if self.cfg.load_image:
            ret.update(self._load_image(index))

        return ret
        
    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        return torch.utils.data.default_collate(batch)
