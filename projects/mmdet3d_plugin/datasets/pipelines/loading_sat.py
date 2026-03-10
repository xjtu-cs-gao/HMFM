import os
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
from mmdet.datasets.pipelines import to_tensor


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations
from mmcv.parallel import DataContainer as DC

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams


@PIPELINES.register_module()
class LoadandProcessSatelliteMapFromFile(object):
    def __init__(self, img_size=(400, 200), to_float32=False, dataset='nusc'):
        self.img_size = img_size
        self.to_float32 = to_float32
        self.dataset = dataset
    def __call__(self, results):
        filename = results['satellite_map_path']
        
        # 处理缺失的卫星地图文件
        if filename is None or not os.path.exists(filename):
            # 创建默认的黑色图像作为占位符
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            print(f"Warning: 卫星地图文件缺失或为None: {filename}，使用默认黑色图像")
        else:
            mean=[103.530, 116.280, 123.675]
            std=[1.0, 1.0, 1.0]
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            to_rgb=False

            img = mmcv.imread(filename)
            img = mmcv.imresize(img, self.img_size, return_scale=False)
            if self.dataset == 'nusc':
                img = mmcv.imflip(img, direction='vertical')
                img = np.rot90(img, k=-1)  # k=-1表示顺时针90度
            img = mmcv.imnormalize(img, mean, std, to_rgb)
        
        if self.to_float32:
            img = img.astype(np.float32)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        results['satellite_map_img'] = DC(to_tensor(img), cpu_only=False, stack=True)

        return results
    