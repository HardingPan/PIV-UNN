# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from scipy import misc
import cv2 as cv

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            
            # 将单通道的piv图像变成三通道的
            img1 = np.expand_dims(img1, axis=2)
            img1 = np.repeat(img1, 3, axis=2)

            img2 = np.expand_dims(img2, axis=2)
            img2 = np.repeat(img2, 3, axis=2)
            
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        
        # print(f"flow数据集数量: {len(self.flow_list)}")
        # print(f"图像对数据集数量: {len(self.image_list)}")
        
        img1 = cv.imread(self.image_list[index][0])
        img2 = cv.imread(self.image_list[index][1])
        
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        # print(f"第一次的尺寸为{img1.shape}")
        # # 将单通道的piv图像变成三通道的
        # img1 = np.expand_dims(img1, axis=2)
        # print(f"第二次的尺寸为{img1.shape}")
        # img1 = np.repeat(img1, 3, axis=2)
        # print(f"第三次的尺寸为{img1.shape}")
        # img2 = np.expand_dims(img2, axis=2)
        # img2 = np.repeat(img2, 3, axis=2)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        
class Piv(FlowDataset):
    """
    Piv的dataset
    路径设置为piv数据集位置
    """
    def __init__(self, aug_params=None, split='train', root='/home/panding/code/UR/piv-data/test'):
        super(Piv, self).__init__(aug_params)
        
        # 加载piv数据集中的tif和flo数据
        # images1 = sorted(glob(osp.join(root, '*_img1.tif')))
        # images2 = sorted(glob(osp.join(root, '*_img2.tif')))
        # flows = sorted(glob(osp.join(root, '*.flo')))
        images1 = sorted(glob(osp.join(root, '*_img1.tif')))
        images2 = sorted(glob(osp.join(root, '*_img2.tif')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        # 确保数据集中tif的图像对数量和flow数据数量相等
        # assert (len(images)//2 == len(flows))
        assert (len(images1) == len(flows))
        
        idx = np.random.permutation(len(images1))
        images1_shuffled = [images1[i] for i in idx]
        images2_shuffled = [images2[i] for i in idx]
        flows_shuffled = [flows[i] for i in idx]
        
        """import numpy as np
        list1 = [1, 2, 3, 4, 5]
        list2 = ['a', 'b', 'c', 'd', 'e']
        # 生成随机排列的索引数组
        idx = np.random.permutation(len(list1))
        # 切片得到随机排序的列表
        list1_shuffled = [list1[i] for i in idx]
        list2_shuffled = [list2[i] for i in idx]
        print(list1_shuffled)
        # 输出：[2, 5, 1, 4, 3]print(list2_shuffled)  # 输出：['b', 'e', 'a', 'd', 'c']"""
        for flow, img1, img2 in zip(flows_shuffled, images1_shuffled, images2_shuffled):
            # 这边载入的是文件名
            self.flow_list.append(flow)
            self.image_list.append([img1, img2])



def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    # 定义piv数据的dataloader
    if args.stage == 'piv':
        # aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        aug_params = None
        # 使用piv数据的dataset方法
        train_dataset = Piv(aug_params, split='training')
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

