"""
python truth.py --model /home/panding/code/UR/UR/raft/checkpoints/best-1.pth --path /home/panding/code/UR/piv-data/test
生成的npy文件内容为
flow_up: 计算得到的flow, 2*w*h, 分别为u和v
flow_truth: flow真值, 2*w*h, 分别为u_t和v_t
flow_loss: 损失函数得到的flow_up与flow_truth之差, 2*w*h
metrics_epe, metrics_1, metrics_3, metrics_5
"""

from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import InputPadder
from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets_truth

DEVICE = 'cuda'

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 8000

# 读取flo为tensor
def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    data2D = data2D.transpose(2,0,1)
    data2D_tensor = torch.from_numpy(data2D)
    return data2D_tensor

# 灰度图像读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return img_rgb[None].to(DEVICE)

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        flow_loss += i_weight * (valid[:, None] * i_loss)
        
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    
    metrics_epe = torch.reshape(epe, (1, 1, 256, 256))
    metrics_1 = torch.reshape((epe < 1), (1, 1, 256, 256))
    metrics_3 = torch.reshape((epe < 3), (1, 1, 256, 256))
    metrics_5 = torch.reshape((epe < 5), (1, 1, 256, 256))

    return flow_loss, metrics_epe, metrics_1, metrics_3, metrics_5


VALID = torch.full((1, 256, 256), 1).to(DEVICE)

def main(args):
    
    # 加载网络模型
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    images1 = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg')) + \
             glob.glob(os.path.join(args.path, '*.ppm')) + \
             glob.glob(os.path.join(args.path, '*_img1.tif'))
        
    images2 = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg')) + \
             glob.glob(os.path.join(args.path, '*.ppm')) + \
             glob.glob(os.path.join(args.path, '*_img2.tif'))

    flow_truth = glob.glob(os.path.join(args.path, '*.flo'))
    
    images1 = sorted(images1)
    images2 = sorted(images2)
    flow_truth = sorted(flow_truth)
    print(f"images1 length: {len(images1)}, images2 length: {len(images2)}")
    assert (len(images1) == len(images2))
    print('data has been loaded!')
    
    images_num = len(images2)
    print(images_num)
    images_loading_num = 1
    print('\n', '--------------images loading...-------------', '\n')
    
    for flow, imfile1, imfile2 in zip(flow_truth, images1, images2):
        
        flow_truth = load_flow_to_numpy(flow).to(DEVICE)
        
        # print(f"flow: {flow}, img1: {imfile1}, img2: {imfile2}")
        images_loading_num = images_loading_num + 1
        # torch.Size([3, 436, 1024])
        image1_rgb_tensor = load_image(imfile1)
        image2_rgb_tensor = load_image(imfile2)
            
        """
        torch.Size([1, 3, 440, 1024])
        这个pad操作会改变张量的尺寸, 后面灰度张量也需要pad一下才可以和光流张量拼接
        """
        padder = InputPadder(image1_rgb_tensor.shape)
        image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        
        flow_loss, metrics_epe, metrics_1, metrics_3, metrics_5 = sequence_loss(flow_up, flow_truth, VALID)
        flow_truth = torch.unsqueeze(flow_truth, 0)
        # print(f"flow_up:{flow_up.shape}, flow_truth:{flow_truth.shape}, flow_loss:{flow_loss.shape}, metrics_epe:{metrics_epe.shape}")
        
        result = torch.cat((flow_up, flow_truth, flow_loss, metrics_epe, metrics_1, metrics_3, metrics_5), 1)
        result = result.squeeze(0)
        # print(result.shape)
        result = result.cpu()
        result_np = result.detach().numpy()
        save_path = '/home/panding/code/UR/piv-data/truth' + imfile1[35:-9]
        # print(save_path)
        np.save(save_path, result_np)
        # data_path = '/home/panding/code/UR/data-chair'
        if images_loading_num % 5 == 0:
            print('\n', '--------------images loaded: ', images_loading_num, ' / ', images_num, '-------------', '\n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)