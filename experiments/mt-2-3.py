"""
python dataset-baseline-multitransform.py --model /home/panding/code/UR/UR/raft/checkpoints/2.pth --path /home/panding/code/UR/piv-data/test
"""

import sys
sys.path.append('core')
sys.path.append('..')

import argparse
import os
import cv2
import glob
import numpy as np
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from baseline import MultiMethod

DEVICE = 'cuda'


# 灰度图像读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return img_rgb[None].to(DEVICE)


def rotate_180(arr):
    
    arr = np.squeeze(arr, 0)
    arr = np.rot90(arr, k=2, axes=(1, 2))
    arr = np.expand_dims(arr, 0)
    
    return arr

def rotate_left_90(arr):
    
    arr = np.squeeze(arr, 0)
    arr = np.rot90(arr, k=1, axes=(1, 2))
    arr = np.expand_dims(arr, 0)
    
    return arr
    
def rotate_right_90(arr):
    
    arr = np.squeeze(arr, 0)
    arr = np.rot90(arr, k=-1, axes=(1, 2))
    arr = np.expand_dims(arr, 0)
    
    return arr

def flip_0(arr):
    
    arr = np.squeeze(arr, 0)
    # print(arr.shape)
    arr = np.flip(arr, 1)
    arr = np.expand_dims(arr, 0)  
    
    return arr

def flip_1(arr):
    
    arr = np.squeeze(arr, 0)
    # print(arr.shape)
    arr = np.flip(arr, 2)
    arr = np.expand_dims(arr, 0)  
    
    return arr

def dataload(args):
    
    un = MultiMethod(1)
    
    print('model loading...')
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    print('model has been loaded!')

    classes = ['backstep', 'cylinder', 'JHTDB_channel', 'JHTDB_isotropic1024', 'JHTDB_mhd1024', 'SQG']
    
    for i in range(len(classes)):
        cls = classes[i]
    
        with torch.no_grad():
            # 读取路径内的成对图像和flow真值
            images1 = glob.glob(os.path.join(args.path,cls+'*.png')) + \
                    glob.glob(os.path.join(args.path, cls+'*.jpg')) + \
                    glob.glob(os.path.join(args.path, cls+'*.ppm')) + \
                    glob.glob(os.path.join(args.path, cls+'*_img1.tif'))
                
            images2 = glob.glob(os.path.join(args.path, cls+'*.png')) + \
                    glob.glob(os.path.join(args.path, cls+'*.jpg')) + \
                    glob.glob(os.path.join(args.path, cls+'*.ppm')) + \
                    glob.glob(os.path.join(args.path, '*_img2.tif'))
            
            images1 = sorted(images1)
            images2 = sorted(images2)
            
            num = len(images1)

            start_time = time.time()
            for imfile1, imfile2 in zip(images1, images2):

                image1_rgb_tensor = load_image(imfile1)
                image2_rgb_tensor = load_image(imfile2)
                
                padder = InputPadder(image1_rgb_tensor.shape)
                image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
                
                image1_array = image1.cpu().numpy()
                image2_array = image2.cpu().numpy()
                
                image1_flip_0 = torch.from_numpy(flip_0(image1_array).copy()).to(DEVICE)
                image2_flip_0 = torch.from_numpy(flip_0(image2_array).copy()).to(DEVICE)
                
                image1_rotate_180 = torch.from_numpy(rotate_180(image1_array).copy()).to(DEVICE)
                image2_rotate_180 = torch.from_numpy(rotate_180(image2_array).copy()).to(DEVICE)
                
                image1_flip_1 = torch.from_numpy(flip_1(image1_array).copy()).to(DEVICE)
                image2_flip_1 = torch.from_numpy(flip_1(image2_array).copy()).to(DEVICE)
                
                
                flow_low_1, flow_up_1 = model(image1, image2, iters=20, test_mode=True)
                flow_low_2, flow_up_2 = model(image1_flip_0, image2_flip_0, iters=20, test_mode=True)
                flow_low_3, flow_up_3 = model(image1_rotate_180, image2_rotate_180, iters=20, test_mode=True)
                flow_low_4, flow_up_4 = model(image1_flip_1, image2_flip_1, iters=20, test_mode=True)
                
                flow_up_1 = torch.squeeze(flow_up_1)
                flow_up_2 = torch.squeeze(flow_up_2)
                flow_up_3 = torch.squeeze(flow_up_3)
                flow_up_4 = torch.squeeze(flow_up_4)
                
                result = torch.cat((flow_up_1, flow_up_2, flow_up_3, flow_up_4), 0)
                result = result.cpu().numpy()
                
                uq = un.uncertainty(result)
                
            end_time = time.time()
            print(f"{cls}: {end_time-start_time}, all: {num}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/panding/code/UR/UR/raft/checkpoints/best-1.pth', help="restore checkpoint")
    # parser.add_argument('--path', default='/home/panding/code/UR/piv-data/raft-test', help="dataset for evaluation")
    parser.add_argument('--path', default='/home/panding/code/UR/piv-data/raft', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    res = dataload(args)