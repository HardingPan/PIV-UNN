"""
baseline对比方案一(使用多个训练的raft模型的结果值计算不确定度)的dataset代码。dataset指令示例如下:
```zsh
python mm-2-3.py --model1 /home/panding/code/UR/UR/raft/checkpoints/1.pth --model2 /home/panding/code/UR/UR/raft/checkpoints/2.pth --model3 /home/panding/code/UR/UR/raft/checkpoints/3.pth --model4 /home/panding/code/UR/UR/raft/checkpoints/4.pth --path /home/panding/code/UR/piv-data/test
"""
import sys
sys.path.append('core')
sys.path.append('..')

import argparse
import os
import cv2
import glob
import time
import numpy as np
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

batch_size=3

transform_origin = transforms.ToTensor()

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

# 灰度图像读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return img_rgb[None].to(DEVICE)

def dataload(args):
    
    un = MultiMethod(0)
    
    model1 = torch.nn.DataParallel(RAFT(args))
    model2 = torch.nn.DataParallel(RAFT(args))
    model3 = torch.nn.DataParallel(RAFT(args))
    model4 = torch.nn.DataParallel(RAFT(args))

    model1.load_state_dict(torch.load(args.model1))
    model2.load_state_dict(torch.load(args.model2))
    model3.load_state_dict(torch.load(args.model3))
    model4.load_state_dict(torch.load(args.model4))

    model1 = model1.module
    model2 = model2.module
    model3 = model3.module
    model4 = model4.module

    model1.to(DEVICE)
    model2.to(DEVICE)
    model3.to(DEVICE)
    model4.to(DEVICE)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    
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
            # assert (len(images1) == len(images2))
            
            num = len(images1)

            start_time = time.time()
            for imfile1, imfile2 in zip(images1, images2):
                
                image1_rgb_tensor = load_image(imfile1)
                image2_rgb_tensor = load_image(imfile2)
                
                padder = InputPadder(image1_rgb_tensor.shape)
                image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
                
                # torch.Size([1, 2, 440, 1024])
                flow_low_1, flow_up_1 = model1(image1, image2, iters=20, test_mode=True)
                flow_low_2, flow_up_2 = model2(image1, image2, iters=20, test_mode=True)
                flow_low_3, flow_up_3 = model3(image1, image2, iters=20, test_mode=True)
                flow_low_4, flow_up_4 = model4(image1, image2, iters=20, test_mode=True)
                # viz(flow_up)
                # torch.Size([2, 440, 1024])
                flow_up_1 = torch.squeeze(flow_up_1)
                flow_up_2 = torch.squeeze(flow_up_2)
                flow_up_3 = torch.squeeze(flow_up_3)
                flow_up_4 = torch.squeeze(flow_up_4)

                flow_up_1_u, flow_up_1_v = flow_up_1.split(1, 0)
                flow_up_2_u, flow_up_2_v = flow_up_2.split(1, 0)
                flow_up_3_u, flow_up_3_v = flow_up_3.split(1, 0)
                flow_up_4_u, flow_up_4_v = flow_up_4.split(1, 0)

                result = torch.cat((flow_up_1_u, flow_up_1_v, flow_up_2_u, flow_up_2_v,flow_up_3_u, flow_up_3_v,flow_up_4_u, flow_up_4_v), 0)
                result = result.cpu().numpy()
                uq = un.uncertainty(result)
            end_time = time.time()
            print(f"{cls}: {end_time-start_time}, all: {num}")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', default='/home/panding/code/UR/UR/raft/checkpoints/best-1.pth', help="restore checkpoint")
    parser.add_argument('--model2', default='/home/panding/code/UR/UR/raft/checkpoints/2.pth', help="restore checkpoint")
    parser.add_argument('--model3', default='/home/panding/code/UR/UR/raft/checkpoints/3.pth', help="restore checkpoint")
    parser.add_argument('--model4', default='/home/panding/code/UR/UR/raft/checkpoints/4.pth', help="restore checkpoint")
    # parser.add_argument('--path', default='/home/panding/code/UR/piv-data/raft-test', help="dataset for evaluation")
    parser.add_argument('--path', default='/home/panding/code/UR/piv-data/raft', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    res = dataload(args)
    