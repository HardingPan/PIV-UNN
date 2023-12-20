"""
baseline对比方案一(使用多个训练的raft模型的结果值计算不确定度)的dataset代码。dataset指令示例如下:
```zsh
python dataset-baseline-multimodel.py --model1 /home/panding/code/UR/UR/raft/checkpoints/1.pth --model2 /home/panding/code/UR/UR/raft/checkpoints/2.pth --model3 /home/panding/code/UR/UR/raft/checkpoints/3.pth --model4 /home/panding/code/UR/UR/raft/checkpoints/4.pth --path /home/panding/code/UR/piv-data/test
"""
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
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

# 图像可视化
def viz(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)
 
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('res1.png', flo[:, :, [2,1,0]])

def remap(img, u, v, device):
    img = img.cpu().numpy().squeeze(0)
    u = u.cpu().numpy().squeeze()
    v = v.cpu().numpy().squeeze()
    # print(img.shape)

    # print(u.shape)
    # print(u.dtype)
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(v.shape[0]))
    x = np.float32(x)
    x = x + u
    y = np.float32(y)
    y = y + v

    # print(x.shape)

    re = cv2.remap(img, x, y, interpolation = 4)
    re = torch.from_numpy(re)
    re = re.unsqueeze(0).to(device)

    return re

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

def dataload(args):
    print('model loading...')
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

    data_path = '/home/panding/code/UR/piv-data/ur'
    
    with torch.no_grad():
        # 读取路径内的成对图像和flow真值
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
            
            flow_truth = load_flow_to_numpy(flow).to(DEVICE)

            """
            torch.Size([6, 440, 1024])
            六通道分别为 灰度后的i1, 灰度后的i2, u, v, u_t, v_t
            """
            result = torch.cat((flow_up_1_u, flow_up_1_v, flow_up_2_u, flow_up_2_v,flow_up_3_u, flow_up_3_v,flow_up_4_u, flow_up_4_v, flow_truth), 0)
            result = result.cpu()
            result_np = result.numpy()
            save_path = imfile1[0:31] + 'baseline-multimodel' + imfile1[35:-9]
            # data_path = data_path + '/' + imfile1[6:-4]
            # data_path = imfile1[0:20] + imfile1[:-9]
            print(f"当前存储位置为: {save_path}")

            np.save(save_path, result_np)
            # data_path = '/home/panding/code/UR/data-chair'
            if images_loading_num % 5 == 0:
                print('\n', '--------------images loaded: ', images_loading_num, ' / ', images_num, '-------------', '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', help="restore checkpoint")
    parser.add_argument('--model2', help="restore checkpoint")
    parser.add_argument('--model3', help="restore checkpoint")
    parser.add_argument('--model4', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    res = dataload(args)