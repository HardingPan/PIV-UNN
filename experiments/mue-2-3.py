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

from muenn import MueNN

DEVICE = 'cuda'

transform_origin = transforms.ToTensor()

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
    
    unn_path = '/home/panding/code/UR/unet-model/best-1.pt'
    unn = MueNN(unn_path, DEVICE)
    
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
                
                image1_gray_tensor = transform_origin(Image.open(imfile1)).to(DEVICE)
                image2_gray_tensor = transform_origin(Image.open(imfile2)).to(DEVICE)

                image1_rgb_tensor = load_image(imfile1)
                image2_rgb_tensor = load_image(imfile2)
                
                padder = InputPadder(image1_rgb_tensor.shape)
                image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
                
                flow_low_1, flow_up_1 = model(image1, image2, iters=20, test_mode=True)

                flow_up = torch.squeeze(flow_up_1)
                
                in_put = torch.cat((image2_gray_tensor, image1_gray_tensor, flow_up), 0)
                
                res = unn.get_sigma(in_put)
                
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