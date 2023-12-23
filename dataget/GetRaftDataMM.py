import sys
sys.path.append('core')
sys.path.append('PIV-UNN/raft/utils/')
sys.path.append('PIV-UNN/raft/')
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
import flow_viz
from utils import InputPadder

DEVICE = 'cuda'

class RaftMM():
    def __init__(self, path, device) -> None:
        self.path = path
        self.device = device
        # 对模型和数据进行初始化
        self.models = []
        self.images1, self.images2, self.truths = [], [], []
        self.model_init()
        self.data_init()
        
    def model_init(self):
        # MultiModel的raft参数初始化
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        
        model_path = [
            'PIV-UNN/raft/checkpoints/1.pth', \
            'PIV-UNN/raft/checkpoints/2.pth', \
            'PIV-UNN/raft/checkpoints/3.pth', \
            'PIV-UNN/raft/checkpoints/4.pth'
        ]
        # 初始化模型
        for i in range(4):
            self.models.append(torch.nn.DataParallel(RAFT(args)))
        for i in range(4):
            self.models[i].load_state_dict(torch.load(model_path[i]))
            self.models[i] = self.models[i].module
            self.models[i].to(self.device)
            self.models[i].eval()
        print("models init ok")
        
    def data_init(self):
        with torch.no_grad():
            self.images1 = glob.glob(os.path.join(self.path, '*_img1.tif'))
            self.images2 = glob.glob(os.path.join(self.path, '*_img2.tif'))
            self.truths = glob.glob(os.path.join(self.path, '*.flo'))

            self.images1 = sorted(self.images1)
            self.images2 = sorted(self.images2)
            self.truths = sorted(self.truths)
            
            print(f"detas init ok: all {len(self.truths)}")
    
    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        img_rgb[:, :, 0] = img
        img_rgb[:, :, 1] = img
        img_rgb[:, :, 2] = img
        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        return img_rgb[None].to(self.device)
    
    def load_flow_to_tensor(self, path):
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        data2D = np.resize(data, (w, h, 2))
        data2D = data2D.transpose(2,0,1)
        data2D_tensor = torch.from_numpy(data2D).to(self.device)
        return data2D_tensor
    
    def get_data(self,index):
        image1_rgb_tensor = self.load_image(self.images1[index])
        image2_rgb_tensor = self.load_image(self.images2[index])
        padder = InputPadder(image1_rgb_tensor.shape)
        image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
        
        result_list = []
        for i in range(4):
            # torch.Size([1, 2, w, h])
            _, flow_up = self.models[i](image1, image2, iters=20, test_mode=True)
            # torch.Size([2, w, h])
            result_list.append(torch.squeeze(flow_up))
        result_list.append(self.load_flow_to_tensor(self.truths[index]))
        
        result = torch.cat((result_list[0], result_list[1], result_list[2], \
                            result_list[3], result_list[4]), 0).cpu().detach().numpy()
        return result

if __name__ == '__main__':
    # 使用实例
    mm = RaftMM('/home/panding/code/UR/piv-data/raft-test', device=DEVICE)
    res = mm.get_data(0)
    print(res.shape)