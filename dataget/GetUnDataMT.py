import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('PIV-UNN/unflownet/src/')
from model.models import *
from data_processing.read_data import *
from train.train_functions import *

class SinglePre():
    def __init__(self, model_path) -> None:
        # PATH = F"./models/{model_name}"
        unliteflownet = Network()
        # unliteflownet.load_state_dict(torch.load(model_path)['model_state_dict'])
        unliteflownet.load_state_dict(torch.load(model_path)['model_state_dict'])
        unliteflownet.eval()
        unliteflownet.to(device)
        self.model = unliteflownet
        
    def prediction(self, input_data):
        # h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]
        input_data = input_data.view(-1, 2, 256, 256)

        h, w = input_data.shape[-2], input_data.shape[-1]
        x1 = input_data[:, 0, ...].view(-1, 1, h, w)
        x2 = input_data[:, 1, ...].view(-1, 1, h, w)
        y_pre = estimate(x1.to(device), x2.to(device), self.model, train=False)

        u = y_pre[0][0].detach()
        v = y_pre[0][1].detach()

        color_data_pre = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
        u = u.numpy()
        v = v.numpy()
        # Draw velocity magnitude

        # Control arrow density
        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]
        color_data_pre_unliteflownet = color_data_pre
        color_data_pre_unliteflownet = color_data_pre_unliteflownet.transpose(2, 0, 1)
        
        # print(color_data_pre_unliteflownet.shape)
        
        return color_data_pre_unliteflownet

class Transform():
    def __init__(self, device) -> None:
        self.device = device
        self.input_data = None
    def rotate_180(self, arr):
        arr = self.input_data
        arr = np.squeeze(arr, 0)
        arr = np.rot90(arr, k=2, axes=(1, 2))
        arr = np.expand_dims(arr, 0)
        return arr
    def rotate_left_90(self, arr):
        arr = self.input_data
        arr = np.squeeze(arr, 0)
        arr = np.rot90(arr, k=1, axes=(1, 2))
        arr = np.expand_dims(arr, 0)
        return arr
    def rotate_right_90(self, arr):
        arr = self.input_data
        arr = np.squeeze(arr, 0)
        arr = np.rot90(arr, k=-1, axes=(1, 2))
        arr = np.expand_dims(arr, 0)
        return arr
    def flip_0(self, arr):
        # arr = np.squeeze(arr, 0)
        # print(arr.shape)
        arr = np.flip(arr, 1)
        # arr = np.expand_dims(arr, 0)  
        return arr
    def flip_1(self, arr):
        # arr = np.squeeze(arr, 0)
        # print(arr.shape)
        arr = np.flip(arr, 2)
        # arr = np.expand_dims(arr, 0)  
        return arr
    def get_arr(self, input_data):
        
        self.input_data = input_data
        # (1, 256, 256)
        self.img1 = torch.unsqueeze(self.input_data[0], 0).cpu().numpy()
        self.img2 = torch.unsqueeze(self.input_data[1], 0).cpu().numpy()
        # print(self.img2.shape)
        
        arr_1_1 = torch.from_numpy(self.flip_0(self.img1).copy()).to(self.device)
        arr_1_2 = torch.from_numpy(self.flip_0(self.img2).copy()).to(self.device)
        mt_1 = torch.cat((arr_1_1, arr_1_2), 0)
        
        arr_2_1 = torch.from_numpy(self.rotate_180(self.img1).copy()).to(self.device)
        arr_2_2 = torch.from_numpy(self.rotate_180(self.img2).copy()).to(self.device)
        mt_2 = torch.cat((arr_2_1, arr_2_2), 0)
        
        arr_3_1 = torch.from_numpy(self.flip_1(self.img1).copy()).to(self.device)
        arr_3_2 = torch.from_numpy(self.flip_1(self.img2).copy()).to(self.device)
        mt_3 = torch.cat((arr_3_1, arr_3_2), 0)
        
        self.input_data = None

        return [input_data, mt_1, mt_2, mt_3]
   
class UnMM():
    def __init__(self, path, device) -> None:
        self.path = path
        self.device = device
        self.model = None
        self.dataset = None
        self.transform = Transform(self.device)
        self.model_init()
        self.data_init()
        
    def model_init(self):
        model_path = 'PIV-UNN/unflownet/models/model-1.pt'
        self.model = SinglePre(model_path)
        print("models init ok")
    
    def data_init(self):
        flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_all(self.path)
        total_index = np.arange(0, len(flow_img1_name_list), 1)
        self.dataset= FlowDataset(
            total_index, [flow_img1_name_list, flow_img2_name_list],
            targets_index_list=total_index,
            targets=flow_gt_name_list)
        print(f"datas init ok: all {len(self.dataset)}")
        self.dataset.eval()
    
    def get_data(self, index):
        res = []
        input_data, label_data = self.dataset[index]
        inputs = self.transform.get_arr(input_data)
        for i in range(4):
            res.append(self.model.prediction(inputs[i]))
        result = np.concatenate((res[0], res[1], res[2], res[3], label_data), 0)

        return result
    
if __name__ == "__main__":

    DEVICE = 'cuda'
    un = UnMM('/home/panding/code/UR/piv-data/liteflownet-test', DEVICE)
    res = un.get_data(0)