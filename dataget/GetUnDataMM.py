import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('PIV-UNN/unflownet/src/')
from src.model.models import *
from src.data_processing.read_data import *
from src.train.train_functions import *

data_path = '/home/panding/code/UR/piv-data/liteflownet-test'
result_path = "/home/panding/code/UR/piv-data/unflownet-mm/"

class SinglePre():
    def __init__(self, model_name) -> None:
        PATH = F"./models/{model_name}"
        unliteflownet = Network()
        unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
        unliteflownet.eval()
        unliteflownet.to(device)
        print('unliteflownet load successfully.')
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
    
def main():
    predictor_1 = SinglePre('model-5.pt')
    predictor_2 = SinglePre('model-2.pt')
    predictor_3 = SinglePre('model-3.pt')
    predictor_4 = SinglePre('model-4.pt')
    
    
    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_all(data_path)
    # flow_img1_name_list, flow_img2_name_list, flow_gt_name_list = read_all(data_path)
    # flow_dir = glob.glob(data_path + "/*[!json]")
    print(len(flow_dir), len(flow_img1_name_list))
    # assert len(flow_dir) == len(flow_img1_name_list)

    # print(flow_img1_name_list)
    
    total_index = np.arange(0, len(flow_img1_name_list), 1)
    test_dataset= FlowDataset(
        total_index, [flow_img1_name_list, flow_img2_name_list],
        targets_index_list=total_index,
        targets=flow_gt_name_list)
    print(len(test_dataset))
    test_dataset.eval()
    
    for number in range(len(test_dataset)):
        name = flow_img1_name_list[number]
        name = name.split('/')[-1][:-9]
        print("name", name)
        # print("xxx", len(flow_img1_name_list), len(test_dataset))
        input_data, label_data = test_dataset[number]
        res_1 = predictor_1.prediction(input_data)
        res_2 = predictor_2.prediction(input_data)
        res_3 = predictor_3.prediction(input_data)
        res_4 = predictor_4.prediction(input_data)
        res = np.concatenate((res_1, res_2, res_3, res_4, res_1), 0)
        # print(res.shape)
        save_path = result_path + name + '.npy'
        # print(save_path)
        np.save(save_path, res)

def test_estimate():

    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_all(data_path)
    # flow_img1_name_list, flow_img2_name_list, flow_gt_name_list = read_all(data_path)
    # flow_dir = glob.glob(data_path + "/*[!json]")
    print(len(flow_dir))
    assert len(flow_dir) == len(flow_img1_name_list)
    flow_dataset = {}
    # print(len(flow_img1_name_list))
    for i, f_name in enumerate(flow_dir):
        print("XXXX", flow_img1_name_list[i])

        total_index = np.arange(0, len(flow_img1_name_list[i]), 1)
        flow_dataset[f_name] = FlowDataset(
            total_index, [flow_img1_name_list, flow_img2_name_list],
            targets_index_list=total_index,
            targets=flow_gt_name_list)

    flow_type = [f_dir for f_dir in flow_dir]
    print("Flow cases: ", flow_type)

    # Load pretrained model
    model_save_name = 'model-1.pt'
    PATH = F"./models/{model_save_name}"
    unliteflownet = Network()
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()
    unliteflownet.to(device)
    print('unliteflownet load successfully.')

    # Visualize results, random select a flow type
    f_type = random.randint(0, len(flow_type) - 1)
    print("Selected flow scenario: ", flow_type[f_type])
    test_dataset = flow_dataset[flow_type[f_type]]
    test_dataset.eval()
    
    resize = False
    save_to_disk = False

    # random select a sample
    number_total = len(test_dataset)
    print(f"number_total: {number_total}")
    number = random.randint(0, number_total - 1)
    input_data, label_data = test_dataset[number]

    h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]

    if resize:
        input_data = F.interpolate(input_data.view(-1, 2, h_origin, w_origin),
                                   (256, 256),
                                   mode='bilinear',
                                   align_corners=False)
    else:
        input_data = input_data.view(-1, 2, 256, 256)

    h, w = input_data.shape[-2], input_data.shape[-1]
    x1 = input_data[:, 0, ...].view(-1, 1, h, w)
    x2 = input_data[:, 1, ...].view(-1, 1, h, w)

    # Visualization
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))

    # ------------Unliteflownet estimation-----------
    b, _, h, w = input_data.size()
    y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
    y_pre = F.interpolate(y_pre, (h, w), mode='bilinear', align_corners=False)

    resize_ratio_u = h_origin / h
    resize_ratio_v = w_origin / w
    u = y_pre[0][0].detach() * resize_ratio_u
    v = y_pre[0][1].detach() * resize_ratio_v

    color_data_pre = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
    u = u.numpy()
    v = v.numpy()
    # Draw velocity magnitude
    axarr[1].imshow(fz.convert_from_flow(color_data_pre))
    # Control arrow density
    X = np.arange(0, h, 8)
    Y = np.arange(0, w, 8)
    xx, yy = np.meshgrid(X, Y)
    U = u[xx.T, yy.T]
    V = v[xx.T, yy.T]
    # Draw velocity direction
    axarr[1].quiver(yy.T, xx.T, U, -V)
    axarr[1].axis('off')
    color_data_pre_unliteflownet = color_data_pre
    color_data_pre_unliteflownet = color_data_pre_unliteflownet.transpose(2, 0, 1)
    print(color_data_pre_unliteflownet.shape)
    # ---------------Label data------------------
    u = label_data[0].detach()
    v = label_data[1].detach()

    color_data_label = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
    u = u.numpy()
    v = v.numpy()
    # Draw velocity magnitude
    axarr[0].imshow(fz.convert_from_flow(color_data_label))
    # Control arrow density
    X = np.arange(0, h, 8)
    Y = np.arange(0, w, 8)
    xx, yy = np.meshgrid(X, Y)
    U = u[xx.T, yy.T]
    V = v[xx.T, yy.T]

    # Draw velocity direction
    axarr[0].quiver(yy.T, xx.T, U, -V)
    axarr[0].axis('off')
    color_data_pre_label = color_data_pre

    if save_to_disk:
        fig.savefig('./output/frame_%d.png' % number, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='train the model')
    parser.add_argument('--run', action='store_true', help='run')
    args = parser.parse_args()
    isTrain = args.train
    isTest = args.test
    isRun = args.run

    if isTest:
        test_estimate()
    if isRun:
        main()
