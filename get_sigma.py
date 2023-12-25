import numpy as np
import matplotlib.pyplot as plt

class GetSigma():
    def __init__(self, method):
        """
        method = 0: MultiModel
        method = 1: MultiTransform
        """
        self.method = method
    def get(self, data, show=0):
        if self.method == 1:
            self.detransform(data)
            
        data_u = [data[0], data[2], data[4], data[6]]
        data_v = [data[1], data[3], data[5], data[7]]
        
        sigma_u = np.sqrt(
            (np.square(data_u[0] - data_u[1]) + \
             np.square(data_u[0] - data_u[2]) + \
             np.square(data_u[0] - data_u[3])) / 2
        )
        
        sigma_v = np.sqrt(
            (np.square(data_v[0] - data_v[1]) + \
             np.square(data_v[0] - data_v[2]) + \
             np.square(data_v[0] - data_v[3])) / 2
        )
        
        if show == 1:
            plt.figure(figsize=(12,8))
            plt.subplot(1, 2, 1)
            plt.title('uncertainty_u')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_u)
            plt.colorbar(fraction=0.05)
            
            plt.subplot(1, 2, 2)
            plt.title('uncertainty_v')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_v)
            plt.colorbar(fraction=0.05)
            
        return sigma_u, sigma_v
    
    def detransform(self, data):
        
        data[2] = np.flip(data[2], 0)
        data[3] = np.negative(np.flip(data[3], 0))
        
        data[4] = np.rot90(np.negative(data[4]), k=2, axes=(0, 1))
        data[5] = np.rot90(np.negative(data[5]), k=2, axes=(0, 1))

        data[6] = np.negative(np.flip(data[6], 1))
        data[7] = np.flip(data[7], 1)
    
