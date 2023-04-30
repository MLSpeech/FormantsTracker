import torch
import torch.nn as nn
import numpy as np
import scipy


# import sys
# sys.path.append(".")



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



def get_smoothing_kernel(kernel_size,kernel_sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 1
    kernel = scipy.ndimage.gaussian_filter(kernel, sigma=kernel_sigma)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    return kernel
