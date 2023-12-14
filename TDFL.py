"""
note: version adaptation for PyTorch > 1.7.1
date: 16th August 2023
"""
import torch
import torch.nn as nn
import torch.fft



class ThreeDFL(nn.Module):

    def __init__(self, train=False, log_matrix=True):
        super(ThreeDFL, self).__init__()
        self.log_matrix = log_matrix
        self.train = train
    def data2spec(self, x):
        """
        three times fft to calculate 3D fft
        torch.fft is only available after PyTorch version 1.7.1
        if your Pytorch version < 1.7.1, please use the rfft
        """
        freq = torch.fft.fft(torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2),dim=-3)
        #  freq = torch.rfft(y, 3, onesided=False, normalized=True)  # if PyTorch < 1.7.1
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq.unsqueeze(1)

    def loss_formulation(self, recon_freq, real_freq):

        pred = recon_freq[..., 0]**2 + recon_freq[..., 1]**2
        real = real_freq[..., 0]**2 + real_freq[..., 1]**2
        if self.log_matrix:
            pred = torch.log(pred + 1.0)
            real = torch.log(real + 1.0)

        loss = (pred-real)**2
        if self.train:
            return torch.mean(loss)
        else:
            return torch.mean(loss).cpu().detach().numpy()

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
        Out:
            numpy ndarray
        Example:
            # definition
            3df_loss = ThreeDFL()
            # calculate
            pred = network(input)
            loss = 3df_loss(pred, target)
        """
        # calculate
        pred_freq = self.data2spec(pred)
        target_freq = self.data2spec(target)

        # calculate loss
        return self.loss_formulation(pred_freq, target_freq)


import numpy as np

def gaussian_3d(T, H, W, mean, std):
    t, h, w = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing='ij')
    dist = ((t - mean[0]) / std[0])**2 + ((h - mean[1]) / std[1])**2 + ((w - mean[2]) / std[2])**2
    gauss = np.exp(-0.5 * dist)
    return gauss

T = 10  # 时间维度
H = 5   # 高度维度
W = 8   # 宽度维度
mean = [T//2, H//2, W//2]  # 均值
std = [2, 1, 1]  # 标准差

gaussian_array = gaussian_3d(T, H, W, mean, std)
print(gaussian_array)