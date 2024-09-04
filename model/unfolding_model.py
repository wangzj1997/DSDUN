"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# import sys
# sys.path.append("/ssd/pyw/medical-reconstruction/Train_on_Dataset/calgary_campinas/MRI-Reconstruction/Method-1/models/TMI/unfoldingNet/kunet")
import torch
from torch import nn
from torch.nn import functional as F
from data.transforms import fft2c, ifft2c, complex_abs, fftshift, ifftshift
import numpy as np
import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append("../../../")
from KUNET import KUnet
from IUNET import IUnet

class NormIUnet(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.unet = IUnet(2, 2, chans, num_pools)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x):
        x, mean, std = self.norm(x)
        x = self.unet(x)
        x = self.unnorm(x, mean, std)
        return x

class NormKUnet(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.unet = KUnet(2, 2, chans, num_pools)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x):
        x, mean, std = self.norm(x)
        x = self.unet(x)
        x = self.unnorm(x, mean, std)
        return x

class unfoldingModel(nn.Module):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__()
        # self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob


        self.dtrans_i = nn.ModuleList()
        self.dtrans_k = nn.ModuleList()

        self.dc_weight_i = nn.ParameterList()
        self.dc_weight_k = nn.ParameterList()


        # self.dc_weight_i.append(nn.Parameter(torch.ones(1)))
        # self.dc_weight_k.append(nn.Parameter(torch.ones(1)))
        # self.fuse_weight_i.append(nn.Parameter(torch.ones(1)))
        # self.dtrans_i.append(NormUnet(4, 2, 32, 3))
        # self.dtrans_k.append(NormKUnet(2, 2, 12, 3))        

        self.iter_num = 6

        # self.prox_x = NormIUnet(40, 3)
        # self.prox_y = NormKUnet(32, 3)

        self.prox_x = nn.ModuleList(
            [NormIUnet(40, 3) for _ in range(self.iter_num)])
        self.prox_y = nn.ModuleList(
            [NormKUnet(32, 3) for _ in range(self.iter_num)])
        
        self.u = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.v = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.delta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.iter_num)])
        self.delta2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(self.iter_num)])        
        self.eta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.eta2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])       
        self.belta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        self.belta2 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(self.iter_num)])
        


    # def forward(self, image, mask, masked_kspace,image_pocs, masked_kspace_pocs):

    def forward(self, mask, under_kspace, under_image):

        # x_u = image.clone()
        # y_u = masked_kspace.clone().permute(0,3,1,2)
        # zero = torch.zeros(1, 1, 1, 1).to(masked_kspace)
        # print(mask.shape)o 


        x_u = under_image
        y_u = fftshift(under_kspace.permute(0,2,3,1), dim=[-3, -2]).permute(0,3,1,2)
    
        x = x_u
        y = y_u
        mask =  mask.unsqueeze(-1)
        u_list = []
        v_list = []
        for i in range(self.iter_num):
            x_t = x
            y_t = y

            if i != 0:
                ut = u_list[-1]
                vt = v_list[-1]
            else:
                ut = torch.zeros_like(x)
                vt = torch.zeros_like(y)
            
            ut = ut + self.u[i]*x + self.prox_x[i](ut + self.u[i]*x)
            vt = vt + self.v[i]*y + self.prox_y[i](vt + self.v[i]*y)

            x = x - self.delta1[i]*((ifft2c(fft2c(x.permute(0,2,3,1)) * mask).permute(0,3,1,2) - x_u) + \
                self.eta1[i]*(x - ut) \
                    + self.belta1[i]*(x - ifft2c(y_t.permute(0,2,3,1)).permute(0,3,1,2)))
            y = y - self.delta2[i]*(((y.permute(0,2,3,1) * mask).permute(0,3,1,2) - y_u) + \
                self.eta2[i]*(y - vt) \
                    + self.belta2[i]*(y - fft2c(x_t.permute(0,2,3,1)).permute(0,3,1,2)))

            u_list.append(ut)
            v_list.append(vt)



        # x_k = ifft2c(y.permute(0,2,3,1))
        x_k = fft2c(x.permute(0,2,3,1))
        # x = complex_abs(x.permute(0,2,3,1))
        return x.permute(0,2,3,1), x_k