import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES

from math import exp
import numpy as np


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


@LOSSES.register_module
class MaxGradLoss(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(MaxGradLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = Sobelxy()
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, *args, **kwargs):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """        
        rgb_grad = self.sobelconv(im_rgb)
        tir_grad = self.sobelconv(im_tir)
        max_grad_joint = torch.max(rgb_grad, tir_grad)
        generate_img_grad = self.sobelconv(im_fusion)

        sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
        loss_grad = self.loss_weight * sobel_loss

        return loss_grad


@LOSSES.register_module
class DetcropPixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(DetcropPixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, detect_box, *args, **kwargs):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """

        mask = torch.zeros_like(im_fusion)
        ones_mask = torch.ones_like(im_fusion)
        for k in range(len(detect_box)):
            if k == len(detect_box)-1:
                for i in range(len(detect_box[k])):
                    for j in range(len(detect_box[k][i])):
                        if int(detect_box[k][i][j])<0:
                            detect_box[k][i][j]=0
                    if int(detect_box[k][i][0])>640:      
                        detect_box[k][i][0]=640
                    if int(detect_box[k][i][2])>640:
                        detect_box[k][i][2]=640          
                    if int(detect_box[k][i][1])>512:
                        detect_box[k][i][1]=512
                    if int(detect_box[k][i][3])>512:
                        detect_box[k][i][3]=512
                    mask_temp = mask[k:,:,int(detect_box[k][i][1]):int(detect_box[k][i][3]),int(detect_box[k][i][0]):int(detect_box[k][i][2])].data.copy_(ones_mask[k:,:,int(detect_box[k][i][1]):int(detect_box[k][i][3]),int(detect_box[k][i][0]):int(detect_box[k][i][2])])
            else:
                for i in range(len(detect_box[k])):
                    for j in range(len(detect_box[k][i])):
                        if int(detect_box[k][i][j])<0:
                            detect_box[k][i][j]=0
                    if int(detect_box[k][i][0])>640:      
                        detect_box[k][i][0]=640
                    if int(detect_box[k][i][2])>640:
                        detect_box[k][i][2]=640          
                    if int(detect_box[k][i][1])>512:
                        detect_box[k][i][1]=512
                    if int(detect_box[k][i][3])>512:
                        detect_box[k][i][3]=512
                    mask_temp = mask[k:k+1,:,int(detect_box[k][i][1]):int(detect_box[k][i][3]),int(detect_box[k][i][0]):int(detect_box[k][i][2])].data.copy_(ones_mask[k:k+1,:,int(detect_box[k][i][1]):int(detect_box[k][i][3]),int(detect_box[k][i][0]):int(detect_box[k][i][2])])

        pixel_max = torch.max(im_rgb, im_tir)
        mask_fusion = torch.where(mask>0, im_fusion, mask)
        mask_pixel = torch.where(mask>0, pixel_max.detach(), mask)

        pixel_mean = (im_rgb + im_tir)/2.0
        bg_mask = 1 - mask                                            
        bg_fusion = torch.where(bg_mask>0, im_fusion, bg_mask)
        bg_pixel = torch.where(bg_mask>0, pixel_mean.detach(), bg_mask) 

        mask_loss = self.L1_loss(mask_fusion, mask_pixel)
        bg_loss = self.L1_loss(bg_fusion, bg_pixel)
        pixel_loss = self.loss_weight * (mask_loss + bg_loss)

        return pixel_loss
