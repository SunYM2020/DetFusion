import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import build_loss
from ..registry import HEADS

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.Conv1 = nn.Conv2d(1,16,3,1,1)
        self.Relu  = nn.ReLU(inplace=True)
        
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16,16,3,1,1),
            'DenseConv2': nn.Conv2d(32,16,3,1,1),
            'DenseConv3': nn.Conv2d(48,16,3,1,1)
        })
        
    def forward(self, x):
        x = self.Relu(self.Conv1(x))
        for i in range(len(self.layers)):
            out = self.layers['DenseConv'+str(i+1)] ( x )
            x = torch.cat([x,out],1)
        return x

@HEADS.register_module
class FusionHeadBox(nn.Module):
    """Fusion RGB and TIR images """
    def __init__(self,
                 loss_grad=dict(
                     type='MaxGradLoss', loss_weight=1.0),
                 loss_pixel=dict(
                     type='DetcropPixelLoss', loss_weight=1.0)):
        super(FusionHeadBox, self).__init__()

        self.encoder_rgb = Encoder()
        self.encoder_tir = Encoder()
        self.fusion_conv = nn.Conv2d(128, 64, 1)
        # Decoder
        self.decoder_Conv2 = nn.Conv2d(128,64,3,1,1)
        self.decoder_Act2 = nn.ReLU(inplace=True)
        self.decoder_Conv3 =nn.Conv2d(64,32,3,1,1)
        self.decoder_Act3 = nn.ReLU(inplace=True)     
        self.decoder_Conv4 = nn.Conv2d(32,16,3,1,1)
        self.decoder_Act4 = nn.ReLU(inplace=True)
        self.decoder_Conv5 = nn.Conv2d(16,1,3,1,1)
        
        self.loss_max_grad = build_loss(loss_grad)
        self.loss_detcrop_pixel = build_loss(loss_pixel)

    def init_weights(self):
        normal_init(self.fusion_conv, std=0.01)

    def forward(self, im_r, im_i, att_rgb, att_tir):
        # extract rgb and tir feature 
        enc_rgb = self.encoder_rgb(im_r)
        enc_tir = self.encoder_tir(im_i)

        enc_fusion = torch.cat((enc_rgb, enc_tir), dim=1) 

        att_w = torch.max(att_rgb[0], att_tir[0])

        up_att_w = F.interpolate(att_w, scale_factor=4, mode='bilinear', align_corners=True)

        # image fusion decoder
        enc_fusion = enc_fusion + enc_fusion * up_att_w 

        de_x = self.decoder_Conv2(enc_fusion)
        de_x = de_x + de_x * up_att_w
        de_x = self.decoder_Act2(de_x)

        de_x = self.decoder_Conv3(de_x)
        de_x = de_x + de_x * up_att_w
        de_x = self.decoder_Act3(de_x)

        de_x = self.decoder_Conv4(de_x)
        de_x = de_x + de_x * up_att_w
        de_x = self.decoder_Act4(de_x)

        de_x = self.decoder_Conv5(de_x)

        return de_x

    def grad_loss(self,
             im_fusion,
             im_rgb,
             im_tir):
        losses = dict()
        losses['max_loss_grad'] = self.loss_max_grad(im_fusion, im_rgb, im_tir)
        return losses

    def pixel_loss(self,
             im_fusion,
             im_rgb,
             im_tir,
             detect_box):
        losses = dict()
        losses['detcrop_loss_pixel'] = self.loss_detcrop_pixel(im_fusion, im_rgb, im_tir, detect_box)
        return losses
