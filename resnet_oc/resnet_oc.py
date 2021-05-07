import torchvision
import torch

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from time import perf_counter

from .base_oc_block import BaseOC_Module

from .resnet_backbone import Resnet34

def get_resnet34_oc(pretrained, num_classes=66):

    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4
    
    
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 512
    
    backbone = Resnet34(pretrained)
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


class ResNet_Base_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Base_OC, self).__init__()
        self.backbone = backbone
        self.context = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            BaseOC_Module(in_channels=outplanes, 
                          out_channels=outplanes, 
                          key_channels=outplanes // 2, 
                          value_channels=outplanes // 2, 
                          dropout=0.05, 
                          sizes=([1]))
            )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        #print('input: ', x.shape)
        # torch.cuda.synchronize()
        # t1 = perf_counter()

        x = self.backbone(x)

        # torch.cuda.synchronize()
        # t2 = perf_counter()
        # print('backbone', t2-t1)
        # print('backbone: ', x.shape)
        x = self.context(x)

        # torch.cuda.synchronize()
        # t3 = perf_counter()
        # print('context', t3-t2)
        # print('context: ', x.shape)
        x = self.cls(x)
        # torch.cuda.synchronize()
        # t4 = perf_counter()
        # print('cls', t4-t3)
        # print('cls: ', x.shape)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        # torch.cuda.synchronize()
        # t5 = perf_counter()
        # print('interpolate', t5-t4)
        #print('output: ', x.shape)
        
        return x