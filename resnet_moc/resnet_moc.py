import torchvision
import torch
from time import perf_counter

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_oc_block import BaseOC_Module

from .resnet_backbone import Resnet34

def get_resnet34_moc(pretrained, num_classes=66):

    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4
    
    
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 128
    
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
        (h,w) = input_shape
        #print('input: ', x.shape)
        # torch.cuda.synchronize()
        # t1 = perf_counter()
        x = self.backbone(x)
        x = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        # torch.cuda.synchronize()
        # t2 = perf_counter()
        # print('backbone', t2-t1)
        # print('backbone: ', x.shape)
        x = self.context(x)
        #print('context: ', x.shape)

        # torch.cuda.synchronize()
        # t3 = perf_counter()
        # print('context', t3-t2)
        # print('interpol: ', x.shape)
        # torch.cuda.synchronize()
        # t4 = perf_counter()
        # print('interpolate', t4-t3)
        x = self.cls(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        # torch.cuda.synchronize()
        # t5 = perf_counter()
        # print('cls', t5-t4)
        # print('cls: ', x.shape)
        
        return x