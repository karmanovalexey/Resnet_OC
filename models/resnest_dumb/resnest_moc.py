import torchvision
import torch
from time import perf_counter

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_oc_block import BaseOC_Module

from .resnest_backbone import Resnest50

def get_resnest50_dumb(pretrained, num_classes=66):
    inplanes_scale_factor = 1
    
    
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 32
    
    backbone = Resnest50(pretrained)
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


class ResNet_Base_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Base_OC, self).__init__()
        self.backbone = backbone
        self.up1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inplanes // 2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(inplanes // 2, inplanes // 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inplanes // 4),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(inplanes // 4, inplanes // 8, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inplanes // 8),
        )
        self.cls = nn.Conv2d(inplanes // 8, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        (h,w) = input_shape
        x = self.backbone(x)
        self.up1(x)
        x = F.interpolate(x, size=(h // 4,w // 4), mode='bilinear', align_corners=True)
        self.up2(x)
        x = F.interpolate(x, size=(h // 2,w // 2), mode='bilinear', align_corners=True)
        self.up3(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        x = self.cls(x)
        
        return x