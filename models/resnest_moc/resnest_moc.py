import torchvision
import torch
from time import perf_counter

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_oc_block import BaseOC_Module

from .resnest_backbone import Resnest50

def get_resnest50_moc(pretrained, num_classes=66):
    inplanes_scale_factor = 8
    
    
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 32
    
    backbone = Resnest50(pretrained)
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
        self.equalize_mid = nn.Sequential(
            nn.Conv2d(inplanes * 2, inplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inplanes),
        )
        self.equalize_out = nn.Sequential(
            nn.Conv2d(inplanes * 4, inplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inplanes),
        )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        (h,w) = input_shape
        mid, x = self.backbone(x)
        
        x = self.equalize_out(x)
        mid = self.equalize_mid(mid)

        x = F.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        mid = F.interpolate(mid, size=(h//4, w//4), mode='bilinear', align_corners=True)

        x = mid+x
        x = self.context(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        x = self.cls(x)
        
        return x