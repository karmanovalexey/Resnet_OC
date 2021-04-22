import torchvision

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .ocr import OCR_Module

from .resnet_backbone import Resnet34

NUM_CLASSES = 66

def get_resnet34_ocr(num_classes=NUM_CLASSES):

    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4
    
    
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 512
    
    backbone = Resnet34()
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


class ResNet_Base_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Base_OC, self).__init__()
        self.backbone = backbone
        self.ocr = OCR_Module(in_channels=inplanes, 
                          out_channels=outplanes, 
                          key_channels=outplanes // 2, 
                          mid_channels=outplanes // 2,
                          num_classes=NUM_CLASSES, 
                          dropout=0.05, 
                          sizes=([1]))
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        #print('input: ', x.shape)
        x = self.backbone(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        aux, x = self.ocr(x)
        #print('output: ', x.shape)
        
        return aux, x