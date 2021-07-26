from .resnet import ResNetV1c
from .aspp_head import ASPPHead

import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode

class Deeplab(nn.Module):
    def __init__(self):
        super(Deeplab, self).__init__()
        self.backbone = ResNetV1c()
        self.decode_head = DepthwiseSeparableASPPHead(dilations=(1, 12, 24, 36))

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        x = Resize((1080,1920), InterpolationMode.BILINEAR)(x)
        return x