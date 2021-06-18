import torch
import types
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    mid = self.layer1(x)
    x = self.layer2(x)
    #x = self.layer3(mid)
    return mid, x

def Resnest50(pretrained):
    r"""ResNest-50 model from
    """
    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

    # load pretrained models, using ResNeSt-50 as an example
    net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
    net.forward = types.MethodType(forward, net)
    return net
