'''
based on: https://github.com/openseg-group/OCNet.pytorch/tree/master/oc_module
'''
import torch
from torch import nn
import torch.nn.functional as F

class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            FReLU(self.key_channels)
        )
        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.value_channels),
            FReLU(self.value_channels)
        )
        self.f_query = self.f_value
        self.W = nn.Conv2d(in_channels=self.key_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        key = key.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.value_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = ((h*w)**(-.5)) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(value, sim_map)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w),
                                    mode='bilinear', align_corners=True)
        
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    out_channels,
                                                    scale)
        
class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.oc = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels,)   
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels+in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )
        
    def forward(self, feats):
        context = self.oc(feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output
