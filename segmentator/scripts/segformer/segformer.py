from .mit import mit_b0
from .seg_head import SegFormerHead

import os
import torch
import torch.nn as nn

class Segformer(nn.Module):
    def __init__(self):
        super(Segformer, self).__init__()
        self.backbone = mit_b0()
        self.decode_head = SegFormerHead()

    def forward(self, x):
        sides = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x, sides)
        return {'aux':x}

def get_model(path):

    model = Segformer()

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device=device)

    assert os.path.exists(path), "No model checkpoint found"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    return model