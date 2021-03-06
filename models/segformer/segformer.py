from .mit import mit_b0
from .seg_head import SegFormerHead

import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode

class Segformer(nn.Module):
    def __init__(self):
        super(Segformer, self).__init__()
        self.backbone = mit_b0()
        self.decode_head = SegFormerHead()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.backbone(x)
        x = self.decode_head(x)
        x = Resize((h,w), InterpolationMode.BILINEAR)(x)
        return x

def main(path):

    model = Segformer()

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device=args.device)

    assert os.path.exists(path), "No model checkpoint found"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    return model