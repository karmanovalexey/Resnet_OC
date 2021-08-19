import os
import numpy as np
import torch
import wandb
from PIL import Image

from tqdm import tqdm
from time import perf_counter
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, InterpolationMode, ToTensor, ToPILImage

from models.resnet_oc.resnet_oc import get_resnet34_oc
from models.resnet_moc.resnet_moc import get_resnet34_moc
from models.resnet_oc_lw.resnet_oc_lw import get_resnet34_oc_lw
from models.resnet_ocr.resnet_ocr import get_resnet34_ocr
from models.resnet_ocold.model import get_resnet34_base_oc_layer3
from models.segformer.segformer import Segformer

from utils.mapillary import mapillary
from utils.transform import Relabel, ToLabel, Colorize
from utils.iouEval import iouEval
from utils.loss import Loss
from utils.mapillary_pallete import MAPILLARY_CLASSNAMES as classnames

NUM_CLASSES = 66

def get_model(model_name, pretrained=False):
    if model_name == 'resnet_oc':
        return get_resnet34_oc(pretrained)
    elif model_name == 'resnet_oc_lw':
        return get_resnet34_oc_lw(pretrained)
    elif model_name == 'resnet_ocr':
        return get_resnet34_ocr(pretrained)
    elif model_name == 'resnet_moc':
        return get_resnet34_moc(pretrained)
    elif model_name == 'resnet_ocold':
        return get_resnet34_base_oc_layer3(66, pretrained)
    elif model_name == 'segformer_b0':
        return Segformer()
    else:
        raise NotImplementedError('Unknown model')

def load_checkpoint(model_path):
    #Must load weights, optimizer, epoch and best value.
    file_resume = f'{model_path}'
    #file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

def inf(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'files')

    time_val = []

    model.eval()
    iouEvalVal = iouEval(NUM_CLASSES, device=args.device)
    color_transform = Colorize(NUM_CLASSES)
    
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(loader_val)):

            images = images.to(device=args.device)
            labels = labels.to(device=args.device)

            torch.cuda.synchronize()
            t1 = perf_counter()

            outputs = model(images)
            out_aux, out = outputs

            torch.cuda.synchronize()
            t2 = perf_counter()

            time_val.append((t2 - t1)/images.shape[0]) #time

    return 

def main(args):
    config = dict(model = args.model,
                    height = args.height,
                    mode = 'Inference')
    
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    print('Run properties:', config)
    model = get_model(args.model, False).to(device=args.device)

    checkpoint = load_checkpoint(args.model_path)

    model.load_state_dict(checkpoint['state_dict'])

    print("========== VALIDATING ===========")
    print(inf(args, model, part=0.5))
    print("========== VALIDATING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['resnet_oc_lw', 'resnet_oc', 'resnet_moc', 'resnet_ocr', 'resnet_ocold', 'segformer_b0'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--model-path', required=True, help='Where to load your model from')
    parser.add_argument('--save-path', required=True, help='Where to save images')
    main(parser.parse_args())
