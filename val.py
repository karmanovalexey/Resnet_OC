import os
import numpy as np
import torch
import wandb

from tqdm import tqdm
from time import perf_counter
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from resnet_oc.resnet_oc import get_resnet34_oc
from resnet_oc_lw.resnet_oc_lw import get_resnet34_oc_lw
from resnet_ocr.resnet_ocr import get_resnet34_ocr

from utils.mapillary import mapillary
from utils.transform import Colorize
from utils.iouEval import iouEval

NUM_CLASSES = 66

def get_model(model_name, pretrained=False):
    if model_name == 'resnet_oc':
        return get_resnet34_oc(pretrained)
    elif model_name == 'resnet_oc_lw':
        return get_resnet34_oc_lw(pretrained)
    elif model_name == 'resnet_ocr':
        return get_resnet34_ocr(pretrained)
    else:
        raise NotImplementedError('Unknown model')

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

def load_checkpoint(model_path):
    #Must load weights, optimizer, epoch and best value.
    file_resume = f'{model_path}'
    #file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

def val(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'examples')

    val_loss = []
    time_val = []
    val_iou = []

    criterion = CrossEntropyLoss2d()
    model.eval()
    iouEvalVal = iouEval(NUM_CLASSES)
    color_transform = Colorize(NUM_CLASSES)
    
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(loader_val)):

            images = images.cuda()
            labels = labels.cuda()

            torch.cuda.synchronize()
            t1 = perf_counter()

            outputs = model(images)

            torch.cuda.synchronize()
            t2 = perf_counter()

            loss = criterion(outputs, labels[:, 0])

            val_loss.append(loss.data.item())
            time_val.append((t2 - t1)/images.shape[0]) #time

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels.data) #IOU
            iouVal, iou_classes = iouEvalVal.getIoU()
            val_iou.append(iouVal)

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':np.mean(val_iou),
        'val_loss':np.mean(val_loss)})

        examples = [np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(labels[0].cpu().data)),0,2)]
        wandb.log({args.model:[wandb.Image(i) for i in examples]})


    return [np.mean(val_iou), 1./np.mean(time_val), np.mean(val_loss)]

def val_ocr(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'files')

    val_loss = []
    aux_loss_epoch = []
    out_loss_epoch = []
    time_val = []
    val_iou = []

    criterion = CrossEntropyLoss2d()
    model.eval()
    iouEvalVal = iouEval(NUM_CLASSES)
    color_transform = Colorize(NUM_CLASSES)
    
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(loader_val)):

            images = images.cuda()
            labels = labels.cuda()

            torch.cuda.synchronize()
            t1 = perf_counter()

            out_aux, out = model(images)

            torch.cuda.synchronize()
            t2 = perf_counter()
            
            aux_loss = criterion(out_aux, labels[:, 0])
            out_loss = criterion(out, labels[:, 0])
            loss = aux_loss + out_loss

            val_loss.append(loss.data.item())
            aux_loss_epoch.append(aux_loss.data.item())
            out_loss_epoch.append(out_loss.data.item())
            time_val.append((t2 - t1)/images.shape[0]) #time

            iouEvalVal.addBatch(out.max(1)[1].unsqueeze(1).data, labels.data) #IOU
            iouVal, iou_classes = iouEvalVal.getIoU()
            val_iou.append(iouVal)

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':np.mean(val_iou),
        'val_loss':np.mean(val_loss),
        'aux_loss':np.mean(aux_loss_epoch),
        'out_loss':np.mean(out_loss_epoch)})

        examples = [np.moveaxis(np.array(color_transform(out[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(labels[0].cpu().data)),0,2),
                np.moveaxis(np.array(color_transform(out_aux[0].cpu().max(0)[1].data.unsqueeze(0))),0,2)]
        wandb.log({args.model:[wandb.Image(i) for i in examples]})

    return [np.mean(val_iou), 1./np.mean(time_val), np.mean(val_loss)]

def main(args):

    
    config = dict(model = args.model,
                    height = args.height,
                    bs = args.batch_size,
                    mode = 'Validation')
    
    with wandb.init(project=args.project_name, config=config):
        print('Run properties:', config)
        model = get_model(args.model, False)
        model = torch.nn.DataParallel(model).cuda()

        checkpoint = load_checkpoint(args.model_path)
        model.load_state_dict(checkpoint['model'])

        print("========== VALIDATING ===========")
        if args.model == 'resnet_ocr':
            val_ocr(args, model, part=0.5)
        else:
            val(args, model, part=0.5)
        print("========== VALIDATING FINISHED ===========")

if __name__ == '__main__':
    wandb.login()
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['resnet_oc_lw', 'resnet_oc', 'resnet_ocr'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-path', required=True, help='Where to load your model from')
    parser.add_argument('--project-name', default='OC Results', help='Project name for weights and Biases')
    main(parser.parse_args())
