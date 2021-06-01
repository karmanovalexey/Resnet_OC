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

from resnet_oc.resnet_oc import get_resnet34_oc
from resnet_oc_lw.resnet_oc_lw import get_resnet34_oc_lw
from resnet_ocr.resnet_ocr import get_resnet34_ocr
from resnet_ocold.model import get_resnet34_base_oc_layer3

from utils.mapillary import mapillary
from utils.transform import Relabel, ToLabel, Colorize
from utils.iouEval import iouEval
from utils.mapillary_pallete import MAPILLARY_CLASSNAMES as classnames

NUM_CLASSES = 66

def get_model(model_name, pretrained=False):
    if model_name == 'resnet_oc':
        return get_resnet34_oc(pretrained)
    elif model_name == 'resnet_oc_lw':
        return get_resnet34_oc_lw(pretrained)
    elif model_name == 'resnet_ocr':
        return get_resnet34_ocr(pretrained)
    elif model_name == 'resnet_ocold':
        return get_resnet34_base_oc_layer3(66, pretrained)
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

def get_example(datadir, height):
    with open(datadir + '/val/1920_1080/images/_4SjTmQ-zn3XSv4D1-Tg4w.jpg', 'rb') as f:
        image = Image.open(f).convert('RGB')
    with open(datadir + '/val/1920_1080/labels/_4SjTmQ-zn3XSv4D1-Tg4w.png', 'rb') as f:
        label = Image.open(f).convert('P')

    input =  Resize((height,height*2), InterpolationMode.BILINEAR)(image)
    target = Resize((height,height*2), InterpolationMode.NEAREST)(label)

    input = ToTensor()(input)
    input = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(input)
    target = ToLabel()(target)
    target = Relabel(255, 65)(target)

    return input, target

def val(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'examples')

    val_loss = []
    time_val = []

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

        data = [[label, val] for (label, val) in zip(classnames[:-1], iou_classes.tolist())]
        table = wandb.Table(data=data, columns = ["label", "value"])
        wandb.log({"Predictions" : wandb.plot.bar(table, "label", "value", title="Predictions")})


        main_classes = np.mean([iou_classes[2],iou_classes[3],iou_classes[11],iou_classes[13],iou_classes[15],iou_classes[17],
                            iou_classes[19],iou_classes[23],iou_classes[24],iou_classes[29],iou_classes[30],iou_classes[31],
                            iou_classes[48],iou_classes[50],iou_classes[54],iou_classes[55]])

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':iouVal,
        'val_loss':np.mean(val_loss),
        'main_classes': main_classes})

        img_ex, lab_ex = get_example(args.data_dir, args.height)
        img_ex, lab_ex = img_ex.cuda(), lab_ex.cuda()
        outputs = model(img_ex.unsqueeze(0))

        examples = [np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(lab_ex.cpu().data)),0,2),]
        wandb.log({args.model:[wandb.Image(i) for i in examples]})


    return [iouVal.data, 1./np.mean(time_val), np.mean(val_loss)]

def val_ocr(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'files')

    val_loss = []
    aux_loss_epoch = []
    out_loss_epoch = []
    time_val = []

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

        val_iou, iou_classes = iouEvalVal.getIoU()

        main_classes = np.mean([iou_classes[2],iou_classes[3],iou_classes[11],iou_classes[13],iou_classes[15],iou_classes[17],
                            iou_classes[19],iou_classes[23],iou_classes[24],iou_classes[29],iou_classes[30],iou_classes[31],
                            iou_classes[48],iou_classes[50],iou_classes[54],iou_classes[55]])

        data = [[label, val] for (label, val) in zip(classnames[:-1], iou_classes.tolist())]
        table = wandb.Table(data=data, columns = ["label", "value"])
        wandb.log({"Predictions" : wandb.plot.bar(table, "label", "value", title="Predictions")})

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':val_iou,
        'val_loss':np.mean(val_loss),
        'aux_loss':np.mean(aux_loss_epoch),
        'out_loss':np.mean(out_loss_epoch),
        'main_classes': main_classes})
        
        img_ex, lab_ex = get_example(args.data_dir, args.height)
        img_ex, lab_ex = img_ex.cuda(), lab_ex.cuda()
        outputs_aux, outputs = model(img_ex.unsqueeze(0))
        
        examples = [np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(lab_ex.cpu().data)),0,2),
                np.moveaxis(np.array(color_transform(outputs_aux[0].cpu().max(0)[1].data.unsqueeze(0))),0,2)]

        wandb.log({args.model:[wandb.Image(i) for i in examples]})

    return [val_iou, 1./np.mean(time_val), np.mean(val_loss)]

def main(args):
    config = dict(model = args.model,
                    height = args.height,
                    bs = args.batch_size,
                    mode = 'Validation')

    log_mode = 'online' if args.wandb else 'disabled'
    with wandb.init(project=args.project_name, config=config, mode=log_mode):
        print('Run properties:', config)
        model = get_model(args.model, False)

        if not args.model=='resnet_ocold': model = torch.nn.DataParallel(model).cuda()
        checkpoint = load_checkpoint(args.model_path)
        model.load_state_dict(checkpoint['model'])  
        model = torch.nn.DataParallel(model).cuda()

        print("========== VALIDATING ===========")
        if args.model == 'resnet_ocr':
            print(val_ocr(args, model, part=0.5))
        else:
            print(val(args, model, part=0.5))
        print("========== VALIDATING FINISHED ===========")

if __name__ == '__main__':
    wandb.login()
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['resnet_oc_lw', 'resnet_oc', 'resnet_ocr', 'resnet_ocold'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-path', required=True, help='Where to load your model from')
    parser.add_argument('--wandb', action='store_true', help='Whether to log metrics to wandb')    
    parser.add_argument('--project-name', default='OC Results', help='Project name for weights and Biases')
    main(parser.parse_args())
