import os
import re
import random
import time
import numpy as np
import torch
import math
import glob
from time import perf_counter

from resnet_oc.resnet_oc import get_resnet34_oc
from resnet_oc_mod.resnet_oc_mod import get_resnet34_oc_mod
from val import val

from utils.mapillary import mapillary
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils.transform import Colorize
from val import val

import wandb

NUM_CLASSES = 66

def get_last_state(path):
    list_of_files = glob.glob(path + "/model-*.pth")
    max=0
    for file in list_of_files:
        num = int(re.search(r'model-(\d*)', file).group(1))  

        max = num if num > max else max 
    return max

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model):
    #Get training data
    assert os.path.exists(args.data_dir), "Error: datadir (dataset directory) could not be loaded"
    dataset_train = mapillary(args.data_dir, 'train', height=args.height, part=1)
    loader = DataLoader(dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)
    print('Loaded', len(loader), 'files')

    criterion = CrossEntropyLoss2d()

    savedir = args.save_dir
    savedir = f'./save/{savedir}'

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value.
        file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
        
        assert os.path.exists(file_resume), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(file_resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
    

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []

        model.train()
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())
            time_train.append(time.time() - start_time)

            if step % 100 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                wandb.log({"epoch":epoch, "loss":average}, step=(epoch-1)*18000/args.batch_size + step)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        scheduler.step()

        print('Val', val(args, model, part=0.05))
        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            filename = f'{savedir}/model-{epoch}.pth'
            torch.save({'model':model.state_dict(), 'opt':optimizer.state_dict(), 'epoch':epoch}, filename)
            print(f'save: {filename} (epoch: {epoch})')

    return

def main(args):
    #Make sure the saving directory for our model exists
    savedir = args.save_dir
    savedir = f'./save/{savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)


    #Init W&b to track results and get model
    config = dict(model=args.model, num_epochs=args.num_epochs, batch_size=args.batch_size, dataset='Mapillary')
    with wandb.init(project=args.project_name, config=config):
        print('Using', args.model)
        if args.model == 'resnet_oc':
            model = get_resnet34_oc()
        elif args.model == 'resnet_oc_lw':
            model = get_resnet34_oc_mod()
        else:
            raise NotImplementedError('Unknown model')
        
        model = torch.nn.DataParallel(model, device_ids=[1,2]).cuda()

        print("========== TRAINING ===========")
        train(args, model)
        print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    wandb.login()
    parser = ArgumentParser()
    
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['resnet_oc_lw', 'resnet_oc'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--num-epochs', type=int, default=10, help='If you use resume, give a number considering for how long it trained')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs-save', type=int, default=3, help='You can use this value to save model every X epochs')
    parser.add_argument('--save-dir', required=True, help='Where to save your model')
    parser.add_argument('--resume', action='store_true', help='Resumes from the last save from --savedir directory')
    parser.add_argument('--project-name', default='Junk', help='Project name for weights and Biases')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                                                                comma-separated list of integers only (default=0)')
    main(parser.parse_args())
