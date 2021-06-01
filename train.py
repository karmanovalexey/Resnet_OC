import wandb
import torch
import os
import time
import glob
import re

from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from resnet_oc.resnet_oc import get_resnet34_oc
from resnet_moc.resnet_moc import get_resnet34_moc
from resnet_oc_lw.resnet_oc_lw import get_resnet34_oc_lw
from resnet_ocr.resnet_ocr import get_resnet34_ocr
from val import val, val_ocr
from utils.mapillary import mapillary
from utils.mapillary_pallete import MAPILLARY_LOSS_WEIGHTS

NUM_CLASSES = 66

class FocalLoss(torch.nn.Module):
    """
    based on https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.nn.functional.log_softmax(input,-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Loss(torch.nn.Module):
    def __init__(self, model_name, loss, ocr_coeff = 0.4):
        super().__init__()
        self.model_name = model_name
        if loss=='BCE':
            loss_weights = torch.Tensor(MAPILLARY_LOSS_WEIGHTS).cuda()
            self.loss = torch.nn.CrossEntropyLoss(weight=loss_weights) 
        elif loss=='Focal':
            self.loss = FocalLoss()
        self.k = ocr_coeff

    def forward(self, outputs, targets):
        if self.model_name == 'resnet_ocr':
            (out_aux, out) = outputs
            aux_loss = self.loss(out_aux, targets)
            out_loss = self.loss(out, targets)
            return self.k*aux_loss + out_loss
        else:
            return self.loss(outputs, targets)

def get_model(model_name, pretrained=False):
    if model_name == 'resnet_oc':
        return get_resnet34_oc(pretrained)
    elif model_name == 'resnet_oc_lw':
        return get_resnet34_oc_lw(pretrained)
    elif model_name == 'resnet_ocr':
        return get_resnet34_ocr(pretrained)
    elif model_name == 'resnet_moc':
        return get_resnet34_moc(pretrained)
    else:
        raise NotImplementedError('Unknown model')

def get_last_state(path):
    list_of_files = glob.glob(path + "/model-*.pth")
    max=0
    for file in list_of_files:
        num = int(re.search(r'model-(\d*)', file).group(1))  

        max = num if num > max else max 
    return max

def train(args):
    #Get training data
    assert os.path.exists(args.data_dir), "Error: datadir (dataset directory) could not be loaded"
    dataset_train = mapillary(args.data_dir, 'train', height=args.height, part=1.)
    loader = DataLoader(dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)
    print('Loaded', len(loader), 'batches')

    model = get_model(args.model, args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    criterion = Loss(args.model, args.loss)

    savedir = args.save_dir
    savedir = f'./save/{savedir}'

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(loader) * args.num_epochs)) ** 0.9)

    
    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value.
        file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
        
        assert os.path.exists(file_resume), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(file_resume)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['opt'])
        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
    
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []

        model.train()
        for step, (images, labels) in enumerate(tqdm(loader)):
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
                wandb.log({"epoch":epoch, "loss":average}, step=(epoch-1)*18000 + step*args.batch_size)
        
        scheduler.step()

        if args.model == 'resnet_ocr': 
            print('Val', val_ocr(args, model, part=0.05))
        else:
            print('Val', val(args, model, part=0.05))
        
        if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            filename = f'{savedir}/model-{epoch}.pth'
            torch.save({'model':model.state_dict(), 'opt':optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'epoch':epoch}, filename)
            print(f'save: {filename} (epoch: {epoch})')
    
    return

def main(args):
    savedir = args.save_dir
    savedir = f'./save/{savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    config = dict(model = args.model,
                    height = args.height,
                    epochs = args.num_epochs,
                    bs = args.batch_size,
                    pretrained = args.pretrained,
                    savedir = args.save_dir)
    
    log_mode = 'online' if args.wandb else 'disabled'
    with wandb.init(project=args.project_name, config=config, mode=log_mode, save_code=True):
        print('Run properties:', config)
        print("========== TRAINING ===========")
        train(args)
        print("========== TRAINING FINISHED ===========")

if __name__== '__main__':
    wandb.login()
    parser = ArgumentParser()
    
    parser.add_argument('--data-dir', required=True, help='Mapillary directory')
    parser.add_argument('--model', required=True, choices=['resnet_oc_lw', 'resnet_oc', 'resnet_ocr', 'resnet_moc'], help='Tell me what to train')
    parser.add_argument('--loss', default='BCE', help='Loss name, either BCE or Focal')
    parser.add_argument('--height', type=int, default=600, help='Height of images, nothing to add')
    parser.add_argument('--num-epochs', type=int, default=10, help='If you use resume, give a number considering for how long it trained')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-dir', help='Where to save your model')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained backbone')
    parser.add_argument('--resume', action='store_true', help='Resumes from the last save from --savedir directory')
    parser.add_argument('--wandb', action='store_true', help='Whether to log metrics to wandb')    
    parser.add_argument('--project-name', default='Junk', help='Project name for weights and Biases')
    parser.add_argument('--epochs-save', type=int, default=3, help='You can use this value to save model every X epochs')
    main(parser.parse_args())
