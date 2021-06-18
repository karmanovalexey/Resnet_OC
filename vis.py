import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, InterpolationMode, ToTensor, ToPILImage
from utils.transform import Relabel, ToLabel, Colorize
from torch.utils.data import Dataset
from tqdm import tqdm
from time import perf_counter
import re


from models.resnet_oc.resnet_oc import get_resnet34_oc
from models.resnet_moc.resnet_moc import get_resnet34_moc
from models.resnet_oc_lw.resnet_oc_lw import get_resnet34_oc_lw
from models.resnet_ocr.resnet_ocr import get_resnet34_ocr
from models.resnet_ocold.model import get_resnet34_base_oc_layer3

NUM_CLASSES = 66

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

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
        return get_resnet34_base_oc_layer3(NUM_CLASSES, pretrained)
    else:
        raise NotImplementedError('Unknown model')

def load_checkpoint(model_path):
    #Must load weights, optimizer, epoch and best value.
    file_resume = f'{model_path}'
    #file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

class Transform(object):
    def __init__(self, height=600):
        self.height = height
        pass

    def __call__(self, input):
        # do something to both images
        if self.height == 1080:
            input =  Resize((1080,1920), InterpolationMode.BILINEAR)(input)
        else:
            input =  Resize((self.height,self.height*2), InterpolationMode.BILINEAR)(input)
        
        input = ToTensor()(input)
        input = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(input)
        return np.array(input)

class Video(Dataset):
    def __init__(self, data_dir, height=600):
        self.data_dir = data_dir
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.data_dir)) for f in fn if f.endswith(".png")]
        self.filenames.sort()
        #self.filenames = self.filenames[:10]
        self.co_transform = Transform(height=height)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.co_transform(input=image)
        return image

    def __len__(self):
        return len(self.filenames)

def make_video(image_folder, fps):
    print('fps is', fps)
    video_name = image_folder + '/video.avi'
    print(video_name)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=natural_keys)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width,height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        text_to_write = str(int(fps))+' fps    (' + str(height) + ',' + str(width) +') res'
        img = cv2.putText(img,text_to_write,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    return


def main(args):
    savedir = args.save_dir
    savedir = f'{savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    os.chmod(savedir, 0o777)
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    data = Video(args.data_dir, height=args.height)
    loader = DataLoader(data, num_workers=4, batch_size=1, shuffle=False)

    model = get_model(args.model, False).to(device=args.device)
    # model = torch.nn.DataParallel(model)
    checkpoint = load_checkpoint(args.load_dir)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    color_transform = Colorize(NUM_CLASSES)
    time = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(loader)):
            images = images.to(device=args.device)

            torch.cuda.synchronize()
            t1 = perf_counter()

            if args.model == 'resnet_ocr':
                aux, outputs = model(images)
            else:
                outputs = model(images)

            torch.cuda.synchronize()
            t2 = perf_counter()

            time.append((t2 - t1)/images.shape[0])

            outputs = np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2)


            img = Image.fromarray(outputs)
            save_point = savedir + '/' + str(i) + '.png'
            img.save(save_point)
            os.chmod(save_point, 0o777)
    fps = 1./np.mean(time)


    make_video(savedir, fps)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Data to visualize')
    parser.add_argument('--model', choices=['resnet_oc_lw', 'resnet_oc', 'resnet_moc', 'resnet_ocr', 'resnet_ocold', 'resnest_moc'], help='Tell me what to use')
    parser.add_argument('--height', type=int, default=1080, help='Height of images to resize, nothing to add')
    parser.add_argument('--load-dir', required=True, help='Where to load your model from')
    parser.add_argument('--save-dir', required=True, help='Where to save output')
    parser.add_argument('--keep_fps', action='store_true', help='Whether to output video in a constant fps, or as the model gives predictions')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    main(parser.parse_args())