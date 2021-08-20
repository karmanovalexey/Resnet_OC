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
from models.resnet_m_base_oc.model import get_resnet34_base_oc_layer3
from models.segformer.segformer import Segformer
from models.deeplab.deeplab import Deeplab

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
    elif model_name == 'resnet_m_base_oc':
        return get_resnet34_base_oc_layer3(NUM_CLASSES, pretrained)
    elif model_name == 'segformer_b0':
        return Segformer()
    elif model_name == 'deeplab':
        return Deeplab()
    else:
        raise NotImplementedError('Unknown model')

def load_checkpoint(model_path):
    file_resume = f'{model_path}'
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

class Transform(object):
    def __init__(self, height=600, norm=True):
        self.height = height
        self.norm = norm
        pass

    def __call__(self, input, size):
        # do something to both images
        # if self.height == 1080:
        #     input =  Resize((1080,1920), InterpolationMode.BILINEAR)(input)
        # else:
        #     input =  Resize((self.height,self.height*2), InterpolationMode.BILINEAR)(input)
        
        input = ToTensor()(input)

        if self.norm:
            input = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)

        return np.array(input)

class Video(Dataset):
    def __init__(self, args):
        self.data_dir = args.data_dir
        print(self.data_dir)
        if args.dataset=='Mapillary':
            self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.data_dir)) for f in fn if f.endswith(".jpg")]
        elif args.dataset=='Kitti':
            self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.data_dir)) for f in fn if f.endswith(".png")]
        print(len(self.filenames))
        self.filenames.sort()
        self.filenames = self.filenames
        if args.model=='resnet_ocr':
            self.co_transform = Transform(height=args.height, norm=False)
        else:
            self.co_transform = Transform(height=args.height, norm=True)


    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.co_transform(input=image, size=image.size)
        return image, filename

    def __len__(self):
        return len(self.filenames)

def make_video(image_folder, fps, vis_fps):
    print('fps is', fps)
    video_name = image_folder + '/video.avi'
    print(video_name)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=natural_keys)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 24, (width,height))

    if vis_fps:
        for image in images: # uncomment for fps visualization
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
    
    data = Video(args)
    loader = DataLoader(data, num_workers=4, batch_size=1, shuffle=False)

    model = get_model(args.model, False).to(device=args.device)
    if args.model == 'resnet_ocr':
        model = torch.nn.DataParallel(model)
    checkpoint = load_checkpoint(args.load_dir)
    
    if args.model == 'segformer_b0':
        model.load_state_dict(checkpoint['state_dict'])
    else:    
        model.load_state_dict(checkpoint['model'])

    model.eval()
    color_transform = Colorize(NUM_CLASSES)
    time = []
    with torch.no_grad():
        for i, (images, filename) in enumerate(tqdm(loader)):
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

            if args.colorize:
                outputs = np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2)
                save_img = Image.fromarray(outputs)
            else:
                output = outputs.argmax(1)
                pred = np.asarray(output.cpu(), dtype=np.uint8)
                save_img = Image.fromarray(pred[0])
            
            if args.dataset=='Mapillary':
                save_point = savedir + '/' + filename[0][42:-4] + '.png'
            elif args.dataset=='Kitti':
                save_point = savedir + '/' + filename[0][17:-4] + '.png'
            save_img.save(save_point, 'PNG')
            os.chmod(save_point, 0o777)

    fps = 1./np.mean(time)
    print(fps)
    # make_video(savedir, fps, args.vis_fps)

    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Data to visualize')
    parser.add_argument('--dataset', choices=['Mapillary', 'Kitti'], help='What Dataset')
    parser.add_argument('--model', choices=['deeplab', 'resnet_oc_lw', 'resnet_oc', 'resnet_moc', 'resnet_ocr', 'resnet_m_base_oc', 'resnest_moc', 'segformer_b0'], help='Tell me what to use')
    parser.add_argument('--height', type=int, default=1080, help='Height of images to resize, nothing to add')
    parser.add_argument('--load-dir', required=True, help='Where to load your model from')
    parser.add_argument('--save-dir', required=True, help='Where to save output')
    parser.add_argument('--vis_fps', action='store_true', help='Whether to visualize fps')
    parser.add_argument('--colorize', action='store_true', help='Whether to colorize images or save them as single channeled')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    main(parser.parse_args())