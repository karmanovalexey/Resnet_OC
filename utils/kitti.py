import numpy as np
import os
import random

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, InterpolationMode
from torchvision.transforms import ToTensor, ToPILImage
from .transform import Relabel, ToLabel, Colorize

from PIL import Image, ImageOps
from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def image_path(root, name):
    return os.path.join(root, f'{name}')

def get_cur_track(lens, index):
    track = 0
    for l in lens:
        if (index - l) < 0:
            return track, index
        else:
            index = index - l
            track += 1
    return track, index

class MyCoTransform(object):
    def __init__(self, augment=True, height=None):
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        if self.height != None:
            if self.height == 1080:
                input =  Resize((1080,1920), InterpolationMode.BILINEAR)(input)
                target = Resize((1080,1920), InterpolationMode.NEAREST)(target)
            else:
                input =  Resize((self.height,self.height*2), InterpolationMode.BILINEAR)(input)
                target = Resize((self.height,self.height*2), InterpolationMode.NEAREST)(target)

        if (self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        input = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(input)
        target = ToLabel()(target)
        target = Relabel(255, 65)(target)

        return input, target

class kitti(Dataset):
    def __init__(self, root, subset='val', height=None, track_nums=[3]):
        assert ((1 not in track_nums) and (8 not in track_nums) and all(i <= 10 for i in track_nums)) #tracks can repeat

        self.track_nums = sorted(track_nums)
        self.image_tracks = [root + "/data_2d_raw/2013_05_28_drive_00%02d_sync/image_00/data_rect/" % i for i in self.track_nums]
        self.label_tracks = [root + "/data_2d_semantics/train/2013_05_28_drive_00%02d_sync/semantic/" % i for i in self.track_nums]


        self.images = []
        self.labels = []

        # as not all images have labels, we firstly run thorugh files in labels directory 
        for img_tr, lab_tr in zip(self.image_tracks, self.label_tracks):
            files = next(os.walk(lab_tr))[2]
            self.images.append(sorted([img_tr + i for i in files]))
            self.labels.append(sorted([lab_tr + i for i in files]))

        if subset=='train':
            self.co_transform = MyCoTransform(augment=True, height=height)
        elif subset=='val':
            self.co_transform = MyCoTransform(augment=False, height=height)


    def __getitem__(self, index):
        pos = get_cur_track([len(i) for i in self.images], index)
        image_path = self.images[pos[0]][pos[1]]
        label_path = self.labels[pos[0]][pos[1]]
        # img = self.images[]
        # with open(, 'rb') as f:
        #     image = load_image(f).convert('RGB')
        # with open(, 'rb') as f:
        #     label = load_image(f).convert('P')

        # image, label = self.co_transform(input=image, target=label)

        return image_path, label_path

    def __len__(self):
        return sum(len(i) for i in self.labels)


if __name__=='__main__':
    kit = kitti('/datasets/KITTI-360')
    for item in kit:
        print(item)