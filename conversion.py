import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser


def main(args):
    if not os.path.exists(args.new_dir):
        os.makedirs(args.new_dir)
    
    #getting all filenames in a given folder to process them
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.data_dir)) for f in fn]
    filenames.sort()

    #loading config in pandas dataframe
    instruction = pd.read_csv(args.config)

    #getting necessary data from config:
    #classnumbers
    map_class = instruction['Mapillary_classnum'].dropna().tolist()

    #new classnumbers
    mku_class = instruction['Super_num'].dropna().tolist()

    #palette to paint new images
    palette = instruction['Palette'].dropna().tolist()
    palette = [list(map(int, i.split(':')[1].split('-'))) for i in palette]
    palette = [item for sublist in palette for item in sublist]

    #building correspondance dict to define relations between classes
    correspond = [(int(i[0]),int(i[1])) for i in list(zip(map_class, mku_class))]

    #going through images
    for image_path in tqdm(filenames):
        image_name = args.new_dir + '/' + image_path.split('/')[-1]
        image = np.array(Image.open(image_path))
        new_image = np.zeros(image.shape)
        
        #transfering labels class by class
        for cls in correspond:
            new_image[image==cls[0]] = cls[1]

        #saving colored images
        new_image = Image.fromarray(new_image.astype(np.uint8))
        new_image.putpalette(palette)
        new_image.save(image_name, "PNG")
        
        #adjusting permissions
        os.chmod(image_name, 666)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data-dir', required=True, help='label images directory')
    parser.add_argument('--new-dir', required=True, help='Where to save new dataset')
    parser.add_argument('--config', required=True, help='CSV file consisting data for conversion')

    main(parser.parse_args())