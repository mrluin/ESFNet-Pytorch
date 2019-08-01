import torch
import argparse
import os
import glob

import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.dataset import image_cropper
from predict import Predictor
from data.dataset import MyDataset
from models.MyNetworks.ESFNet import ESFNet

'''
    # instructions --input: path to high-resolution images used to predict
    #                       the path to patches is set as '--input/patches/' by default
    #              --output: path to output patches
    #                        path to re-merged images is set as '--output/remerge/' by default
    #              --ckpt_path: path to checkpoint file
    #              overlap part is obtained by averaging
'''

rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)

class dataset_predict(Dataset):
    def __init__(self, args):
        super(dataset_predict, self).__init__()

        self.images_path = os.path.join(args.input, 'patches')
        self.images_list = glob.glob(os.path.join(self.images_path, '*'))

    def transformations(self, images):

        images = TF.to_tensor(images)
        images = TF.normalize(images, mean=rgb_mean, std=rgb_std)

        return images

    def  __getitem__(self, index):

        images = Image.open(self.images_list[index])
        images = self.transformations(images)
        return images

    def __len__(self):

        return len(self.images_list)

def config_parser():

    parser = argparse.ArgumentParser(description='configurations')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help='directory of input images, including images used to train and predict')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help='directory of output images, for predictions')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join('.', 'checkpoint-best.pth'),
                        help='path to the checkpoint file, default name checkpoint-best.pth')


    # whether shuffle or not is set in DataLoader
    parser.add_argument('--size_x', type=int, default=224,
                        help='the width of image patches')
    parser.add_argument('--size_y', type=str, default=224,
                        help='the height of image patches')
    # step is set equal to patch size by default, that is the overlap is zero
    # overlap = size - step

    # dataloader settings
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--pin_memory', type=bool, default=False,
                        help='When True, it will accelerate the prediction phase but with high CPU-Utilization, and it '
                             'will also allocate additional GPU-Memory')
    parser.add_argument('--nb_workers', type=int, default=1,
                        help='workers for DataLoader')
    # patches settings
    parser.add_argument('--step_x', type=int, default=224,
                        help='the horizontal step of cropping images')
    parser.add_argument('--step_y', type=int, default=224,
                        help='the vertical step of cropping images')
    parser.add_argument('--image_margin_color', type=list, default=[255, 255, 255],
                        help='the color of image margin color')
    parser.add_argument('--label_margin_color', type=list, default=[255, 255, 255],
                        help='the color of label margin color')

    return parser.parse_args()

def main():


    args = config_parser()
    image_cropper(args)
    my_dataset = dataset_predict(args=args)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=args.batch_size, shuffle=False,
                               pin_memory=args.pin_memory, drop_last=False, num_workers=args.nb_workers)

    model = ESFNet(config=config)

    Predictor(args=args, model=model, dataloader_predict=my_dataloader)
    # when the predict phase is finish, get patches, then merge

if __name__ == '__main__':

    main()