import torch
import argparse
import os
import glob
import random

import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
from utils.dataset import Cropper
from predict import Predictor, dataset_predict
from configs.config import Configurations
from models.MyNetworks.ESFNet import ESFNet
from utils.unpatchy import unpatchify
'''
    # instructions high-resolution images are saved in '--input'
    #              then we use Cropper, get patches and saved in '--input/image_patches/'
    #              use torch.utils.data.Dataset and torch.utils.data.DataLoader to load data
    #              get predictions by the pre-trained model
    #              save the output patches in '--output/patches'
    #              re-merge the output patches into high-resolution images and save them in '--output/remerge'
'''

def config_parser():

    parser = argparse.ArgumentParser(description='configurations')
    parser.add_argument('--gpu', type=int, default=0,
                        help='0 and 1 means gpu id, and -1 means cpu')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help='directory of input images, including images used to train and predict')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help='directory of output images, for predictions')
    parser.add_argument('--ckpt_path', type=str, default=os.path.join('.', 'checkpoint-best.pth'),
                        help='path to the checkpoint file, default name checkpoint-best.pth')
    # dataloader settings
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--pin_memory', type=bool, default=False,
                        help='When True, it will accelerate the prediction phase but with high CPU-Utilization, and it '
                             'will also allocate additional GPU-Memory')
    parser.add_argument('--nb_workers', type=int, default=1,
                        help='workers for DataLoader')
    # patches settings, some configs have already included in config.cfg
    parser.add_argument('--image_margin_color', type=list, default=[255, 255, 255],
                        help='the color of image margin color')
    parser.add_argument('--label_margin_color', type=list, default=[255, 255, 255],
                        help='the color of label margin color')

    return parser.parse_args()

def main():

    args = config_parser()
    config = Configurations()

    # for duplicating
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # model load the pre-trained weight, load ckpt once out of predictor
    model = ESFNet(config=config).to('cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu')
    ckpt = torch.load(args.ckpt_path, map_location='cuda:{}'.format(args.gpu) if args.gpu >=0 else 'cpu')
    model.load_state_dict(ckpt['state_dict'])

    # path for each high-resolution images -> crop -> predict -> merge
    source_image_pathes = glob.glob(os.path.join(args.input, '*.png'))
    for source_image in tqdm(source_image_pathes):
        # get high-resolution image name
        filename = source_image.split('/')[-1].split('.')[0]
        # cropper get patches and save to --input/patches
        c = Cropper(args=args, configs=config, predict=True)
        _, n_h, n_w = c.image_processor(image_path=source_image)
        my_dataset = dataset_predict(args=args)
        my_dataloader = data.DataLoader(my_dataset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=args.pin_memory, drop_last=False, num_workers=args.nb_workers)

        # predict using pre-trained network
        p = Predictor(args=args, model=model, dataloader_predict=my_dataloader)
        p.predict()
        # patches [total_size, C, H, W] p.patches tensor -> reshape -> [total_size, H, W, C]
        patches_tensor = torch.transpose(p.patches, 1, 3)
        patches_tensor = patches_tensor.view(n_h, n_w, -1)
        # merge and save the output image
        patches = patches_tensor.cpu().numpy()
        img = unpatchify(patches, n_h, n_w)
        img = Image.fromarray(img)
        save_path = os.path.join(args.output, 'remerge', filename+'.png')
        img.save(save_path)

if __name__ == '__main__':
    main()