import os
import sys
sys.path.append(os.path.abspath('..'))
from configs.config import MyConfiguration
import glob
from tqdm import tqdm
from scipy import misc
import numpy as np
import re
import functools

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def _get_sum(self):
        return self.sum


def cropped_dataset(config, subset):

    assert subset == 'train' or subset == 'val' or subset == 'test'
    dataset_path = os.path.join(config.root_dir, subset, config.data_folder_name)
    target_path = os.path.join(config.root_dir, subset, config.target_folder_name)

    new_dataset_path = dataset_path.replace('512', '256')
    new_target_path = target_path.replace('512', '256')
    #print(dataset_path)
    #print(target_path)

    data_paths = glob.glob(os.path.join(dataset_path, '*.tif'))
    for data_path in tqdm(data_paths):
        target_path = data_path.replace(config.data_folder_name, config.target_folder_name)

        filename = data_path.split('/')[-1].split('.')[0]

        data = misc.imread(data_path)
        target = misc.imread(target_path)

        h,w = config.original_size, config.original_size

        subimgcounter = 1
        stride = config.cropped_size-config.overlapped
        for h_pixel in range(0, h-config.cropped_size+1, stride):
            for w_pixel in range(0, w-config.cropped_size+1, stride):
                new_data = data[h_pixel:h_pixel+config.cropped_size, w_pixel:w_pixel+config.cropped_size, :]
                new_target = target[h_pixel:h_pixel+config.cropped_size, w_pixel:w_pixel+config.cropped_size]

                data_save_path = os.path.join(new_dataset_path, filename+'.{}.tif'.format(subimgcounter))
                target_save_path = data_save_path.replace(config.data_folder_name, config.target_folder_name)

                misc.imsave(data_save_path, new_data)
                misc.imsave(target_save_path, new_target)
                subimgcounter += 1
