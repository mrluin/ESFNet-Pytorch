import torch
import random
import glob
import os
import argparse
import time
import numpy as np
from PIL.Image import Image
from PIL import Image
import torch.utils.data as data
from models.MyNetworks import ESFNet
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.util import AverageMeter
from data.dataset import MyDataset

# mean and std for WHU Building dataset
# whether using depends on your use case: if your dataset is larger than WHU Building dataset, you could use the mean and
# std w.r.t. your own dataset, otherwise we recommend to use these mean and std.
rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)

class dataset_predict(data.Dataset):
    def __init__(self,
                 args):
        super(dataset_predict, self).__init__()

        self.args = args
        self.input_path = os.path.join(self.args.input, 'image_patches')
        self.data_list = glob.glob(os.path.join(self.input_path, '*'))

    def transform(self, image):

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std)

        return image

    def __getitem__(self, index):

        datas = Image.open(self.data_list[index])
        t_datas = self.transform(datas)
        # return filename for saving patch predictions.
        return t_datas, self.data_list[index]

    def __len__(self):

        return len(self.data_list)


class Predictor(object):
    def __init__(self,
                 args, model, dataloader_predict):
        super(Predictor, self).__init__()

        self.args = args
        self.model = model
        self.dataloader_predict = dataloader_predict
        self.patches = None

    def predict(self):

        self.model.eval()
        #predict_time = AverageMeter()
        #batch_time = AverageMeter()
        #data_time = AverageMeter()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, filenames) in enumerate(self.dataloader_predict, start=1):
                data = data.to(self.model.device, non_blocking = True)
                #data_time.update(time.time() - tic)
                pre_tic = time.time()
                logits = self.model(data)
                self._save_pred(logits, filenames)
                # here depends on the use case, logits -> mask
                if self.patches is None:
                    self.patches = torch.argmax(logits) * 255.
                else:
                    self.patches = torch.cat([self.patches, torch.argmax(logits)*255.], 0)
                #predict_time.update(time.time() - pre_tic)
                #batch_time.update(time.time() - tic)
                tic = time.time()

            #print("Predicting and Saving Done!\n"
            #      "Total Time: {:.2f}\n"
            #      "Data Time: {:.2f}\n"
            #      "Pre Time: {:.2f}"
            #      .format(batch_time._get_sum(), data_time._get_sum(), predict_time._get_sum()))
    def _save_pred(self, predictions, filenames):

        for index, map in enumerate(predictions):

            map = torch.argmax(map, dim=0)
            map = map * 255
            map = np.asarray(map.cpu(), dtype=np.uint8)
            map = Image.fromarray(map)
            # filename /0.1.png [0] 0 [1] 1
            filename = filenames[index].split('/')[-1].split('.')
            save_filename = filename[0]+'.'+filename[1]
            save_path = os.path.join(self.args.output, 'patches', save_filename+'.png')

            map.save(save_path)
