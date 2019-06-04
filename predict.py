import torch
import glob
import os
import argparse
import time
import numpy as np
from PIL.Image import Image
import torch.utils.data as data
from models.MyNetworks import ESFNet
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from configs.config import MyConfiguration
from utils.util import AverageMeter
from data.dataset import MyDataset

# mean and std for WHU Building dataset
# whether using depends on your use case: if your dataset is larger than WHU Building dataset, you could use the mean and
# std w.r.t. your own dataset, otherwise we recommend to use these mean and std.
rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)

class dataset_predict(data.Dataset):
    def __init__(self,
                 config,
                 args):
        super(dataset_predict, self).__init__()
        self.config = config
        self.args = args
        self.root = self.args.input

        # self.args.input
        self.data_list = glob.glob(os.path.join(self.root, '*'))

    def untrain_transforms(self, image):

        resize = transforms.Resize(size=(self.config.input_size, self.config.input_size))
        image = resize(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=rgb_mean, std=rgb_std)

        return image

    def __getitem__(self, index):

        datas = Image.open(self.data_list[index])

        t_datas = self.untrain_transforms(datas)

        # return data and its filename
        return t_datas, self.data_list[index]

    def __len__(self):

        return len(self.data_list)


class Predictor(object):
    def __init__(self,
                 args,
                 model,
                 dataloader_predict):
        super(Predictor, self).__init__()

        self.args = args
        self.device = torch.device('cpu') if self.args == -1 else torch.device('cuda:{}'.format(self.args.gpu))
        self.model = model.to(self.device)
        self.dataloader_predict = dataloader_predict

    def predict(self):

        self.model.eval()
        predict_time = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target, filenames) in enumerate(self.dataloader_predict, start=1):
                # data
                data = data.to(self.device, non_blocking = True)
                data_time.update(time.time() - tic)

                pre_tic = time.time()
                logits = self.model(data)
                predict_time.update(time.time() - pre_tic)
                self._save_pred(logits, filenames)

                batch_time.update(time.time() - tic)
                tic = time.time()

            print("Predicting and Saving Done!\n"
                  "Total Time: {:.2f}\n"
                  "Data Time: {:.2f}\n"
                  "Pre Time: {:.2f}"
                  .format(batch_time._get_sum(), data_time._get_sum(), predict_time._get_sum()))

    def _save_pred(self, predictions, filenames):
        """
        save predictions after evaluation phase
        :param predictions: predictions (output of model logits(after softmax))
        :param filenames: filenames list correspond to predictions
        :return: None
        """
        for index, map in enumerate(predictions):

            map = torch.argmax(map, dim=0)
            map = map * 255
            map = np.asarray(map.cpu(), dtype=np.uint8)
            map = Image.fromarray(map)
            # filename /0.1.png [0] 0 [1] 1
            filename = filenames[index].split('/')[-1].split('.')
            save_filename = filename[0]+'.'+filename[1]
            save_path = os.path.join(self.args.output, save_filename+'.png')

            map.save(save_path)

        # pred is tensor  --> numpy.ndarray save as single-channel --> save
        # get a mask 不用管channel的问题

if __name__ == '__main__':

    config = MyConfiguration()

    parser = argparse.ArgumentParser("configurations for prediction")
    parser.add_argument('-input', metavar='input', type=str, default=None,
                        help='root path to directory containing images used for predicting')
    parser.add_argument('-output', metavar='output', type=str, default=None,
                        help='root path to directory output predicted images')
    parser.add_argument('-weight', metavar='weight', type=str, default=None,
                        help='path to ckpt which will be loaded')
    parser.add_argument('-threads', metavar='threads', type=int, default=2,
                        help='number of thread used for DataLoader')
    parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                        help='gpu id to be used for prediction')
    args = parser.parse_args()

    model = ESFNet.ESFNet(config = config)

    dataset_predict = dataset_predict(config = config, args= args)

    dataloader_predict = data.DataLoader(dataset=dataset_predict,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=args.threads,
                                         drop_last=False)

    predictor = Predictor(args = args, model = model, dataloader_predict=dataloader_predict)
    predictor.predict()