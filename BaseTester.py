import torch
import numpy as np
import os
import sys
import time
import torch.optim as optim
import torch.nn as nn
import cv2
import glob
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from tqdm import tqdm
from metrics import Accuracy, MIoU
from utils.util import AverageMeter, ensure_dir
from PIL import Image



class BaseTester(object):
    def __init__(self,
                 model,
                 config,
                 test_data_loader,
                 begin_time,
                 loss_weight,
                 do_predict):

        # for general
        self.config = config
        self.device = torch.device('cuda:{}'.format(self.config.device_id)) if self.config.use_gpu else torch.device('cpu')
        self.do_predict = do_predict

        # for train
        self.model = model.to(self.device)
        self.loss_weight = loss_weight.to(self.device)
        self.loss = self._loss(loss_function= self.config.loss).to(self.device)
        self.optimizer = self._optimizer(lr_algorithm=self.config.lr_algorithm)
        self.lr_scheduler = self._lr_scheduler()

        # for time
        self.begin_time = begin_time

        # for data
        self.test_data_loader = test_data_loader

        # for resume/save path
        self.history = {
            'eval': {
                'loss': [],
                'acc': [],
                'miou': [],
            },
        }
        self.test_log_path = os.path.join(self.config.test_log_dir, model.name, self.begin_time)
        self.predict_path = os.path.join(self.config.pred_dir, model.name, self.begin_time)
        self.resume_ckpt_path = os.path.join(self.config.save_dir, model.name, self.begin_time, 'checkpoint-best.pth')
        ensure_dir(self.test_log_path)
        ensure_dir(self.predict_path)

    def _optimizer(self, lr_algorithm):

        if lr_algorithm == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.config.init_lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.config.weight_decay,
                                   amsgrad=False)
            return optimizer
        if lr_algorithm == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.config.init_lr,
                                  momentum=self.config.momentum,
                                  dampening=0,
                                  weight_decay=self.config.weight_decay,
                                  nesterov=True)
            return optimizer

    def _loss(self, loss_function):
        """
        loss weight, ignore_index
        :param loss_function: bce_loss / cross_entropy
        :return:
        """
        if loss_function == 'bceloss':
            loss = nn.BCEWithLogitsLoss(weight=self.loss_weight)
            return loss

        if loss_function == 'crossentropy':
            loss = nn.CrossEntropyLoss(weight=self.loss_weight)
            return loss

    def _lr_scheduler(self):

        lambda1 = lambda epoch: pow((1-((epoch-1)/self.config.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        return lr_scheduler

    def eval(self):

        self._resume_ckpt()

        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_total_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target, _) in enumerate(self.test_data_loader,start=1):

                # data
                data = data.to(self.device)
                target = target.to(self.device)
                data_time.update(time.time()-tic)

                # output, loss, and metrics
                logits = self.model(data)
                loss = self.loss(logits, target)
                acc = Accuracy(logits, target)
                miou = MIoU(logits, target, self.config.nb_classes)

                # update ave loss and metrics
                batch_time.update(time.time()-tic)
                tic = time.time()
                ave_total_loss.update(loss.data.item())
                ave_acc.update(acc)
                ave_iou.update(miou)

            # display evaluation result at the end
            print('Evaluation phase !'
                  'Time: {:.2f},  Data: {:.2f},'
                  'MIoU: {:6.4f}, Accuracy: {:6.4f}, Loss: {:.6f}'
                  .format(batch_time.average(), data_time.average(),
                          ave_iou.average(), ave_acc.average(), ave_total_loss.average()))

        self.history['eval']['loss'].append(ave_total_loss.average())
        self.history['eval']['acc'].append(ave_acc.average())
        self.history['eval']['miou'].append(ave_iou.average())
        #self.history['test']['time'].append(batch_time.average() - data_time.average())

        #TODO
        print("Saved history of evaluation phase !")
        hist_path = os.path.join(self.test_log_path, "history.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history))


    def predict(self):

        self._resume_ckpt()

        self.model.eval()

        predictions = []
        filenames = []

        ave_total_time = AverageMeter()
        #test_paths = glob.glob(os.path.join(self.config.root_dir, 'test', self.config.data_folder_name, '*.tif'))

        # test_data_loader batch_size=1
        with torch.no_grad():

            for (data, _, filename) in tqdm(self.test_data_loader):
                # only need data in predict phase
                data = data.to(self.device)

                tic = time.time()
                pred_map = self.model(data)
                ave_total_time.update(time.time()-tic)

                predictions.extend(pred_map)
                filenames.extend(filename)

            print("Saveing ... ... ")
            self._save_pred(predictions, filenames)
            print("Total time cost : {}s ,"
                  "Per image time cost : {}s"
                  .format(ave_total_time._get_sum(), ave_total_time.average()))
            '''
            for test_path in tqdm(test_paths):
                data_index = test_path.split('/')[-1].split('.')[0]
                save_path = os.path.join(self.predict_path, data_index+'.png')
                # in predict phase only use data
                data = Image.open(test_path)
                data = self._untrain_data_transform(data)
                # here transform convert channel_last to channel_first
                tic = time.time()
                pred_map = self.model(data)
                # output of model is logic need process
                ave_total_time.update(time.time()- tic)
                self._save_pred(pred_map, save_path)

            print("Total time cost : {}s , "
                  "Per image time cost : {}s"
                  .format(ave_total_time._get_sum(), ave_total_time.average()))
            '''
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
            save_path = os.path.join(self.predict_path, save_filename+'.png')

            map.save(save_path)



        # pred is tensor  --> numpy.ndarray save as single-channel --> save
        # get a mask 不用管channel的问题
        '''
        pred = torch.argmax(pred, dim=0)
        mapping = {
            0: 0,
            1: 255,
        }
        for k in mapping:
            pred[pred == k] = mapping[k]
        pred = np.asarray(pred, dtype=np.uint8)
        pred = Image.fromarray(pred)
        pred.save(path)
        '''

    def _resume_ckpt(self):

        print("Loading ckpt path : {} ...".format(self.resume_ckpt_path))
        checkpoint = torch.load(self.resume_ckpt_path)

        self.model.load_state_dict(checkpoint['state_dict'])
        print("Model State Loaded ! :D ")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer State Loaded ! :D ")
        print("Checkpoint file: '{}' , Loaded ! "
              "Prepare to test ! ! !"
              .format(self.resume_ckpt_path))


    def _untrain_data_transform(self, data):

        rgb_mean = (0.4353, 0.4452, 0.4131)
        rgb_std = (0.2044, 0.1924, 0.2013)

        data = TF.resize(data, size=self.config.input_size)
        data = TF.to_tensor(data)
        data = TF.normalize(data, mean=rgb_mean, std=rgb_std)

        return data
