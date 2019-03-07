import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import time
from utils.util import AverageMeter, ensure_dir
from metrics import Accuracy, MIoU
#from visdom import Visdom


class BaseTrainer(object):

    def __init__(self,
                 model,
                 config,
                 train_data_loader,
                 valid_data_loader,

                 visdom,
                 begin_time,
                 resume_file=None,
                 loss_weight=None):

        print("Training Start ... ...")
        # for general
        self.config = config
        self.device = (self._device(self.config.use_gpu, self.config.device_id))
        self.model = model.to(self.device)
        self.train_data_loader = train_data_loader
        self.valid_data_loder = valid_data_loader

        # for time
        self.begin_time = begin_time               # part of ckpt name
        self.save_period = self.config.save_period # for save ckpt
        self.dis_period = self.config.dis_period   # for display

        # for save directory : model, best model, log{train_log, validation_log} per epoch
        '''
        Directory:
            #root | -- train 
                  | -- valid
                  | -- test
                  | -- save | -- {model.name} | -- datetime | -- ckpt-epoch{}.pth.format(epoch)
                            |                               | -- best_model.pth
                            |
                            | -- log | -- {model.name} | -- datetime | -- history.txt
                            | -- test| -- log
                                     | -- predict
        '''
        # /home/jingweipeng/ljb/Building_Detecion/cropped_aerial_torch/save/model.name/time
        # TODO model name setting
        self.checkpoint_dir = os.path.join(self.config.save_dir, self.model.name, self.begin_time)
        # /home/jingweipeng/ljb/Building_Detection/cropped_aerial_torch/save/log/model.name/time
        self.log_dir = os.path.join(self.config.log_dir, self.model.name, self.begin_time)
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
        self.history = {
            'train': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
            },
            'valid': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
            }
        }

        # for optimize
        self.loss_weight = loss_weight.to(self.device)
        self.loss = self._loss(loss_function=self.config.loss).to(self.device)
        self.optimizer = self._optimizer(lr_algorithm=self.config.lr_algorithm)
        self.lr_scheduler = self._lr_scheduler()
        self.weight_init_algorithm = self.config.init_algorithm
        self.current_lr = self.config.init_lr

        print(self.optimizer)
        print(self.loss)

        # for train
        self.start_epoch = 1
        self.early_stop = self.config.early_stop # early stop steps
        self.monitor_mode = self.config.monitor.split('/')[0]
        self.monitor_metric = self.config.monitor.split('/')[1]
        self.monitor_best = 0
        self.best_epoch = -1
        # monitor init
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        if resume_file is not None:
            self._resume_ckpt(resume_file=resume_file)

        # TODO visualization
        # value needed to visualize: loss, metrics[acc, miou], learning_rate
        self.visdom = visdom

    def _device(self, use_gpu, device_id):

        if use_gpu == False:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(device_id))
            return device

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

    def _weight_init(self, m):

        # no bias use
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if self.weight_init_algorithm == 'kaiming':
                init.kaiming_normal_(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self):

        # create panes for training phase for loss metrics learning_rate
        print("Visualization init ... ...")
        loss_window = self.visdom.line(
            X = torch.stack((torch.ones(1),torch.ones(1)),1),
            Y = torch.stack((torch.ones(1),torch.ones(1)),1),
            opts= dict(title='train_val_loss',
                       # for different size panes, the result of download is the same!
                       showlegend=True,
                       legend=['training_loss', 'valid_loss'],
                       xtype='linear',
                       xlabel='epoch',
                       xtickmin=0,
                       xtick=True,
                       xtickstep=10,
                       ytype='linear',
                       ylabel='loss',
                       ytickmin=0,
                       #ytickmax=1,
                       #ytickstep=0.1,
                       ytick=True,)
        )
        lr_window = self.visdom.line(
            X = torch.ones(1),
            Y = torch.tensor([self.current_lr]),
            opts = dict(title = 'learning_rate',
                        showlegend=True,
                        legend=['learning_rate'],
                        xtype='linear',
                        xlabel='epoch',
                        xtickmin=0,
                        xtick=True,
                        xtickstep=10,
                        ytype='linear',
                        ytickmin=0,
                        #ytickmax=1,
                        #ytickstep=0.1,
                        ylabel='lr',
                        ytick=True)
        )
        miou_window = self.visdom.line(
            X = torch.stack((torch.ones(1),torch.ones(1)),1),
            Y = torch.stack((torch.ones(1),torch.ones(1)),1),
            opts = dict(title='train_val_MIoU',
                        showlegend=True,
                        legend=['Train_MIoU', 'Val_MIoU'],
                        xtype='linear',
                        xlabel='epoch',
                        xtickmin=0,
                        xtick=True,
                        xtickstep=10,
                        ytype='linear',
                        ylabel='MIoU',
                        ytickmin=0,
                        #ytickmax=1,
                        #ytickstep=0.1,
                        ytick=True
                        )
        )
        acc_window = self.visdom.line(
            X = torch.stack((torch.ones(1), torch.ones(1)),1),
            Y = torch.stack((torch.ones(1), torch.ones(1)),1),
            opts = dict(title='train_val_Accuracy',
                        showlegend=True,
                        legend=['Train_Acc', 'Val_Acc'],
                        xtype='linear',
                        xlabel='epoch',
                        xtickmin=0,
                        xtick=True,
                        xtickstep=10,
                        ytype='linear',
                        ylabel='Accuracy',
                        ytickmin=0,
                        #ytickmax=1,
                        #ytickstep=0.1,
                        ytick=True)
        )

        print("Loaded, Training !")

        epochs = self.config.epochs
        # init weights at first
        self.model.apply(self._weight_init)
        for epoch in range(self.start_epoch, epochs+1):

            # get log information of train and evaluation phase
            train_log = self._train_epoch(epoch)
            eval_log = self._eval_epoch(epoch)

            # TODO visualization
            # for loss
            self.visdom.line(
                X = torch.stack((torch.ones(1)*epoch,torch.ones(1)*epoch),1),
                Y = torch.stack((torch.tensor([train_log['loss']]),torch.tensor([eval_log['val_Loss']])),1),
                win = loss_window,
                update='append' if epoch!=1 else 'insert',
            )
            # for learning_rate
            self.visdom.line(
                X = torch.ones(1)*epoch,
                Y = torch.tensor([self.current_lr]),
                win = lr_window,
                update='append' if epoch!=1 else 'insert',
            )
            # for metrics_miou
            self.visdom.line(
                X = torch.stack((torch.ones(1)*epoch, torch.ones(1)*epoch),1),
                Y = torch.stack((torch.tensor([train_log['miou']]), torch.tensor([eval_log['val_MIoU']])),1),
                win = miou_window,
                update='append' if epoch!=1 else 'insert',
            )
            # for metrics_accuracy
            self.visdom.line(
                X = torch.stack((torch.ones(1)*epoch, torch.ones(1)*epoch),1),
                Y = torch.stack((torch.tensor([train_log['acc']]), torch.tensor([eval_log['val_Accuracy']])),1),
                win = acc_window,
                update='append' if epoch!=1 else 'insert',
            )


            # save best model and save ckpt
            best = False
            not_improved_count = 0
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log['val_'+self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_'+self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_'+self.monitor_metric]
                    best = True
                    self.best_epoch = eval_log['epoch']
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation Performance didn\'t improve for {} epochs."
                          "Training stop :/"
                          .format(not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True:
                self._save_ckpt(epoch, best=best)
        # save history file
        print("Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _train_epoch(self, epoch):

        # lr update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.current_lr = param_group['lr']

        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_total_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()

        # set model mode
        self.model.train()
        tic = time.time()

        for steps, (data, target) in enumerate(self.train_data_loader, start=1):

            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # 加载数据所用的时间
            data_time.update(time.time() - tic)

            # forward calculate
            logits = self.model(data)
            loss = self.loss(logits, target)
            acc = Accuracy(logits, target)
            miou = MIoU(logits, target, self.config.nb_classes)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update average metrics
            batch_time.update(time.time() - tic)
            ave_total_loss.update(loss.data.item())
            ave_acc.update(acc.item())
            ave_iou.update(miou.item())

            # display on the screen per display_steps
            if steps % self.dis_period == 0:
                print('Epoch: [{}][{}/{}],\n'
                      'Learning_Rate: {:.6f},\n'
                      'Time: {:.4f},       Data:     {:.4f},\n'
                      'MIoU: {:6.4f},      Accuracy: {:6.4f},      Loss: {:.6f}'
                      .format(epoch, steps, len(self.train_data_loader),
                              self.current_lr,
                              batch_time.average(), data_time.average(),
                              ave_iou.average(), ave_acc.average(), ave_total_loss.average()))
            tic = time.time()
        #  train log and return
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_total_loss.average())
        self.history['train']['acc'].append(ave_acc.average())
        self.history['train']['miou'].append(ave_iou.average())
        return {
            'epoch': epoch,
            'loss': ave_total_loss.average(),
            'acc': ave_acc.average(),
            'miou': ave_iou.average(),
        }

    def _eval_epoch(self, epoch):


        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_total_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()

        # set model mode
        self.model.eval()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target) in enumerate(self.valid_data_loder, start=1):

                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                data_time.update(time.time() - tic)

                logits = self.model(data)
                loss = self.loss(logits, target)
                # calculate metrics
                acc = Accuracy(logits, target)
                miou = MIoU(logits, target, self.config.nb_classes)
                #print("===========acc, miou==========", acc, miou)

                # update ave metrics
                batch_time.update(time.time()-tic)

                ave_total_loss.update(loss.data.item())
                ave_acc.update(acc.item())
                ave_iou.update(miou.item())
                tic = time.time()
            # display validation at the end
            print('Epoch {} validation done !'.format(epoch))
            print('Time: {:.4f},       Data:     {:.4f},\n'
                  'MIoU: {:6.4f},      Accuracy: {:6.4f},      Loss: {:.6f}'
                  .format(batch_time.average(), data_time.average(),
                          ave_iou.average(), ave_acc.average(), ave_total_loss.average()))

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_total_loss.average())
        self.history['valid']['acc'].append(ave_acc.average())
        self.history['valid']['miou'].append(ave_iou.average())
        #  validation log and return
        return {
            'epoch': epoch,
            'val_Loss': ave_total_loss.average(),
            'val_Accuracy': ave_acc.average(),
            'val_MIoU': ave_iou.average(),
        }

    def _save_ckpt(self, epoch, best):

        # save model ckpt
        state = {
            'epoch': epoch,
            'arch': str(self.model),
            'history': self.history,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        best_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best.pth')
        if best:
            print("Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
        else:
            print("Saving Checkpoint per {} epochs, path: {} ... ".format(self.save_period, filename))
            torch.save(state, filename)

    def _resume_ckpt(self, resume_file):
        """
        :param resume_file: checkpoint file name
        :return:
        """
        resume_path = os.path.join(self.checkpoint_dir, resume_file)
        print("Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1 # 即将开始的epoch 存储的时候是结束时候的epoch
        self.monitor_best = checkpoint['monitor_best']

        self.model.load_state_dict(checkpoint['state_dict'])
        print("Model State Loaded ! :D ")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer State Loaded ! :D ")
        self.history = checkpoint['history']
        print("Checkpoint file: '{}' , Start epoch {} Loaded !"
              "Prepare to run ! ! !"
              .format(resume_path, self.start_epoch))

    def state_cuda(self, msg):
        print("--", msg)
        print("allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM"%(
            torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            torch.cuda.max_memory_allocated(self.device) / 1024/ 1024,
            torch.cuda.memory_cached(self.device) / 1024/ 1024,
            torch.cuda.max_memory_cached(self.device) / 1024/ 1024,
        ))