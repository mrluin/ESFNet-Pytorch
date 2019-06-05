import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from BaseTrainer import BaseTrainer
from BaseTester import BaseTester
from data.dataset import MyDataset
from torch.utils.data import DataLoader
from models.MyNetworks import ESFNet
from visdom import Visdom

def for_train(model,
              config,
              args,
              train_data_loader,
              valid_data_loader,
              begin_time,
              resume_file,
              loss_weight,
              visdom):
    """
    :param model:
    :param config:
    :param train_data_loader:
    :param valid_data_loader:
    :param resume_file:
    :param loss_weight:
    :return:
    """
    Trainer = BaseTrainer(model=model, config=config, args= args,
                          train_data_loader= train_data_loader,
                          valid_data_loader= valid_data_loader,
                          begin_time=begin_time,
                          resume_file = resume_file,
                          loss_weight= loss_weight,
                          visdom=visdom)
    Trainer.train()
    print(" Training Done ! ")

def for_test(model, config, args, test_data_loader, begin_time, resume_file, loss_weight):
    """
    :param model:
    :param config:
    :param test_data_loader:
    :param begin_time:
    :param resume_file:
    :param loss_weight:
    :param predict:
    :return:
    """
    Tester = BaseTester(model= model, config= config, args = args,
                        test_data_loader= test_data_loader,
                        begin_time= begin_time,
                        resume_file = resume_file,
                        loss_weight= loss_weight)

    Tester.eval_and_predict()
    print(" Evaluation Done ! ")
    #if do_predict == True :
    #    Tester.predict()
    #    print(" Make Predict Image Done ! ")

def main(config, args):

    loss_weight = torch.ones(config.nb_classes)
    loss_weight[0] = 1.53297775619
    loss_weight[1] = 7.63194124408

    # Here config in model, only used for nb_classes, so we do not use args

    model = ESFNet.ESFNet(config= config)
    print(model)
    
    # create visdom
    viz = Visdom(server=args.server, port=args.port, env=model.name)
    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'

    # TODO there are somewhat still need to change in ../configs/config.cfg
    train_dataset = MyDataset(config=config, args= args, subset='train')
    valid_dataset = MyDataset(config=config, args= args, subset='val')
    test_dataset = MyDataset(config=config, args= args, subset='test')

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.threads,
                                   drop_last=True)
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.threads,
                                   drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.threads,
                                  drop_last=True)

    begin_time = datetime.datetime.now().strftime('%m%d_%H%M%S')


    for_train(model = model, config=config, args = args,
              train_data_loader = train_data_loader,
              valid_data_loader= valid_data_loader,
              begin_time= begin_time,
              resume_file = args.weight,
              loss_weight= loss_weight,
              visdom=viz)

    """
    # testing phase does not need visdom, just one scalar for loss, miou and accuracy
    """
    for_test(model = model, config=config, args= args,
             test_data_loader=test_data_loader,
             begin_time= begin_time,
             resume_file = args.weight,
             loss_weight= loss_weight,
             )



if __name__ == '__main__':

    config = MyConfiguration()

    # for visdom
    DEFAULT_PORT=8097
    DEFAULT_HOSTNAME="http://localhost"

    parser = argparse.ArgumentParser(description="Efficient Semantic Segmentation Network")
    parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str, default=DEFAULT_HOSTNAME,
                        help='Server address of the target to run the demo on.')
    parser.add_argument('-input', metavar='input', type=str, default=config.root_dir,
                        help='root path to directory containing input images, including train & valid & test')
    parser.add_argument('-output', metavar='output', type=str, default=config.save_dir,
                        help='root path to directory containing all the output, including predictions, logs and ckpt')
    parser.add_argument('-weight', metavar='weight', type=str, default=None,
                        help='path to ckpt which will be loaded')
    parser.add_argument('-threads', metavar='threads', type=int, default=8,
                        help='number of thread used for DataLoader')
    parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                        help='gpu id to be used for prediction')

    args = parser.parse_args()

    # GPU setting init
    # keep prediction results the same when model runs each time

    """
    You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.'
    """
    '''
    # It will improve the efficiency, internal cuDNN and auto-tuner will find the proper configurations in your use case,
      so that it will optimize the running efficiency.
    # If the dims and type do not have a magnitude difference, it will improve the running efficiency, otherwise it will
      find the proper configurations every time when it meets a new data format, thus it will have bad influence on efficiency.
    '''
    torch.backends.cudnn.benchmark = True
    '''
    # using deterministic mode can have performance impact(speed), depending on your model.
    '''
    #torch.backends.cudnn.deterministic = True
    #torch.cuda.manual_seed(config.random_seed)
    # for distribution
    #torch.cuda.manual_seed_all(config.random_seed)
    # seed the RNG for all devices(both CPU and GPUs)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    main(config= config, args = args)

