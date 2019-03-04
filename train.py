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
from models.ERFNet import ERFNet
from models.EDANet import EDANet
from visdom import Visdom

def for_train(model,
              config,
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
    Trainer = BaseTrainer(model=model, config=config,
                          train_data_loader= train_data_loader,
                          valid_data_loader= valid_data_loader,
                          begin_time=begin_time,
                          resume_file= resume_file,
                          loss_weight= loss_weight,
                          visdom=visdom)
    Trainer.train()
    print(" Training Done ! ")

def for_test(model, config, test_data_loader, begin_time, loss_weight,):# do_predict,):
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
    Tester = BaseTester(model= model, config= config,
                        test_data_loader= test_data_loader,
                        begin_time= begin_time,
                        loss_weight= loss_weight,
                        #do_predict = do_predict
                        ,)
    Tester.eval()
    print(" Evaluation Done ! ")
    #if do_predict == True :
    #    Tester.predict()
    #    print(" Make Predict Image Done ! ")

def main(config):

    loss_weight = torch.ones(config.nb_classes)
    loss_weight[0] = 1.53297775619
    loss_weight[1] = 7.63194124408

    model = EDANet(config= config)
    print(model)
    
    # create visdom
    viz = Visdom(server=args.server, port=args.port, env=model.name)
    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'
    
    train_dataset = MyDataset(config=config, subset='train')
    valid_dataset = MyDataset(config=config, subset='val')
    test_dataset = MyDataset(config=config, subset='test')

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   drop_last=True)
    # TODO will drop_last will have effects on accuracy? no
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=2,
                                   drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=True)

    begin_time = datetime.datetime.now().strftime('%m%d_%H%M%S')

    for_train(model = model, config=config,
              train_data_loader = train_data_loader,
              valid_data_loader= valid_data_loader,
              begin_time= begin_time,
              resume_file= None,
              loss_weight= loss_weight,
              visdom=viz)

    """
    # testing phase does not need visdom, just one scalar for loss, miou and accuracy
    """
    for_test(model = model, config=config,
             test_data_loader=test_data_loader,
             begin_time= begin_time,
             loss_weight= loss_weight,
             do_predict= True,)

if __name__ == '__main__':

    # for visdom
    DEFAULT_PORT=8097
    DEFAULT_HOSTNAME="http://localhost"

    parser = argparse.ArgumentParser(description="Efficient Semantic Segmentation Network")
    parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str, default=DEFAULT_HOSTNAME,
                        help='Server address of the target to run the demo on.')
    args = parser.parse_args()

    # GPU setting init
    # keep prediction results the same when model runs each time
    config = MyConfiguration()
    """
    You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.'
    """
    # 可以增加运行效率 内置cuDNN的auto-tuner 自动寻找适合当前配置的高校算法 来优化运行效率
    # 如果网络输入数据维度和类型上变化不大，可以增加运行效率，如果相差很大则不行，每次都要重新寻找适合的配置
    torch.backends.cudnn.benchmark = True
    # 采用确定性卷积 相当于把所有操作seed=0 以便重现 但是会变慢
    # Deterministic mode can have a performance impact, depending on your model. 会降低速度
    #torch.backends.cudnn.deterministic = True
    #torch.cuda.manual_seed(config.random_seed)
    # for distribution
    #torch.cuda.manual_seed_all(config.random_seed)
    # seed the RNG for all devices(both CPU and GPUs)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    main(config= config)
