import torch
import random
import numpy as np
import argparse
from torch.utils.data.dataloader import DataLoader
from data.dataset import MyDataset
from models.ERFNet import ERFNet
from configs.config import MyConfiguration
from BaseTester import BaseTester

def for_test(model, config, test_data_loader, begin_time, loss_weight, do_predict):

    Tester = BaseTester(model= model, config=config,
                        test_data_loader= test_data_loader,
                        begin_time=begin_time,
                        loss_weight= loss_weight,
                        do_predict= do_predict)

    if do_predict:
        Tester.predict()
        print("Predict phase Done !")

def main(config):

    loss_weight = torch.ones(config.nb_classes)
    loss_weight[0] = 1.53297775619
    loss_weight[1] = 7.63194124408

    model = ERFNet(config= config)
    print(model)

    test_dataset = MyDataset(config=config, subset='test')
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=True)
    begin_time = '0224_143528'

    for_test(model = model, config=config,
             test_data_loader=test_data_loader,
             begin_time= begin_time,
             loss_weight= loss_weight,
             do_predict= True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Efficient Semantic Segmentation Network")
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
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    main(config= config)