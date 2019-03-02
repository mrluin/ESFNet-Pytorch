import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        """
        Foward pass logic
        :param input:
        :return: model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model Summary
        :return: None
        """
        # 把可训练参数挑出来
        #model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # np.prod 用来计算所有元素的乘积 axis 内
        #params = sum([np.prod(p.size()) for p in model_parameters])
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Trainable parameters: {}".format(params))

    def __str__(self):
        """
        Model prints with number of trainable paramters
        :return:
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
