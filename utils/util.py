import os
import torch
import sys
from configs.config import MyConfiguration
import torch.nn as nn
import glob
from tqdm import tqdm
from scipy import misc
import numpy as np
import re
import functools
from collections import OrderedDict
import pandas as pd
# from torch.autograd import Variable
import torch.nn.functional as F


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
        self.sum = val * weight
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
    # print(dataset_path)
    # print(target_path)

    data_paths = glob.glob(os.path.join(dataset_path, '*.tif'))
    for data_path in tqdm(data_paths):
        target_path = data_path.replace(config.data_folder_name, config.target_folder_name)

        filename = data_path.split('/')[-1].split('.')[0]

        data = misc.imread(data_path)
        target = misc.imread(target_path)

        h, w = config.original_size, config.original_size

        subimgcounter = 1
        stride = config.cropped_size - config.overlapped
        for h_pixel in range(0, h - config.cropped_size + 1, stride):
            for w_pixel in range(0, w - config.cropped_size + 1, stride):
                new_data = data[h_pixel:h_pixel + config.cropped_size, w_pixel:w_pixel + config.cropped_size, :]
                new_target = target[h_pixel:h_pixel + config.cropped_size, w_pixel:w_pixel + config.cropped_size]

                data_save_path = os.path.join(new_dataset_path, filename + '.{}.tif'.format(subimgcounter))
                target_save_path = data_save_path.replace(config.data_folder_name, config.target_folder_name)

                misc.imsave(data_save_path, new_data)
                misc.imsave(target_save_path, new_target)
                subimgcounter += 1


# feature select from intermediate layer in Network from module or Sequential
class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)

        self._to_select = to_select

    def forward(self, input):
        list = []
        for name, module in self._modules.iteritems():
            x = module(x)
            if name in self._to_select:
                list.append(x)

        return list


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]

        return outputs + [x]


class model_utils():
    # class global variable
    flops_list_conv = []
    flops_list_linear = []
    flops_list_bn = []
    flops_list_relu = []
    flops_list_pooling = []

    mac_list_conv = []
    mac_list_linear = []
    mac_list_bn = []
    mac_list_relu = []
    mac_list_pooling = []

    MULTIPLY_ADDS = False
    summary = OrderedDict()
    names = {}
    display_input_shape = True
    display_weights = False
    display_nb_trainable = False

    def __init__(self, model, config, index=None):
        # TODO include batch_size or not include batch_size
        """
        for hooks register
        Generally, since majority of flops are in conv and linear, nflops ~= X might show that you are approximating it, and
        that is prob sufficient for almost all things.


        :param model:
        :param config: for simulating input_size, batch_size
        """
        self.model = model
        self.config = config
        self.hooks = []
        # for test downsampling flops and params
        # input_size=256
        self.channels_list = [3,   64,  128, 256, 512, 1024]
        self.size_list =     [256, 128, 64 , 32,  16,  8]
        self.index = index
    def model_params(self):

        trainable_params = sum([p.nelement() for p in filter(lambda p: p.requires_grad, self.model.parameters())])
        total_params = sum([p.nelement() for p in self.model.parameters()])

        return trainable_params, total_params


    def _register_hooks(self):

        # register_forward_hook called everytime after forward() compute an output
        # children 会返回init中定义的module, modules()会递归返回所有module
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                self.hooks.append(module.register_forward_hook(self.conv_hook))
                self.hooks.append(module.register_forward_hook(self.hook))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self.linear_hook))
                self.hooks.append(module.register_forward_hook(self.hook))
            elif isinstance(module, nn.BatchNorm2d):
                self.hooks.append(module.register_forward_hook(self.bn_hook))
                self.hooks.append(module.register_forward_hook(self.hook))
            elif isinstance(module, nn.ReLU):
                self.hooks.append(module.register_forward_hook(self.relu_hook))
                self.hooks.append(module.register_forward_hook(self.hook))
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                self.hooks.append(module.register_forward_hook(self.pooling_hook))
                self.hooks.append(module.register_forward_hook(self.hook))

    def simulate_forward(self):

        total_params, trainable_params = self.model_params()
        model_utils.names = self.get_names_dict()
        self._register_hooks()

        input = torch.randn(size=(1, 3, self.config.input_size, self.config.input_size))
        # for test module performance
        #input = torch.randn(size=(1, self.channels_list[self.index], self.size_list[self.index], self.size_list[self.index]))
        if next(self.model.parameters()).is_cuda:
            input = input.cuda()

        self.model(input)

        total_mac = sum(self.mac_list_conv) + sum(self.mac_list_linear) + sum(self.mac_list_bn) + \
                    sum(self.mac_list_relu) + sum(self.mac_list_pooling)
        total_flops = sum(self.flops_list_conv) + sum(self.flops_list_linear) + sum(self.flops_list_bn) + \
                      sum(self.flops_list_relu) + sum(self.flops_list_pooling)
        df_summary = pd.DataFrame.from_dict(model_utils.summary, orient='index')

        print(df_summary)
        print("     + Number MAC of Model: {:.2f}M ".format(float(total_mac) / 1e6))
        print("     + Number FLOPs of Model: {:.2f}M ".format(float(total_flops) / 1e6))
        print("     + Number Total Params of Model: {:.2f}M ".format(float(total_params) / 1e6))
        print("     + Number Trainanble Params of Model: {:.2f}M ".format(float(trainable_params) / 1e6))
        for h in self.hooks:
            h.remove()

    def get_names_dict(self):
        """
        Recursive walk to get names including path
        """
        names = {}

        def _get_names(module, parent_name=''):
            for key, module in module.named_children():
                name = parent_name + '.' + key if parent_name else key
                names[name] = module
                if isinstance(module, torch.nn.Module):
                    _get_names(module, parent_name=name)

        _get_names(self.model)
        # print(names.keys())
        return names

    @staticmethod
    def conv_hook(self, input, output):

        batch_size, in_channels, height, width = input[0].shape
        out_channels, out_height, out_width = output[0].shape

        kernel_flops = self.kernel_size[0] * self.kernel_size[1] * (in_channels // self.groups) * (
             2 if model_utils.MULTIPLY_ADDS else 1)

        kernel_mac = self.kernel_size[0] * self.kernel_size[1] * in_channels * out_channels
        map_mac = height * width * in_channels * out_channels
        MAC = kernel_mac + map_mac

        bias_flops = 1 if self.bias is not False else 0
        # Dk*Dk*in*Df*Df*out
        flops = batch_size * (kernel_flops + bias_flops) * out_height * out_width * out_channels
        # access class variable via class_name.variable_name
        model_utils.flops_list_conv.append(flops)
        model_utils.mac_list_conv.append(MAC)


    @staticmethod
    def linear_hook(self, input, output):

        batch_size = input[0].shape[0] if input[0].dim() == 2 else 1
        # in_features, out_features matrix multiply,
        # for each element* out_features = sum(each_features(1)* out_features) * in_features
        weight_flops = self.weight.nelement() * (2 if model_utils.MULTIPLY_ADDS else 1)
        bias_flops = self.bias.nelement()

        weight_mac = self.weight.nelement()
        map_mac = self.in_features+ self.out_features
        MAC = weight_mac + map_mac

        flops = batch_size * (weight_flops + bias_flops)
        model_utils.flops_list_linear.append(flops)
        model_utils.mac_list_linear.append(MAC)


    @staticmethod
    def bn_hook(self, input, output):
        model_utils.flops_list_bn.append(input[0].nelement())
        map_mac = input[0].nelement() * 2
        model_utils.mac_list_bn.append(map_mac)

    @staticmethod
    def relu_hook(self, input, output):
        model_utils.flops_list_relu.append(input[0].nelement())
        map_mac = input[0].nelement() * 2
        model_utils.mac_list_relu.append(map_mac)

    # TODO does pooling has flops ? maxpooling or average pooling?
    # average and maxpooling ommit adds auto
    @staticmethod
    def pooling_hook(self, input, output):
        batch_size, in_channels, height, width = input[0].shape
        out_channels, out_height, out_width = output[0].shape

        # kernel_flops in each channels, doesn't share weight along all channels
        kernel_flops = self.kernel_size * self.kernel_size
        # MAC only for feature map, kernel doesn't have parameters
        map_mac = height * width * in_channels + out_height * out_width * out_channels

        bias_flops = 0  # pooling for no bias
        flops = batch_size * (kernel_flops + bias_flops) * out_height * out_width * out_channels
        model_utils.flops_list_pooling.append(flops)
        model_utils.mac_list_pooling.append(map_mac)

    @staticmethod
    def hook(self, input, output):
        name = ''
        for key, item in model_utils.names.items():
            if item == self:
                name = key
        # <class 'torch.nn.modules.conv.Conv2d'>
        class_name = str(self.__class__).split('.')[-1].split("'")[0]
        module_idx = len(model_utils.summary)

        # key_id for new module
        m_key = module_idx + 1

        model_utils.summary[m_key] = OrderedDict()
        model_utils.summary[m_key]['name'] = name
        model_utils.summary[m_key]['class_name'] = class_name

        if model_utils.display_input_shape:
            model_utils.summary[m_key]['input_shape'] = (-1,) + tuple(input[0].size())[1:]
        model_utils.summary[m_key]['output_shape'] = (-1,) + tuple(output.size())[1:]

        if model_utils.display_weights:
            model_utils.summary[m_key]['weights'] = list([tuple(p.size()) for p in self.parameters()])

        #summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])
        if model_utils.display_nb_trainable:
            params_trainable = sum(
                [torch.LongTensor(list(p.size())).prod() for p in self.parameters() if p.requires_grad]
            )
            model_utils.summary[m_key]['nb_trainable'] = params_trainable

        params = sum([torch.LongTensor(list(p.size())).prod() for p in self.parameters()])
        model_utils.summary[m_key]['nb_params'] = params
        # hook ends

# Net for testing util function
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = SelectiveSequential(
            ['conv1', 'conv3'],
            {
                'conv1': nn.Conv2d(1, 1, 3),
                'conv2': nn.Conv2d(1, 1, 3),
                'conv3': nn.Conv2d(1, 1, 3)
            }

        )

    def forward(self, input):
        return self.features(input)


# only for testing
class test_Net(nn.Module):
    def __init__(self):
        super(test_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(10, 3, 3, padding=1, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class test_hook_register():
    aaa = 1

    def __init__(self, model):
        self.value = 2
        self.model = model

    @staticmethod
    def hook(self, input, output):
        # print(self.kernel_size)
        print('1')
        print(test_hook_register.aaa)
        # print(value)

    def get_value(self):
        pass

    def register(self):
        for child in model.children():
            child.register_forward_hook(self.hook)

    def simulate_forward(self):
        self.register()
        input = torch.randn(size=(1, 3, 64, 64))
        model(input)
        # print("     + Number flops of Module: {}".format())


if __name__ == '__main__':
    model = test_Net()
    # for name, module in model._modules.items():
    #    print(name, module)
    #    print(module.in_channels, module.out_channels, module.kernel_size)
    # model.named_modules() 类似树形搜索
    # model.children() 只输出init部分定义的内容 与 model._modules.items()输出相同
    # model_flops(model)

    # input = torch.randn(1,3,224,224)
    # output = model(input)

    model_utils = model_utils(model)
