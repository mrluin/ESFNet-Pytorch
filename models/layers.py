import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class channel_shuffle(nn.Module):

    def __init__(self,groups):
        super(channel_shuffle, self).__init__()

        self.groups = groups
    def forward(self, input):
        '''
        view 需要的tensor内存是整块的, view只能作用在contiguous的varibale, 如果在view之前用了
        transpose, permute操作, 需要用contiguous copy
        有些 tensor 并不是占用整块内存 需要用contiguous(）
        torch.Tensor.is_contiguous()
        '''
        batch_size, channels, height, width = input.data.size()
        channels_per_group = channels // self.groups
        x = input.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        return x

