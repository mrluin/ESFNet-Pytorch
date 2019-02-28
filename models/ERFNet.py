import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel


class down_sampling_block(nn.Module):

    def __init__(self, inpc, oupc):
        super(down_sampling_block, self).__init__()
        self.branch_conv = nn.Conv2d(inpc, oupc-inpc, 3, stride=2, padding= 1, bias=False)
        self.branch_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(oupc, eps=1e-03)

    def forward(self, x):
        output = torch.cat([self.branch_conv(x), self.branch_mp(x)], 1)
        output = self.bn(output)
        return F.relu(output)

class up_sampling_block(nn.Module):

    def __init__(self, inpc, oupc):
        super(up_sampling_block, self).__init__()
        self.conv = nn.ConvTranspose2d(inpc, oupc, 3, 2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(oupc)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d(nn.Module):
    # stride = 1  kernel_size=3
    def __init__(self, inpc, oupc, dilated_rate, dropout_rate):
        super(non_bottleneck_1d, self).__init__()
        self.conv3x1_1 = nn.Conv2d(inpc, oupc, (3, 1), stride=1, padding=(1,0), bias=False)
        self.conv1x3_1 = nn.Conv2d(inpc, oupc, (1, 3), stride=1, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(oupc)
        self.conv3x1_2 = nn.Conv2d(inpc, oupc, (3, 1), stride=1, padding=(1*dilated_rate,0), dilation=(dilated_rate,1), bias=False)
        self.conv1x3_2 = nn.Conv2d(inpc, oupc, (1, 3), stride=1, padding=(0,1*dilated_rate), dilation=(1,dilated_rate), bias=False)
        self.bn2 = nn.BatchNorm2d(oupc)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)
        output += input
        return F.relu(output)

class ERFNet(BaseModel):

    def __init__(self,
                 config):
        super(ERFNet, self).__init__()

        self.config = config
        self.name="ERFNet"
        self.nb_classes = self.config.nb_classes
        #self.nb_classes = nb_classes

        # input 512
        self.initial_block = down_sampling_block(3, 16)
        # output 256
        self.layers = nn.ModuleList()

        self.layers.append(down_sampling_block(16, 64))
        # output 128
        for index in range(0,5):
            self.layers.append(non_bottleneck_1d(64, 64, 1, 0.03))

        self.layers.append(down_sampling_block(64, 128))

        for index in range(0,2):
            self.layers.append(non_bottleneck_1d(128, 128, 2, 0.3))
            self.layers.append(non_bottleneck_1d(128, 128, 4, 0.3))
            self.layers.append(non_bottleneck_1d(128, 128, 8, 0.3))
            self.layers.append(non_bottleneck_1d(128, 128, 16, 0.3))

        self.layers.append(up_sampling_block(128, 64))
        self.layers.append(non_bottleneck_1d(64, 64, 1, 0))
        self.layers.append(non_bottleneck_1d(64, 64, 1, 0))

        self.layers.append(up_sampling_block(64, 16))
        self.layers.append(non_bottleneck_1d(16, 16, 1, 0))
        self.layers.append(non_bottleneck_1d(16, 16, 1, 0))

        self.output_conv = nn.ConvTranspose2d(16, self.nb_classes, 2, 2, padding=0, output_padding=0, bias=False)

    def forward(self, x):
        output = self.initial_block(x)
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        return output

if __name__ == '__main__':

    model = ERFNet()
    print(model)