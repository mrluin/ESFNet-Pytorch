import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel
# MobileNet as Encoder in Semantic Segmentation

class separable_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=1,
                 bias=False):
        super(separable_conv2d, self).__init__()
        # depth-multiply=1 means in_channels=out_channels
        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            groups=in_channels,
        )
        # out_channels = in_channels only for dw
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise_conv2d = nn.Conv2d(
            in_channels= in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            groups=1,
        )

        self.bn_pw = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.depthwise_conv2d(input)
        out = F.relu(self.bn_dw(out))
        out = self.pointwise_conv2d(out)
        out = F.relu(self.bn_pw(out))

        return out

class conv2d_bn_relu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=1,
                 bias=False,
                 groups=1):
        super(conv2d_bn_relu, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size= kernel_size, stride=stride,
            dilation=dilation, padding=padding,
            bias=bias, groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        out = self.conv2d(input)
        return F.relu(self.bn(out))


class MobileNet(BaseModel):
    def __init__(self, config):
        super(MobileNet, self).__init__()
        # for semantic segmentation
        self.name = 'MobileNet8x'
        self.nb_classes = config.nb_classes

        # 8x down-sampling
        self.encoder = nn.Sequential(
            conv2d_bn_relu(3, 32, 3, stride=2),#
            separable_conv2d(32, 64, 3, stride=1),
            separable_conv2d(64, 128, 3, stride=2),#
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 256, 3, stride=2),#
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
        )
        '''
        # 16x down-sampling
        self.encoder = nn.Sequential(
            conv2d_bn_relu(3, 32, 3, stride=2),#
            separable_conv2d(32, 64, 3, stride=1),
            separable_conv2d(64, 128, 3, stride=2),#
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 256, 3, stride=2),#
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 512, 3, stride=2),#
            separable_conv2d(512, 512, 3, stride=1),
            separable_conv2d(512, 512, 3, stride=1),
            separable_conv2d(512, 512, 3, stride=1),
            separable_conv2d(512, 512, 3, stride=1),
            separable_conv2d(512, 512, 3, stride=1),
        )
        '''
        # use deconvolution to decode like using in ERFNet
        # for 2x2x2x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 128, 3, stride=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            separable_conv2d(64, 64, 3, stride=1),
            separable_conv2d(64, 32, 3, stride=1),
            nn.ConvTranspose2d(32, self.nb_classes, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False)
        )
        '''
        # for 2x4x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 128, 3, stride=1),
            nn.ConvTranspose2d(128, self.nb_classes, kernel_size=3, stride=4,
                               padding=1, output_padding=3, bias=False),
        )
        # for 8x
        self.decoder = nn.ConvTranspose2d(256, self.nb_classes, kernel_size=3,
                                          stride=8, padding=1, output_padding=7,bias=False)
        
        # for 2x2x2x2x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 128, 3, stride=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            separable_conv2d(64, 64, 3, stride=1),
            separable_conv2d(64, 32, 3, stride=1),
            nn.ConvTranspose2d(32, self.nb_classes, kernel_size=3, stride=2,
                           padding=1, output_padding=1, bias=False)
        )
        # for 2x2x4x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            separable_conv2d(128, 128, 3, stride=1),
            separable_conv2d(128, 128, 3, stride=1),
            nn.ConvTranspose2d(128, self.nb_classes, kernel_size=3, stride=4,
                               padding=1, output_padding=3, bias=False),
        )
        # 2x8x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            separable_conv2d(256, 256, 3, stride=1),
            separable_conv2d(256, 256, 3, stride=1),
            nn.ConvTranspose2d(256, self.nb_classes, kernel_size=3, stride=8,
                               padding=1, output_padding=7, bias=False),

        )
        '''

    def forward(self, input):

        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)

        return decoder_out

class MobileNet_test(nn.Module):
    def __init__(self, config):
        super(MobileNet_test, self).__init__()
        # for testing FLOPs calculation
        self.name = 'MobileNet'
        self.nb_classes = config.nb_classes

        self.layers = nn.ModuleList()

        self.layers.append(conv2d_bn_relu(3, 32, 3, stride=2))
        self.layers.append(separable_conv2d(32, 64, 3, stride=1))
        self.layers.append(separable_conv2d(64, 128, 3, stride=2))
        self.layers.append(separable_conv2d(128, 128, 3, stride=1))
        self.layers.append(separable_conv2d(128, 256, 3, stride=2))
        self.layers.append(separable_conv2d(256, 256, 3, stride=1))
        self.layers.append(separable_conv2d(256, 512, 3, stride=2))
        self.layers.append(separable_conv2d(512, 512, 3, stride=1))
        self.layers.append(separable_conv2d(512, 512, 3, stride=1))
        self.layers.append(separable_conv2d(512, 512, 3, stride=1))
        self.layers.append(separable_conv2d(512, 512, 3, stride=1))
        self.layers.append(separable_conv2d(512, 512, 3, stride=1))
        self.layers.append(separable_conv2d(512, 1024, 3, stride=2))
        self.layers.append(separable_conv2d(1024, 1024, 3, stride=1))
        self.layers.append(nn.AvgPool2d(7))

        self.fc = nn.Linear(1024, 1000)

    def forward(self, input):

        out=input
        for layer in self.layers:
            out = layer(out)

        out = out.view(-1, 1024)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    pass

















