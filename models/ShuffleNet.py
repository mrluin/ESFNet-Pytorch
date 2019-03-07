import torch
import torch.nn as nn
import torch.nn.functional as F

# Modules do not use bias=True for the all
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

def conv3x3(in_channels, out_channels, stride=1, groups=1):

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, dilation=1,
                     groups=groups, bias=False)


def conv1x1(in_channels, out_channels, groups):
    """
    -Normal pointwise convolution : groups == 1
    -Grouped pointwise convolution : groups > 1
    """
    return nn.Conv2d(in_channels= in_channels, out_channels= out_channels,
                     kernel_size=1, stride=1, padding=0, dilation=1,
                     groups=groups, bias=False)


def channel_shuffle(x, groups):

    batch_size, in_channels, height, width = x.data.size()
    channels_per_group = in_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

'''
class channel_shuffle(nn.Module):
    def __init__(self, x, groups):
        super(channel_shuffle, self).__init__()

        self.groups = groups
        self.batch_size, self.in_channels, self.height, self.width = x.data.size()
        self.channels_per_group = self.in_channels // groups

    def forward(self, input):

        input = input.view(self.batch_size, self.groups, self.channels_per_group,
                           self.height, self.width)
        input = torch.transpose(input, 1, 2).contiguoug()
        input = input.view(self.batch_size, -1, self.height, self.width)

        return input
'''

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride=1,):
        # groups for GConv_1x1
        super(ShuffleUnit, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride= stride

        self.internal_channels = out_channels // 4

        self.gconv_1x1_compress = nn.Sequential(
            conv1x1(in_channels= in_channels,
                    out_channels= self.internal_channels,
                    groups=groups),
            nn.BatchNorm2d(self.internal_channels),
            nn.ReLU()
        )
        self.dwConv_bn = nn.Sequential(
            conv3x3(in_channels= self.internal_channels,
                    out_channels = self.internal_channels,
                    stride=stride, groups= self.internal_channels),
            nn.BatchNorm2d(self.internal_channels),
        )
        self.gconv_1x1_expand = nn.Sequential(
            conv1x1(in_channels= self.internal_channels,
                    out_channels= self.out_channels if self.stride==1 else (self.out_channels-self.in_channels),
                    groups=groups),
            nn.BatchNorm2d(self.out_channels if self.stride==1 else (self.out_channels -self.in_channels))
        )


    def forward(self, input):
        #print(self.in_channels, input.shape)
        x = input
        res_out = self.gconv_1x1_compress(input)

        #print(self.in_channels, res_out.shape)
        res_out = channel_shuffle(res_out, self.groups)
        res_out = self.dwConv_bn(res_out)

        res_out = self.gconv_1x1_expand(res_out)
        #print(self.in_channels, res_out.shape) # 64
        if self.stride == 2:
            main_out = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            out = torch.cat([main_out, res_out], dim=1)
        else:
            out = x + res_out
        #print(self.in_channels, out.shape)
        return F.relu(out)

class ShuffleSeg(nn.Module):
    def __init__(self, config, groups=None):
        super(ShuffleSeg, self).__init__()

        self.name= 'shuffleSeg16x2x'
        self.config = config
        self.groups = groups

        if self.groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 576]
        elif self.groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif self.groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif self.groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif self.groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]

        # encoder for 8x
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),#
            separable_conv2d(32, 64, stride=1),
            ShuffleUnit(in_channels=64, out_channels=128, groups=8, stride=2),#
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=256, groups=8, stride=2),#
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
        )
        '''
        # encoder for 16x
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  #
            separable_conv2d(32, 64, stride=1),
            ShuffleUnit(in_channels=64, out_channels=128, groups=8, stride=2),#
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=256, groups=8, stride=2),#
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=512, groups=8, stride=2),#
            ShuffleUnit(in_channels=512, out_channels=512, groups=8, stride=1),
            ShuffleUnit(in_channels=512, out_channels=512, groups=8, stride=1),
            ShuffleUnit(in_channels=512, out_channels=512, groups=8, stride=1),
            ShuffleUnit(in_channels=512, out_channels=512, groups=8, stride=1),
            ShuffleUnit(in_channels=512, out_channels=512, groups=8, stride=1),
        )
        '''
        # for 2x2x2x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=64, out_channels=64, groups=8, stride=1),
            ShuffleUnit(in_channels=64, out_channels=64, groups=8, stride=1),
            nn.ConvTranspose2d(64, self.config.nb_classes, 3, stride=2, padding=1,
                               output_padding=1, bias=False)
        )
        '''
        # for 2x4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            nn.ConvTranspose2d(128, self.config.nb_classes, kernel_size=3, stride=4, padding=1,
                               output_padding=3, bias=False),

        )
        # for 8x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, self.config.nb_classes, kernel_size=3, stride=8, padding=1,
                               output_padding=7, bias=False),
        )
        '''
        '''
        # for 2x2x2x2x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ShuffleUnit(64, 64, groups=8, stride=1),
            ShuffleUnit(64, 64, groups=8, stride=1),
            nn.ConvTranspose2d(64, self.config.nb_classes, kernel_size=3, stride=2, padding=1,
                            output_padding=1, bias=False)
        )
        # for 2x2x4x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            ShuffleUnit(in_channels=128, out_channels=128, groups=8, stride=1),
            nn.ConvTranspose2d(128, self.config.nb_classes, 3, stride=4, padding=1,
                               output_padding=3, bias=False),
        )
        # for 2x8x
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            ShuffleUnit(in_channels=256, out_channels=256, groups=8, stride=1),
            nn.ConvTranspose2d(256, self.config.nb_classes, kernel_size=3, stride=8, padding=1,
                               output_padding=7, bias=False),
        )
        '''
        '''
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            ShuffleUnit(in_channels=self.stage_out_channels[1],
                        out_channels=self.stage_out_channels[1],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[1],
                        out_channels=self.stage_out_channels[1],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[1],
                        out_channels=self.stage_out_channels[1],
                        groups=groups, stride=1),
        )
        self.stage2 = nn.Sequential(
            ShuffleUnit(in_channels= self.stage_out_channels[1],
                        out_channels= self.stage_out_channels[2],
                        groups=groups, stride=2),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[2],
                        groups=groups, stride=1)

        )
        self.stage3 = nn.Sequential(
            ShuffleUnit(in_channels=self.stage_out_channels[2],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=2),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[3],
                        groups=groups, stride=1),
        )
        self.stage4 = nn.Sequential(

            ShuffleUnit(in_channels=self.stage_out_channels[3],
                        out_channels=self.stage_out_channels[4],
                        groups=groups, stride=2),
            ShuffleUnit(in_channels=self.stage_out_channels[4],
                        out_channels=self.stage_out_channels[4],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[4],
                        out_channels=self.stage_out_channels[4],
                        groups=groups, stride=1),
            ShuffleUnit(in_channels=self.stage_out_channels[4],
                        out_channels=self.stage_out_channels[4],
                        groups=groups, stride=1),

        )# 32x down sampling
        '''
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == '__main__':

    input = torch.randn(1, 8, 64, 64)
    conv = conv1x1(8, 32, groups=8)
    output = conv(input)
    print(output.shape)






















