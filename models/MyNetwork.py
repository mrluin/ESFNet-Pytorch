import torch
import torch.nn as nn
import torch.nn.functional as F

class Separabel_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 kernel_size=(3,3),
                 dilation=(1,1),
                 #padding=(1,1),
                 stride=(1,1),
                 bias=False):
        """
        # Note: Default for kernel_size=3,
        for Depthwise conv2d groups should equal to in_channels and out_channels == in_channels
        Only bn after depthwise_conv2d and no no-linear

        padding = (kernel_size-1) / 2
        padding = padding * dilation

        """

        super(Separabel_conv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding= (int((kernel_size[0]-1)/2)*dilation[0],int((kernel_size[1]-1)/2)*dilation[1]),
            dilation= dilation, groups=groups,bias=bias
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
    def forward(self, *input):
        out = self.depthwise_conv2d(*input)
        out = self.dw_bn(out)
        out = self.pointwise_conv2d(out)
        out = self.pw_bn(out)
        return out

def channel_shuffle(input, groups):
    """
    # Note that groups set to channels by default
    if depthwise_conv2d means groups == in_channels thus, channels_shuffle doesn't work for it.
    """
    batch_size, channels, height, width = input.shape
    #groups = channels
    channels_per_group = channels // groups

    input = input.view(batch_size, groups, channels_per_group, height, width)
    input = input.transpose(1,2).contiguous()
    input = input.view(batch_size, -1, height, width)

    return input

class DownSamplingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,):
        """
        # Note: Initial_block from ENet
        Default that out_channels = 2 * in_channels


        Add: channel_shuffle after concatenate
        to be testing

        """
        super(DownSamplingBlock, self).__init__()

        # MaxPooling or AvgPooling
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels-in_channels,
                              kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_bn = nn.BatchNorm2d(out_channels-in_channels)
        self.ret_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        ext_branch = self.pooling(input)
        main_branch = self.conv(input)
        main_branch = self.conv_bn(main_branch)

        ret = self.ret_bn(torch.cat([ext_branch, main_branch], dim=1))

        return channel_shuffle(ret, groups=2)

class DownSamplingBlock_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,):
        """
        # Note: Initial_block from ENet
        Default that out_channels = 2 * in_channels
        compared to downsamplingblock_v1, change conv3x3 into Depthwise and projection_layer

        Add: channel_shuffle after concatenate
        to be testing

        """
        super(DownSamplingBlock_v2, self).__init__()

        # MaxPooling or AvgPooling
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels-in_channels,
        #                      kernel_size=3, stride=2, padding=1, bias=False)
        self.depthwise_conv2d = nn.Conv2d(in_channels= in_channels, out_channels=in_channels,
                                          kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.project_layer = nn.Conv2d(in_channels= in_channels, out_channels=out_channels-in_channels,
                                       kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels-in_channels)

        self.ret_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        ext_branch = self.pooling(input)
        main_branch = self.depthwise_conv2d(input)
        main_branch = self.project_layer(self.dw_bn(main_branch))

        ret = self.ret_bn(torch.cat([ext_branch, self.project_bn(main_branch)], dim=1))

        return channel_shuffle(ret, groups=2)


class ConvTranspose2d_bn_relu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 output_padding,
                 bias):
        super(ConvTranspose2d_bn_relu, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                         bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        output = self.deconv(input)

        return F.relu(self.bn(output))


class Separable_bt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 decouple=False):
        """
        # Note: for bottleNeck, 3x3 kernel stride=2 and in residual branch apply avgpooling
        """
        super(Separable_bt, self).__init__()

        self.stride = stride
        self.decoupled = decouple
        self.internal_channels = in_channels // 4
        # main branch

        self.main_compress = nn.Conv2d(in_channels=in_channels, out_channels=self.internal_channels,
                                       kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.main_expand = nn.Conv2d(in_channels=self.internal_channels, out_channels=out_channels,
                                     kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        if decouple == False:
            self.main_sep = Separabel_conv2d(in_channels=self.internal_channels, out_channels=self.internal_channels,
                                             kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                                             dilation=(dilation, dilation), groups=self.internal_channels, bias=False)
        if decouple == True:
            self.main_sep_1 = Separabel_conv2d(in_channels=self.internal_channels, out_channels=self.internal_channels,
                                               kernel_size=(kernel_size, 1), stride=(stride,1),
                                               dilation=(dilation, 1), groups=self.internal_channels, bias=False)
            self.main_sep_2 = Separabel_conv2d(in_channels=self.internal_channels, out_channels=self.internal_channels,
                                               kernel_size=(1, kernel_size), stride=(1,stride),
                                               dilation=(1, dilation), groups=self.internal_channels, bias=False)
        if stride == 2:
            self.residual_pooling = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):

        residual = input
        main = self.main_compress(input)
        if self.decoupled:
            main = self.main_sep_1(main)
            main = self.main_sep_2(main)
        else:
            main = self.main_sep(main)
        main = self.main_expand(main)

        if self.stride == 2:
            residual = self.residual_pooling(residual)

        return F.relu(residual + main)


class Separable_bt_v1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 ):

        # default decoupled
        super(Separable_bt_v1, self).__init__()

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        # Depthwise_conv 3x1 and 1x3
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv3 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        residual = input
        main = self.conv1(input)
        main = self.conv2(F.relu(self.conv1_bn(main), inplace=True))
        main = self.conv3(self.conv2_bn(main))
        main = self.conv4(self.conv3_bn(main))

        return F.relu(self.conv4_bn(main)+residual)


class Separable_non_bt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 decouple=False):
        """
        # Note: for non_bottleneck, conv_layer in main_branch stride=1
        """
        super(Separable_non_bt, self).__init__()

        self.decouped = decouple

        if decouple == False:
            # main_branch
            self.main_conv1 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(kernel_size,kernel_size),
                dilation=(1,1), stride=(stride,stride), bias=False
            )
            self.main_conv2 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(kernel_size, kernel_size),
                dilation=(dilation, dilation), stride=(stride,stride),bias= False
            )
        #self.conv2_bn = nn.BatchNorm2d(out_channels)
        if decouple == True:

            self.main_conv1_1 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(kernel_size, 1),
                dilation=(1,1), stride=(stride, 1), bias=False
            )
            #self.main_conv1_1_bn = nn.BatchNorm2d(out_channels)
            self.main_conv1_2 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(1, kernel_size),
                dilation=(1,1), stride=(1,stride), bias=False
            )
            #self.main_conv1_2_bn = nn.BatchNorm2d(out_channels)
            self.main_conv2_1 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(kernel_size, 1),
                dilation=(dilation, 1), stride=(stride,1), bias=False
            )
            #self.main_conv2_1_bn = nn.BatchNorm2d(out_channels)
            self.main_conv2_2 = Separabel_conv2d(
                in_channels=in_channels, out_channels=out_channels,
                groups=in_channels, kernel_size=(1, kernel_size),
                dilation=(1, dilation), stride=(1, stride)
            )

        # residual_branch

    def forward(self, input):

        residual = input
        main_out = input
        if self.decouped == True:
            main_out = F.relu(self.main_conv1_1(main_out))
            main_out = F.relu(self.main_conv1_2(main_out))
            main_out = F.relu(self.main_conv2_1(main_out))
            main_out = self.main_conv2_2(main_out)
        else:
            main_out = F.relu(self.main_conv1(main_out))
            main_out = self.main_conv2(main_out)

        return F.relu(residual + main_out)

class Separable_non_bt_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 decouple=True):
        """
        # Note: compared to Separable_non_bt_v1 change the first main_conv1 from separable_conv2d to only dw and then channel_shuffle
        or others

        default decoupled
        """
        super(Separable_non_bt_v2, self).__init__()

        self.decouple = decouple
        '''
        self.main_conv1 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=(kernel_size, kernel_size), stride=(stride,stride),
                                           dilation=(dilation,dilation), groups=in_channels, bias=False)
                                          '''
        # 这个是无效的
        if decouple == False:
            self.main_conv1 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size), stride=(stride,stride),
                                               dilation=(dilation,dilation), groups=in_channels, bias=False)
            # here add channel_shuffle

            self.main_conv2 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(kernel_size,kernel_size), stride=(stride, stride),
                                               dilation=(dilation, dilation), groups=in_channels, bias=False)
        if decouple == True:
            self.main_conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size, 1), stride=(stride, 1),
                                          dilation=(1, 1), groups=in_channels, bias=False,
                                          padding =(1, 0))
            self.main_conv1_1_bn = nn.BatchNorm2d(out_channels)
            # here add channels_shuffle
            self.main_conv1_2 = Separabel_conv2d(in_channels= in_channels, out_channels=out_channels,
                                                 groups=in_channels, kernel_size=(1, kernel_size), dilation=(1, 1),
                                                 stride=(1, stride), bias=False)
            # here add relu
            self.main_conv2_1 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size, 1), stride=(stride, 1),
                                          dilation=(dilation, 1), groups=in_channels, bias=False,
                                          padding=(1*dilation, 0))
            self.main_conv2_1_bn = nn.BatchNorm2d(out_channels)
            # here add channels_shuffle
            self.main_conv2_2 = Separabel_conv2d(in_channels= in_channels, out_channels=out_channels,
                                                 groups=in_channels, kernel_size=(1, kernel_size), dilation=(1, dilation),
                                                 stride=(1, stride), bias=False)
    def forward(self, input):

        #print(input.shape)
        residual = input
        main = input
        if self.decouple:
            main = self.main_conv1_1(main)
            main = self.main_conv1_1_bn(main)
            main = self.main_conv1_2(main)

            main = self.main_conv2_1(F.relu(main))
            main = self.main_conv2_1_bn(main)
            main = self.main_conv2_2(main)

        else:
            main = self.main_conv1(main)
            main = self.main_conv2(main)

        #print(main.shape, residual.shape)

        return F.relu(main + residual)


class Separable_non_bt_v3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation,
                 decouple=True):
        """
        # Note: compared to Separable_non_bt_v2 change the pw_conv2d in Sep into channel_shuffle
        or others

        default decoupled
        """
        super(Separable_non_bt_v3, self).__init__()

        self.decouple = decouple
        '''
        self.main_conv1 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=(kernel_size, kernel_size), stride=(stride,stride),
                                           dilation=(dilation,dilation), groups=in_channels, bias=False)
                                          '''
        # 这个是无效的
        if decouple == False:
            #self.main_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            #                            kernel_size=(kernel_size, kernel_size), stride=(stride,stride),
            #                            dilation=(1, 1), groups=in_channels, bias=False)
            self.main_conv1 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size), stride=(stride,stride),
                                               dilation=(dilation,dilation), groups=in_channels, bias=False)
            #self.main_conv1_bn = nn.BatchNorm2d(out_channels)
            # here add channel_shuffle

            self.main_conv2 = Separabel_conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(kernel_size,kernel_size), stride=(stride, stride),
                                               dilation=(dilation, dilation), groups=in_channels, bias=False)
        if decouple == True:
            self.main_conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size, 1), stride=(stride, 1),
                                          dilation=(1, 1), groups=in_channels, bias=False,
                                          padding=(1,0))
            self.main_conv1_1_bn = nn.BatchNorm2d(out_channels)
            # here add channels_shuffle
            self.main_conv1_2 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
                                          groups=in_channels, kernel_size=(1, kernel_size), dilation=(1, 1),
                                          stride=(1, stride), bias=False,
                                          padding=(0,1))
            self.main_conv1_2_bn = nn.BatchNorm2d(out_channels)
            # here add channels_shuffle
            # here add relu ?
            self.main_conv2_1 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
                                          kernel_size=(kernel_size, 1), stride=(stride, 1),
                                          dilation=(dilation, 1), groups=in_channels, bias=False,
                                          padding=(dilation,0))
            self.main_conv2_1_bn = nn.BatchNorm2d(out_channels)
            # here add channels_shuffle
            self.main_conv2_2 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
                                          groups=in_channels, kernel_size=(1, kernel_size), dilation=(1, dilation),
                                          stride=(1, stride), bias=False,
                                          padding=(0,dilation))
            # here ad channels_shuffle?
            self.main_conv2_2_bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        residual = input
        main = input
        if self.decouple:
            main = self.main_conv1_1(main)
            main = self.main_conv1_1_bn(main)
            main = self.main_conv1_2(main)
            main = self.main_conv1_2_bn(main)
            main = self.main_conv2_1(main)
            main = self.main_conv2_1_bn(main)
            main = self.main_conv2_2(main)
            main = self.main_conv2_2_bn(main)

        else:
            main = self.main_conv1(main)
            main = self.main_conv2(main)

        return F.relu(main + residual)


class MyNetwork(nn.Module):
    def __init__(self,
                 config,
                 decoupled=False):
        super(MyNetwork, self).__init__()

        self.name='MyNetwork_{}'.format('decoupled' if decoupled else 'undecoupled')
        self.nb_classes = config.nb_classes

        self.stage_channels=[-1, 32, 64, 128, 256]

        # encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2,padding=1, bias=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            Separable_non_bt_v2(in_channels=self.stage_channels[2],
                                out_channels=self.stage_channels[2],decouple=decoupled),
            nn.Conv2d(self.stage_channels[2], self.stage_channels[3], 3, stride=2, padding=1, bias=False),
            Separable_non_bt_v2(in_channels=self.stage_channels[3],
                                out_channels=self.stage_channels[3],decouple=decoupled),
            nn.Conv2d(self.stage_channels[3], self.stage_channels[4], 3, stride=2, padding=1, bias=False),
            Separable_non_bt_v2(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],decouple=decoupled),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[3]),
            nn.ReLU(inplace=True),
            Separable_non_bt_v2(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],decouple=decoupled),
            nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[2]),
            nn.ReLU(inplace=True),
            Separable_non_bt_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],decouple=decoupled),
            Separable_non_bt_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],decouple=decoupled),
            nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        )
    def forward(self, input):

        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

class MyNetwork_bt(nn.Module):
    def __init__(self,
                 config,
                 decoupled):
        super(MyNetwork_bt, self).__init__()

        self.name = 'MyNetwork_bt_{}'.format('decoupled' if decoupled else 'undecoupled')
        self.nb_classes = config.nb_classes

        self.stage_channels = [-1, 32, 64, 128, 256]

        self.encoder = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            DownSamplingBlock(in_channels=3, out_channels=64),
            Separable_bt(in_channels=self.stage_channels[2],
                             out_channels=self.stage_channels[2], decouple=decoupled),
            #nn.Conv2d(self.stage_channels[2], self.stage_channels[3], 3, stride=2, padding=1, bias=False),
            DownSamplingBlock(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
            Separable_bt(in_channels=self.stage_channels[3],
                             out_channels=self.stage_channels[3], decouple=decoupled),
            #nn.Conv2d(self.stage_channels[3], self.stage_channels[4], 3, stride=2, padding=1, bias=False),
            DownSamplingBlock(in_channels=self.stage_channels[3], out_channels=self.stage_channels[4]),
            Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                             decouple=decoupled),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[3]),
            nn.ReLU(inplace=True),
            Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                             decouple=decoupled),
            nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[2]),
            nn.ReLU(inplace=True),
            Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                             decouple=decoupled),
            Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                             decouple=decoupled),
            nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False)
        )
    def forward(self, input):

        encoder_out = self.encoder(input)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

class MyNetwork_bt_v1(nn.Module):
    def __init__(self,
                 config,
                 down_factor=8,
                 interpolate=True
                 ):
        super(MyNetwork_bt_v1, self).__init__()

        self.name = 'MyNetwork_bt_v1_{}x_{}'.format(down_factor, 'cz' if interpolate else 't')
        self.nb_classes = config.nb_classes
        self.down_factor = down_factor
        self.interpolate = interpolate
        self.stage_channels = [-1, 32, 64, 128, 256, 512]

        if down_factor==8:
            # 8x downsampling
            self.encoder = nn.Sequential(
                #nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                DownSamplingBlock_v2(in_channels=3, out_channels=self.stage_channels[2]),
                Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                #nn.Conv2d(self.stage_channels[2], self.stage_channels[3], 3, stride=2, padding=1, bias=False),
                DownSamplingBlock_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
                Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                #nn.Conv2d(self.stage_channels[3], self.stage_channels[4], 3, stride=2, padding=1, bias=False),
                DownSamplingBlock_v2(in_channels=self.stage_channels[3], out_channels=self.stage_channels[4]),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],),
            )
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[4], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[3]),
                    nn.ReLU(inplace=True),
                    Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False)
                )
        elif down_factor==16:
            # 16x downsampling
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                #DownSamplingBlock(in_channels=3, out_channels=self.stage_channels[2]),
                Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                nn.Conv2d(self.stage_channels[2], self.stage_channels[3], 3, stride=2, padding=1, bias=False),
                #DownSamplingBlock(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
                Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                nn.Conv2d(self.stage_channels[3], self.stage_channels[4], 3, stride=2, padding=1, bias=False),
                #DownSamplingBlock(in_channels=self.stage_channels[3], out_channels=self.stage_channels[4]),
                Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], ),
                nn.Conv2d(self.stage_channels[4], self.stage_channels[5], 3, stride=2, padding=1, bias=False),
                #DownSamplingBlock(in_channels=self.stage_channels[4], out_channels=self.stage_channels[5]),
                Separable_bt_v1(in_channels=self.stage_channels[5], out_channels=self.stage_channels[5], ),
                Separable_bt_v1(in_channels=self.stage_channels[5], out_channels=self.stage_channels[5], ),
                Separable_bt_v1(in_channels=self.stage_channels[5], out_channels=self.stage_channels[5], ),
                Separable_bt_v1(in_channels=self.stage_channels[5], out_channels=self.stage_channels[5], ),
                Separable_bt_v1(in_channels=self.stage_channels[5], out_channels=self.stage_channels[5], ),
            )
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[5], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[5], self.stage_channels[4], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[4]),
                    nn.ReLU(inplace=True),
                    Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], ),
                    Separable_bt(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], ),
                    nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[3]),
                    nn.ReLU(inplace=True),
                    Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    Separable_bt(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    Separable_bt(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                )
    def forward(self, input):

        encoder_out = self.encoder(input)
        #decoder_out = self.decoder(encoder_out)
        if self.interpolate == False:
            decoder_out = self.project_layer(encoder_out)
        else:
            decoder_out = self.decoder(encoder_out)

        decoder_out = F.interpolate(decoder_out, scale_factor=self.down_factor, mode='bilinear', align_corners=True)

        return decoder_out

if __name__ == '__main__':

    pass
