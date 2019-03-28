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
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):
        #print("a", input.shape)
        out = self.depthwise_conv2d(input)
        out = self.dw_bn(out)
        out = self.pointwise_conv2d(out)
        out = self.pw_bn(out)
        #print(out.shape)
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

        gc prelu/ relu
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
        # here dont need project_bn, need to bn with ext_branch
        #self.project_bn = nn.BatchNorm2d(out_channels-in_channels)

        self.ret_bn = nn.BatchNorm2d(out_channels)
        self.ret_gc = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, groups=2, bias=False)
        self.ret_prelu = nn.ReLU(inplace= True)

    def forward(self, input):

        ext_branch = self.pooling(input)

        main_branch = self.depthwise_conv2d(input)
        main_branch = self.dw_bn(main_branch)
        main_branch = self.project_layer(main_branch)

        ret = torch.cat([ext_branch, main_branch], dim=1)
        ret = self.ret_bn(ret)
        ret = self.ret_gc(ret)

        return self.ret_prelu(ret)


class Separable_bt_v1(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate =0.0,
                 ):

        # default decoupled
        super(Separable_bt_v1, self).__init__()

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # self.conv1_prelu = nn.PReLU()
        # a relu
        # Depthwise_conv 3x1 and 1x3
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv3 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=self.internal_channels, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)

        # regularization
        self.dropout = nn.Dropout2d(inplace=True, p=dropout_rate)
        #self.conv4_prelu = nn.PReLU()
    def forward(self, input):

        residual = input
        main = self.conv1(input)
        main = self.conv1_bn(main)
        main = F.relu(main, inplace=True)

        main = self.conv2(main)
        main = self.conv2_bn(main)
        main = self.conv3(main)
        main = self.conv3_bn(main)
        main = self.conv4(main)
        main = self.conv4_bn(main)
        #main = F.relu(main, inplace=True)

        if self.dropout.p != 0:
            main = self.dropout(main)

        #ret = torch.cat([main, residual], dim=1)
        # add or concate+gc
        return F.relu(torch.add(main, residual), inplace=True)
        #return ret

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
        #print(out.shape)
        return F.relu(self.bn(out))

class EDABlock(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3,1), padding=(1,0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1,3), padding=(0,1))
        self.bn1 = nn.BatchNorm2d(k)
        # ConvLayer with dilated_rate padding [(kernel_size-1)/2]*(dilated_rate-1)+1
        # ConvLayer (kernel_size-1)/2
        self.conv3x1_2 = nn.Conv2d(k, k, kernel_size=(3,1), padding=(dilated,0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, kernel_size=(1,3), padding=(0,dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        x = input
        output = self.conv1x1(input)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)
        output = torch.cat([output, x], 1)

        return output


class EDNet(nn.Module):
    def __init__(self,
                 config,
                 down_factor=8,
                 interpolate=True,
                 dilation=True,
                 dropout=False,
                 ):
        super(EDNet, self).__init__()

        self.name = 'EDNNet'
        self.nb_classes = config.nb_classes
        self.down_factor = down_factor
        self.interpolate = interpolate
        self.stage_channels = [-1, 32, 64, 128, 256, 512]
        if dilation == True:
            self.dilation_list = [1, 2, 4, 8, 16]
        else:
            self.dilation_list = [1, 1, 1, 1, 1 ]

        if dropout == True:
            self.dropout_list = [0.01, 0.001]

        if down_factor==8:
            # 8x downsampling
            self.encoder = nn.Sequential(
                conv2d_bn_relu(3, 32, kernel_size=7, stride=2, padding=1, bias=False),
                DownSamplingBlock_v2(in_channels=self.stage_channels[1], out_channels=self.stage_channels[2]),
                #DownSamplingBlock_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
                Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0],), #dropout_rate=self.dropout_list[0] if dropout else None),
                #nn.Conv2d(self.stage_channels[2], self.stage_channels[3], 3, stride=2, padding=1, bias=False),
                Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], dilation= self.dilation_list[0],), #dropout_rate=self.dropout_list[0] if dropout else None),
                #Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                #nn.Conv2d(self.stage_channels[3], self.stage_channels[4], 3, stride=2, padding=1, bias=False),
                DownSamplingBlock_v2(in_channels=self.stage_channels[2], out_channels=self.stage_channels[3]),
            )

            self.layers_list = nn.ModuleList()
            for i in range(5):
                self.layers_list.append(
                    EDABlock(ninput=self.stage_channels[2]+40*i, dilated=self.dilation_list[i], dropprob=0.0)
                )

            #self.gp_context = global_context(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4])
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[3], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[3]),
                    nn.ReLU(inplace=True),
                    Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
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
                    Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], ),
                    Separable_bt_v1(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4], ),
                    nn.ConvTranspose2d(self.stage_channels[4], self.stage_channels[3], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[3]),
                    nn.ReLU(inplace=True),
                    Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    Separable_bt_v1(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3], ),
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),
                    Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    Separable_bt_v1(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    nn.ConvTranspose2d(self.stage_channels[2], self.nb_classes, kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                )
    def forward(self, input):

        encoder_out = self.encoder(input)
        #decoder_out = self.decoder(encoder_out)
        #gpc = self.gp_context(encoder_out)

        #encoder_out_mul = torch.mul(encoder_out, gpc)
        # global_context fusion
        #encoder_out = torch.add(encoder_out, torch.mul(encoder_out, gpc))
        if self.interpolate == True:
            decoder_out = self.project_layer(encoder_out)
        else:
            decoder_out = self.decoder(encoder_out)

        decoder_out = F.interpolate(decoder_out, scale_factor=self.down_factor, mode='bilinear', align_corners=True)

        return decoder_out
