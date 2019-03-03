import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.BaseModel import BaseModel

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            self.conv = nn.Conv2d(ninput, noutput-ninput,
                                  kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            self.conv = nn.Conv2d(ninput, noutput,
                                  kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, input):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
        else:
            output = self.conv(input)

        output = self.bn(output)
        return F.relu(output)

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

class EDANet(BaseModel):
    def __init__(self, config):
        super(EDANet, self).__init__()
        
        self.name='EDANet'
        self.nb_classes = config.nb_classes

        self.layers = nn.ModuleList()
        # for stage1
        self.dilation1 = [1,1,1,2,2]
        # for stage2
        self.dilation2 = [2,2,4,4,8,8,16,16]

        self.layers.append(DownsamplerBlock(3,15))
        self.layers.append(DownsamplerBlock(15,60))

        for i in range(5):
            self.layers.append(EDABlock(60+40*i, self.dilation1[i]))

        self.layers.append(DownsamplerBlock(260,130))

        for j in range(8):
            self.layers.append(EDABlock(130+40*j, self.dilation2[j]))

        # projection layer
        self.project_layer = nn.Conv2d(450, self.nb_classes, kernel_size=1)

        self.weights_init()


    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.kaiming_normal_(m.weight.data)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self,x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        # bilinear interpolation x8
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=True)

        # bilinear interpolation x2
        #if not self.training:
        #    output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output

if __name__ == '__main__':

    input = torch.randn(1,3,512,512)
    # for the inference only
    model = EDANet().eval()
    print(model)
    output = model(input)
    print(output.shape)













