import torch
import numpy as np
import torch.nn as nn
import time
from utils.util import AverageMeter
# normal conv2d stacked 50 layers


class separable_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        """
            in_channels = out_channels = groups, using thcunn backends of pytorch, otherwise using cudnn
        """
        super(separable_conv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size, stride=1,
                                          padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, dilation=dilation, groups=1)
    def forward(self, input):
        output = self.depthwise_conv2d(input)
        output = self.pointwise_conv2d(output)
        return output


class normal_convnet(nn.Module):
    def __init__(self):
        super(normal_convnet, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(3, 256, 3, padding=1, groups=1))
        for index in range(48):
            self.layer_list.append(nn.Conv2d(256, 256, 3, padding=1, groups=1))
        self.layer_list.append(nn.Conv2d(256, 10, 3, padding=1, groups=1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class sep_convnet(nn.Module):
    def __init__(self):
        super(sep_convnet, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(3, 256, 3, padding=1, groups=1))
        for index in range(48):
            self.layer_list.append(separable_conv2d(256,256,3,1))
        self.layer_list.append(nn.Conv2d(256, 10, 3, padding=1, groups=1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)

        return x

def inference_test_both():

    #torch.manual_seed(1)
    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic=True

    random_input = torch.randn((1, 3, 256, 256))
    random_output = torch.randint(low=0, high=10, size=(1,256,256)) # for 0,1,2,3,4,5,6,7,8,9
    #random_output = torch.randn((1,3,256,256))

    net1 = normal_convnet().to('cuda:0')
    net2 = sep_convnet().to('cuda:0')

    # params = 0
    #params_normal = sum(p.numel() for p in normal_net.parameters() if p.requires_grad)
    #print("Trainable Parameters :", params_normal)

    #criterion = nn.MSELoss().to('cuda:0')
    criterion = nn.CrossEntropyLoss().to('cuda:0')

    #optimizer_1 = torch.optim.Adam(params=net1.parameters(), lr=0.1)
    #optimizer_2 = torch.optim.Adam(params=net2.parameters(), lr=0.1)
    optimizer_1 = torch.optim.SGD(params=net1.parameters(),lr=0.1)
    optimizer_2 = torch.optim.SGD(params=net2.parameters(),lr=0.2)
    #

    cost1 = AverageMeter()
    cost2 = AverageMeter()
    print("Simulate Training ... ...")


    input1 = random_input.to('cuda:0')
    target1 = random_output.to('cuda:0')
    torch.cuda.synchronize()
    tic = time.time()
    optimizer_1.zero_grad()
    output1 = net1(input1)
    loss = criterion(output1, target1)
    loss.backward()
    optimizer_1.step()
    torch.cuda.synchronize()
    cost1.update(time.time() - tic)

    #print(dw_net)
    #params_dw = sum(p.numel() for p in normal_net.parameters() if p.requires_grad)
    #print("Trainable Parameters :", params_dw)
    #optimizer_dw = torch.optim.Adam(params=dw_net.parameters())

    input2 = random_input.to('cuda:0')
    target2 = random_output.to('cuda:0')
    torch.cuda.synchronize()
    tic = time.time()
    optimizer_1.zero_grad()
    output2 = net2(input2)
    loss = criterion(output2, target2)
    loss.backward()
    optimizer_2.step()
    torch.cuda.synchronize()
    cost2.update(time.time() - tic)

    print("Done for All !")

    print("Trainable Parameters:\n"
          "Normal_conv2d: {}\n"
          "Sep_conv2d    : {}".format(parameters_sum(net1), parameters_sum(net2)))
    print("Inference Time cost:\n"
          "Normal_conv2d: {}s\n"
          "Sep_conv2d    : {}s".format(cost1._get_sum(), cost2._get_sum()))


def parameters_sum(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    '''
    #for separable conv2d 
    #Duration:  0.07718038558959961 for normal 
    #Duration:  0.041310787200927734 for separable conv2d including depthwise and pointwise
    input = torch.randn(size=(1,256,256,256))
    target = torch.randn(size=(1,256,256,256))

    m1 = nn.Conv2d(256,256,3,padding=1, groups=1,bias=False).to('cuda:0')
    m2 = nn.Conv2d(256,256,3,padding=1, groups=256,bias=False).to('cuda:0')
    m2_p = nn.Conv2d(256,256,1,padding=0, groups=1,bias=False).to('cuda:0')
    criterion = nn.MSELoss().to('cuda:0')
    optimizer_m1 = torch.optim.SGD(params=m1.parameters(),lr=0.1)
    optimizer_m2 = torch.optim.SGD(params=m2.parameters(),lr=0.1)

    tic = time.time()
    input1 = input.to('cuda:0')
    target1 = target.to('cuda:0')
    out1 = m1(input1)
    loss = criterion(out1, target1)
    optimizer_m1.zero_grad()
    loss.backward()
    optimizer_m1.step()
    print("Duration: ", time.time()-tic)

    tic = time.time()
    input2 = input.to('cuda:0')
    target2 = target.to('cuda:0')
    out2 = m2(input2)
    out2 = m2_p(out2)
    loss = criterion(out2, target2)
    optimizer_m2.zero_grad()
    loss.backward()
    optimizer_m2.step()
    print("Duration: ", time.time()-tic)
    '''
    inference_test_both()
    """
    output: for 50 layers stacked up and run 100 iterations
    Normal_conv2d: 17.160080671310425s
    sep_conv2d    : 7.4953773021698s
    """
    """
    output: net1 trainable parameters:  29504000
            net2 trainable parameters:  3417600
            
            29504000 / 3417600 = 8.6
    net1 = normal_convnet()
    net2 = sep_convnet()
    print("net1 trainable parameters: ", parameters_sum(net1))
    print("net2 trainable parameters: ", parameters_sum(net2))
    
    """
