import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.util import cropped_dataset
from configs.config import MyConfiguration
from data.dataset import MyDataset
from torch.utils.data import DataLoader
from models import ERFNet
import time



# initial seed for generating random numbers
#print(torch.initial_seed())
# return the random number generator state as torch.ByteTensor
#print(torch.get_rng_state())



#config = MyConfiguration()

#monitor = config.monitor
#print(monitor)

"""
    # BatchNorm2d test
    

# nte 64 channels in
bn = nn.BatchNorm2d(64)
#print(bn)
bn.weight.data.fill_(1)
bn.bias.data.fill_(0)
#print(bn.weight)
#print(bn.bias.data)

x = torch.rand(6, 64, 224, 224)
# channel <-> batch_size
tmp = x.permute(1,0,2,3).reshape(64, -1)
mu = tmp.mean(dim=1).reshape(1,64,1,1)
sigma = tmp.std(dim=1).reshape(1,64,1,1)
#mu = x.mean(dim=(0,2,3), keepdim=True)
#sigma = x.std(dim=(0,2,3), keepdim=True)         # argument 'dim' must be int, not tuple

x_ = (x-mu) / (sigma+1e-5)
# x_ normalized data   approximately equal to bn(x_)
print(x_.shape)
print((bn(x_)- x_).abs().max())
"""

"""
    # Visualization
"""
'''
config = MyConfiguration()
print(config.__dict__['add_section'])

a = torch.randn(2,4)
print(a.device)
'''
#print(torch.tensor([3]) / torch.tensor([3]))

"""
atten = torch.randn(1,13)
feat = torch.randn(13, 1024, 7, 7)

# want to do sum_i(atten[0][i] * feat[i]) 即atten的13个值分别乘feat的第一个维度
atten = atten.view(13, 1,1,1)
#output = atten[0]* feat
output = atten * feat
#output = torch.einsum('i,ijkl', atten[0], feat)

print(output.shape)
"""

"""
        ----  dataset with filename  ----
        
class MyDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, index):
        x = torch.randn(3, 24, 24)
        y = torch.randint(0, 10, (1,))
        return x, y,  'lala'

    def __len__(self):
        return len(self.image_paths)


dataset = MyDataset(['']*100)
loader = DataLoader(dataset, batch_size=10)
x, y, s = next(iter(loader))
print(s)
"""

"""
from torch.utils.checkpoint import checkpoint_sequential
model = nn.Sequential(
    nn.Linear(100,50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20,5),
    nn.ReLU()
)
input_var = torch.randn(1,100)
input_var.requires_grad=True
segments = 2
#print(model._modules)  # orderedDict [(index, module)]
modules = [module for k, module in model._modules.items()]
#print(modules)
out = checkpoint_sequential(modules, segments, input_var)
#print(out)
model.zero_grad()
out.sum().backward()
output_checkpointed = out.data.clone()
grad_checkpointed= {}
for name, param in model.named_parameters():
    grad_checkpointed[name] = param.grad.data.clone()

# print(grad_checkpointed)
# non-checkpointed one 
original= model
x = input_var.data
x.requires_grad=True
out = original(x)
out_not_checkpointed = out.data.clone()

original.zero_grad()
out.sum().backward()
grad_not_checkpointed={}
for name, param in model.named_parameters():
    grad_not_checkpointed[name] = param.grad.data.clone()
"""

'''
from PIL import Image

logic = torch.randn(size=(2, 224, 224))
#print(logic.shape)
pred = torch.argmax(logic, dim=0)
#print(pred)
class_to_mask = {
    0: 0,
    1: 255,
}

for k in class_to_mask:
    pred[pred == k] = class_to_mask[k]
    
#print(pred)
#pred = torch.cat((pred,pred,pred,), dim=0)

#pred = np.asarray(pred, dtype=np.uint8)
#print(pred.shape)
#print(pred.dtype)
#pred = Image.fromarray(pred)
#pred.save('./image.png')
print(pred)
valid = (pred > 0).long()

pred = (valid * pred)
print(torch.sum(valid*pred))
print(torch.sum(valid).float() / torch.sum(valid*pred).float())

'''
'''
x = torch.zeros(1, requires_grad=True)


with torch.no_grad():
    # here only a reference when use to calculate requires_grad=False
    # use .detach() or .clone() 
    z = x
    '''
'''
m1 = torch.nn.Conv1d(256, 256, 3, groups=1, bias=False).cuda()
m2 = torch.nn.Conv1d(256, 256, 3, groups=256, bias=False).cuda()
a = torch.randn(1,256,5, device='cuda')
b1 = m1(a)
b2 = m2(a)
print(b1.grad_fn)
print(b2.grad_fn)
print(b1.grad_fn.next_functions)
print(b2.grad_fn.next_functions)

config = MyConfiguration()
dataset = MyDataset(subset='test', config=config)

data, target, filename = next(iter(dataset))
print(data.shape)
print(target.shape)
print(filename)

dataloader = DataLoader(dataset=dataset, batch_size= config.batch_size,
                        shuffle=False, num_workers=2)

data, target, filename = next(iter(dataloader))
print(data.shape)
print(target.shape)
print(filename)
'''
'''
config = MyConfiguration()
cropped_dataset(config=config, subset='train')
'''
'''
filename= '0.1.png'
split_filename = filename.split('.')[0]+'.'+filename.split('.')[1]
print(split_filename)
'''

'''
# convert to onehot label
def to_onehot(y, nb_classes):
    # zeros param ->shape
    y_onehot = torch.zeros(y.size(0), nb_classes)
    y_onehot.scatter_(1, y.view(-1,1).long(), 1).float()

    return y_onehot


y = torch.tensor([0, 1, 2, 2])
y_enc = to_onehot(y, 3)
print('one-hot encoding:\n', y_enc)
#print(pred_onehot)

Z = torch.tensor([[-0.3,  -0.5, -0.5],
                  [-0.4,  -0.1, -0.5],
                  [-0.3,  -0.94, -0.5],
                  [-0.99, -0.88, -0.5]])
# tensor.t() transposed
#print(Z.t())
def softmax(z):
    return (torch.exp(z.t()) / torch.sum(torch.exp(z), dim=1)).t()

smax = softmax(Z)
print(smax)

def to_classlabel(z):
    return torch.argmax(z, dim=1)

print('predicted class labels: ', to_classlabel(smax))
print('true class labels: ', to_classlabel(y_enc))


def cross_entropy(softmax, y_target):
    return - torch.sum(torch.log(softmax) * (y_target), dim=1)

xent = cross_entropy(smax, y_enc)
print('Cross Entropy:', xent)

#
F.nll_loss(torch.log(smax), y, reduction='none')
F.cross_entropy(Z, y, reduction='none')

#
F.cross_entropy(Z, y)
torch.mean(cross_entropy(smax, y_enc))
'''
'''
input = torch.zeros(1,3,512,512)
print(input.shape)
downsampled = F.upsample(input, size=(1, 3, input.size(2)*2, input.size(3)*2))
print(downsampled.shape)
'''

'''
class Discrim(nn.Module):
    channels, maxpool_mask = [13, 64, 192, 384, 256, 256], [1, 1, 0, 0, 1]
    ker_size, strd, pad = [2, 5, 3, 3, 3], [2, 1, 1, 1, 1], [0, 2, 1, 1, 1]

    def __init__(self, classes=13, conv_layers=5):
        super(Discrim, self).__init__()
        self.classes = classes
        conv_features = []
        for index in range(conv_layers):
            conv_features.append(nn.Conv2d(Discrim.channels[index], Discrim.channels[index + 1],
                                           kernel_size=Discrim.ker_size[index],
                                           stride=Discrim.strd[index],
                                           padding=Discrim.pad[index],
                                           bias=False))
            conv_features.append(nn.BatchNorm2d(Discrim.channels[index + 1]))
            conv_features.append(nn.ReLU(inplace=True))
            if Discrim.maxpool_mask[index] == 1:
                conv_features.append(nn.MaxPool2d(3, stride=2, padding=1))
            else:
                conv_features.append(nn.ReLU(inplace=True))

        self.features = nn.ModuleList(conv_features[layer_num] for layer_num in range(4 * conv_layers))

    def forward(self, x):
        out = self.features(x)
        return out


net = Discrim()
'''
'''
input = torch.tensor([[[1,2],
                       [3,4]],
                      [[5,6],
                       [7,8]],
                      [[9,10],
                       [11,12]],
                      [[13,14],
                       [15,16]]])
#input.shape = 4, 2, 2
output = torch.tensor([[1,2,5,6],
                       [3,4,7,8],
                       [9,10,13,14],
                       [11,12,15,16]])
output1 = torch.tensor([[1,2],
                        [5,6],
                        [3,4],
                        [7,8],
                        [9,10],
                        [13,14],
                        [11,12],
                        [15,16]])
# shape=4, 4
#input = input.view(1,4,4)
#print(input)
#input_windows = input.unfold(1, 2, 2)
#input_windows = input_windows.unfold(2, 2, 2)
#print(input_windows)
#print(input_windows.contiguous().view(4, 4))
#print(input.shape)
A = input.view(2, 2, 2, 2)
print(A)
print(A.permute(0,2,1,3))
print(A.permute(0,2,1,3).contiguous().shape)
'''

'''
# for torch.tensor.unfold
x = torch.arange(1, 17).float().view(1, 1, 4, 4)
# x.shape = 1, 1, 4, 4
print(x)
input_windows = x.unfold(2, 2, 2)
# input_windows.shape = 1, 1, 2, 4, 2
#print(input_windows)
input_windows = input_windows.unfold(3 ,2, 2)
# input_windows.shape = 1, 1, 2, 2, 2, 2
#print(input_windows.shape)
input_windows = input_windows.contiguous().view(4,4)
print(input_windows)
'''
'''
# grad.zero_()
from torch.autograd.variable import Variable
x = Variable(torch.Tensor([[0]]), requires_grad=True)
for t in range(5):
    y = x.sin()
    y.backward()

print(x.grad) # output x.grad=5 it just do accumulate without grad.zero_()

for t in range(5):
    x.grad.data.zero_()
    y = x.sin()
    y.backward()
print(x.grad) # output x.grad=1
# set grad.zero_() so that the gradients computed previously do not interfere
# with the ones you are currently computing
'''
'''
a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a+1
c = b+1
#b.retain_grad()
c.backward(torch.ones_like(c))
print(a.grad)
print(a.is_leaf)
print(b.grad)
print(b.is_leaf)
'''
'''
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 5)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.conv2(output)
        return output


a = torch.randn(size=(1, 3, 5, 5))
model = model()
output = model(a)
print(output.shape)
'''
'''
config = MyConfiguration()
dataset = MyDataset(config=config, subset='train')
data, target = next(iter(dataset))
print(d ata.shape)
print(target.shape)


a = torch.zeros(size=(224, 224, 1))
map = np.asarray(a, dtype=np.uint8)
map = Image.fromarray(map)
map.save('test.png')
'''

'''
class_sample_counts = [568330.0, 43000.0, 34900.0, 20910.0, 14590.0, 9712.0]
# weights.shape = [num_classes]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
# samples_weights.shape = [num_samples] containing [weights[y0], weights[y1] ...]
samples_weights = weights[train_targets] # 每个sample的weight
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True
)
'''
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

random_input = torch.randn(10, requires_grad=True)
random_target = torch.randn(1, requires_grad=True)


ckpt = torch.load('model.pth')
net = Net()
net.load_state_dict(ckpt)

for param in net.parameters():
    print(param.type())

input = random_input.to('cuda:0')
print(input.type())

output = net(input)
'''

'''
from utils.util import AverageMeter
config = MyConfiguration()
dataset = MyDataset(config=config, subset='train')
dataloader = DataLoader(dataset=dataset, batch_size=64, num_workers=2)
data_time = AverageMeter()
tic = time.time()
for index, (data, target) in enumerate(dataloader):
    data_time.update(time.time()-tic)
    tic = time.time()

print(data_time.average())

'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = nn.ModuleList([nn.Conv2d(3, 6, 3),
                                    nn.BatchNorm2d(6, ),
                                    nn.Conv2d(6, 10, 3),
                                    nn.Conv2d(10, 10, 3)])
        self.fcs = nn.Sequential(nn.Linear(320, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 5),
                                 nn.ReLU(),
                                 nn.Linear(5, 1))

    def forward(self, input):
        out = self.convs[0](input)
        out = self.convs[1](out)
        out = self.convs[2](out)
        out = self.convs[3](out)
        out = out.view(-1, )
        out = self.fcs(out)

        return out


model = Net()
loss = nn.L1Loss()
target = torch.ones(1)


for name, param in model.named_parameters():
    if name == 'convs.0.bias' or name == 'fcs.2.weight':
        # only conv.0.bias and fcs.2.weight = True
        param.requires_grad=True
    else:
        param.requires_grad=False
# output: conv ... fc ... weights and bias

old_state_dict = {}

print(model.state_dict())
#model parameter weight and bias value dict key: value
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

#print(model.state_dict().keys())
#print(old_state_dict.keys())
optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad ,model.parameters()), lr=0.001)
#model.eval()
for epoch in range(5):

    x = torch.rand(2,3,10,10)
    out = model(x)
    output = loss(out, target)
    output.backward()
    optimizer.step()

new_state_dict = {}
# 看看是否有改变 正常只有convs.0.bias和fcs.2.weight有变化
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

count = 0
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print("Diff in {}".format(key))
        count+=1

print(count)