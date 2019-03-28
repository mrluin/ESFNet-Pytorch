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

#config = MyConfiguration()

#monitor = config.monitor
#print(monitor)


'''
config = MyConfiguration()
print(config.__dict__['add_section'])

a = torch.randn(2,4)
print(a.device)
'''
#print(torch.tensor([3]) / torch.tensor([3]))


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





'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,5,bias=False)
        self.fc2 = nn.Linear(5,5,bias=False)
        self.fc3 = nn.Linear(5,1,bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(10, 3, 3, padding=1, bias= False)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.dropout(out)
        return out



random_input = torch.randn((1,3,10,10), requires_grad=True)
random_target = torch.randn((3,10,10), requires_grad=True)

model = ConvNet()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
loss = nn.MSELoss()
#new_param = nn.Parameter(data=torch.randn(10,5), requires_grad=True)

output = model(random_input)
loss = loss(output, random_target)
loss.backward()
'''

'''
for name, param in model.named_parameters():
    print(name,param)
print("-------------------")
print(model.state_dict())
'''
'''
old_state_dict = {}
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key]

print(old_state_dict)


for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

print(old_state_dict)
#for param in model.parameters():
#    print(param.grad)


print(optimizer.state_dict())
optimizer.zero_grad()
#print(optimizer.state_dict())
output = model(random_input)
loss = loss(output, random_target)
loss.backward()

optimizer.step()

print(optimizer.state_dict())

new_state_dict = {}

for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key]

for key in old_state_dict.keys():
    #print(key)
    print(torch.equal(old_state_dict[key],new_state_dict[key]))


#optimizer.step()
#for param in model.parameters():
#    print(param.grad)
#for name, param in model.named_parameters():
#    if name == 'fc1':
#        model.state_dict()[name].copy_(new_param)

#print(optimizer.state_dict())
#print(model.state_dict())
'''

'''
input = torch.randn((16,14,14))
MPlayer = nn.AdaptiveMaxPool2d(output_size=(1,1))

Avlayer = nn.AdaptiveAvgPool1d(output_size=(4))
out = MPlayer(input)
out = out.squeeze(2)
print(out.shape)
out = Avlayer(out)
print(out.shape)'''

'''
from graphviz import Digraph
import torch
from torch.autograd import Variable
def print_autograd_graph():


    def make_dot(var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph

        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function

        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            # assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    # name = param_map[id(u)] if params is not None else ''
                    # node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')

                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        return dot

    from torchvision import models

    torch.manual_seed(1)
    inputs = torch.randn(1, 3, 224, 224)
    model = models.resnet18(pretrained=False)
    y = model(Variable(inputs))
    # print(y)

    g = make_dot(y, params=model.state_dict())
    g.view()
    # g


print_autograd_graph()
'''
'''
#import numpy as np
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 3, 24, 24)
        self.target = torch.cat((
            torch.zeros(940, dtype=torch.long),
            torch.ones(60, dtype=torch.long)
        ))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)
#t0 = time.time()
t0 = time.perf_counter()
dataset = MyDataset()
targets = dataset.target.numpy()

train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets)
print(len(train_indices), len(test_indices))
# for 750 250

# for np.unique  return unique value[value1, value2 ...] --> [counts1, counts2 ...]
_, train_counts = np.unique(targets[train_indices], return_counts=True)
_, test_counts = np.unique(targets[test_indices], return_counts=True)
print(train_counts)
print(test_counts)
# TODO 训练集和测试集中 正样本和负样本所占的比例相同
t1 = time.perf_counter()
print(t1)
t2 = time.perf_counter()
print(t2)
#print(time.time()-t0)
# 0.01802515983581543
# 0.017665996136105568
'''
'''
data = torch.randint(0,10,(5,5))
index = torch.tensor([2,3])
#idx = torch.ones(5).byte()
'''
'''
idx[0] = 0
print(idx.nonzero())
print(idx.nonzero()[:,None])


print(data)
# index with same level
print(data[idx.nonzero(), idx.nonzero()])
print(data[idx.nonzero()[:,None], idx.nonzero()])
'''
'''
print(data)
#print(idx)
print(data[[[2],[3]],[2,3]],)
#print(data[index[:,None]])
#print(data[index[:,None],2])
'''




'''
new_grad = {}
for name , params in model.named_parameters():
    new_grad[name] = params.grad

print(new_grad)
'''

'''

model = Net()
ckpt = model.state_dict()
torch.save(ckpt, 'model.pth')

class New_Net(nn.Module):
    def __init__(self):
        super(New_Net, self).__init__()

        self.conv11 = nn.Conv2d(3,3,3, padding=1)
        self.conv22 = nn.Conv2d(3,3,3, padding=1)
        self.conv33 = nn.Conv2d(3,3,3, padding=1)
        self.ll = nn.Linear(9, 3)

    def forward(self, input):

        out = self.conv11(input)
        out = self.conv22(out)
        out = self.conv33(out)
        out = self.ll(out)

        return out

new_model = New_Net()
cckpt = torch.load('model.pth')
new_model.load_state_dict(ckpt, strict=True)
'''
'''
about model.load_state_dict(strict=True) key match key in new_model
                            strict=False allow key mismatch key in new_model

'''


'''
new_model = torch.load('model_1.pth')

print(new_model)

input = torch.randn(1,3,3,3)
out = new_model(input)


class New_network(nn.Module):
    def __init__(self):
        super(New_network, self).__init__()

        self.pre_trained_model = new_model
        self.last_linear = nn.Linear(27, 2)

    def forward(self, input):

        out = self.pre_trained_model(input)
        out = out.view(1,-1)
        out = self.last_linear(out)

        return out

model = New_network()
out = model(input)
'''
'''
class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3, padding=1, bias=False)
    def forward(self, x):
        conv_filter = nn.Parameter(torch.randn(3, 10, 3, 3))
        return F.conv2d(self.conv1(x), conv_filter, padding=1)


model = MyNet2()

groups1 = model.parameters()
#groups2 = model.conv_filter
opt = torch.optim.SGD(groups1, lr=0.001)
print(opt.state_dict())

count = 0
for params in model.parameters():
    count += 1

print(count)
'''
'''
class MyNet4(nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1,bias=False)
        self.relu1 = nn.ReLU(True)
     #  move this to forward
     #  self.conv2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, input):
        # I change its type to Parameter
        self.conv2_filters = nn.Parameter(torch.randn(1, 64, 3, 3))
        return F.conv2d(self.relu1(self.conv1), self.conv2_filters, padding=1)


model = MyNet4()
params = model.parameters()

for params in model.parameters():
    print(params)


#params_group = {'params': model.conv2_filters}

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(optimizer.state_dict())

'''

'''
# intermediate variables GPU consumption.
torch.manual_seed(1)
l1 = nn.Conv2d(3, 3, 1, bias=False)

intermediate = torch.randn(1, 3, 3, 3)
c_intermediate = intermediate

intermediate_out = l1(intermediate)


print(id(intermediate))
print(id(c_intermediate))
print(intermediate is c_intermediate)
'''
'''
# different from .data and just the weight
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1)
        self.conv2 = nn.Conv2d(3,3,3, padding=1)
        self.conv3 = nn.Conv2d(3,3,3, padding=1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

model = Net()

def weight_init(module):
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight.data)
'''
'''
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1)
        self.conv2 = nn.Conv2d(3,3,3, padding=1)
        self.conv3 = nn.Conv2d(3,3,3, padding=1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

model = Net()
random_input = torch.randn(1,3,3,3)
random_target = torch.randn(3,3,3)

criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001)


for i in range(5):
    opt.zero_grad()
    output = model(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    opt.step()
'''

'''
import os

config = MyConfiguration()
ckpt_path = os.path.join(config.root_dir, 'save', 'ENNet_nonbtdw1d_trans_mini_version', '0315_202825', 'checkpoint-best.pth')
ckpt = torch.load(ckpt_path)

arch = ckpt['arch']

print(arch)
'''

'''
Restart optimizer and scheduler with different learning rate
'''


'''
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset as dset


torch.manual_seed(1)

batch_size = 5
inputs = torch.randn(15,2)
# print(inputs)
target = torch.floor(4*torch.rand(15))
print('target:', target)
trainData = dset(inputs, target)

count_labels = [sum(target==i) for i in range(4)]
print('count_labels:', count_labels)

num_sample = len(inputs)
weight = 1.0/torch.Tensor(count_labels).clone().detach()
#weight = 1.0/ torch.Tensor(count_labels)
print('weight:', weight)
# weight 应该传入一个list
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, num_sample)
trainLoader = torch.utils.data.DataLoader(trainData, batch_size , shuffle=False, sampler=sampler)

print("load data")
for epoch in range(100):
    for i, (inp, tar) in enumerate(trainLoader):
        print("Epoch:{} step:{} target:{}".format(
            epoch, i, tar
        ))

#torch.optim.Adam()
'''



'''
def update_a():
    global a
    loss = mse(y_hat, y)
    print(loss)
    loss.backward(a)
    print(a.grad)
    a.grad.zero_()
    a = a - lr * a.grad

if __name__ == '__main__':

    n = 70  # num of points
    # x is a tensor
    x = torch.linspace(0, 10, steps=n)
    k = torch.tensor(2.5)
    # y is a tensor
    y = k * x + 5 * torch.rand(n)

    # loss function
    def mse(y_hat, y):
        return ((y_hat - y) ** 2).mean()

    a = torch.tensor(-1., requires_grad=True)
    a = nn.Parameter(a)


    lr = 0.005

    for t in range(10):
        print(t, ':', a.grad, a.requires_grad)
        if a.grad is not None:
            a.grad.zero_()
        y_hat = a * x
        loss = mse(y_hat, y)
        loss.backward()
        print(t, 'after backward:', a.grad, a.requires_grad, a.is_leaf)
        a = (a.data - lr * a.grad).requires_grad_(True)
        print(t, 'after update:', a.requires_grad, a.is_leaf)



#torch.optim.SGD.step()

    plt.scatter(x, y)
    plt.scatter(x, y_hat.detach())
    plt.show()
'''

'''
tensor1 = torch.randn(5, 256, 10)
tensor2 = torch.randn(5, 256, 10)
tensor3 = torch.randn(5, 256, 10)

print(len(tensor1))

concat_slices=[]
for i in range(len(tensor1)):
    sub_slice = torch.cat([tensor1[i].view(1, 256, 10), tensor2[i].view(1, 256, 10), tensor3[i].view(1, 256,10)], dim=0)
    print(sub_slice.shape)
    concat_slices.append(sub_slice)

tensor = torch.cat([sub_slice for sub_slice in concat_slices], dim=0)

print(tensor.shape)
'''

'''
class custom_dataset1(torch.utils.data.Dataset):
    def __init__(self):
        super(custom_dataset1, self).__init__()

        self.tensor_data = torch.tensor([1., 2., 3., 4., 5.])

    def __getitem__(self, index):

        return self.tensor_data[index], index

    def __len__(self):

        return len(self.tensor_data)

class custom_dataset2(torch.utils.data.Dataset):
    def __init__(self):
        super(custom_dataset2, self).__init__()

        self.tensor_data = torch.tensor([6., 7., 8., 9., 10.])

    def __getitem__(self, index):

        return self.tensor_data[index], index

    def __len__(self):

        return len(self.tensor_data)

dataset1 = custom_dataset1()
dataset2 = custom_dataset2()

concate_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
value ,index = next(iter(concate_dataset))
print(value, index)
'''

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3,3,3, padding=1, bias=False)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

def weight_init(module):
    with torch.no_grad():
        if isinstance(module, nn.Conv2d):
            torch.nn.init.ones_(module.weight)


model = Net()
random_input = torch.randn(1,3,3,3)
random_target = torch.randn(3,3,3)

model.apply(weight_init)
#print(model.conv1.weight)
c = nn.Parameter(torch.randn(1), requires_grad=True)
#c = torch.randn(1, requires_grad=True)
def hooks(module, input):

    module.weight = nn.Parameter(module.weight * c)
    print(module.weight.requires_grad, module.weight.is_leaf)


for module in model.children():
    if isinstance(module, nn.Conv2d):
        module.register_forward_pre_hook(hooks)


opt = torch.optim.SGD(model.parameters(), lr=0.001)
opt.add_param_group({'params': c})
criterion = nn.MSELoss()

# leaf node grad_fn 都是 None
for step in range(5):
    opt.zero_grad()
    logits = model(random_input)
    loss = criterion(logits, random_target)
    loss.backward()
    opt.step()
    #print(model.conv1.weight.grad_fn)
    print(c, c.grad_fn, c.requires_grad, c.is_leaf, c.grad)
    #print(model.conv1.weight)
'''


#print(linear.weight)


'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3,3,3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(3,3,3, padding=1, bias=False)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

model = Net()
print(model.conv2.weight[0])
with torch.no_grad():
    model.conv2.weight[0].requires_grad_(False)

print(model.conv2.weight.requires_grad)
random_input = torch.randn(1,3,3,3)
random_target = torch.randn(3,3,3)
criterion = nn.MSELoss()

for i in range(5):
    print(model.conv2.weight.requires_grad)
    output = model(random_input)
    loss = criterion(output, random_target)
    loss.backward()
'''
'''
a = torch.tensor(1.5)
a.requires_grad_()
b = torch.round(a)
b.backward()
print(a.grad)
'''
'''
linear = nn.Linear(10, 20)
x = torch.randn(1, 10)
L = linear(x).sum() ** 2
grad = torch.autograd.grad(L, linear.parameters(), create_graph=True) # grad for linear.weight and linear.bias
#print(grad[0].requires_grad)
#print(grad[1].requires_grad)
grad[0].requires_grad_(True)
grad[1].requires_grad_(True)
z = 0
for g in grad:
    z = z + g.pow(2).sum()
#z = grad @ grad
z.backward()
print(linear.weight.grad) # do not have gradient flow back
'''
'''
class g_phi_network(nn.Module):
    def __init__(self):
        super(g_phi_network, self).__init__()

        self.conv1 = nn.Conv2d(3,3,1, bias=False)
        self.conv2 = nn.Conv2d(3,3,1, bias=False)

    def forward(self, x):

        out = self.conv2(self.conv1(x))
        return out


class f_theta_network(nn.Module):
    def __init__(self):
        super(f_theta_network, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out

x = torch.randn((1,3,3,3), requires_grad=True)
y = torch.randn((1,3,3,3), requires_grad=True)

g_phi = g_phi_network()
f_theta = f_theta_network()
g_y = g_phi(y)
grad_y = torch.autograd.grad(g_y.sum(), y, create_graph=True)

f_grad_y = 0
for grad in grad_y:
    f_grad_y = f_grad_y + f_theta(grad)
f_x = f_theta(x)


loss = f_grad_y - f_x - torch.sum(y*[grad for grad in grad_y], dim=1)

loss.sum().backward()
'''
'''
target = torch.ones([2, 5, 5], dtype=torch.float32)
print(target)
output = torch.randn((2, 5, 5), requires_grad=True)
positive_weights = torch.FloatTensor([2, 2])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights)
loss = criterion(output, target)
loss.backward()
'''

id = torch.randint(low=0, high=1702, size=(1,))
print(id)



