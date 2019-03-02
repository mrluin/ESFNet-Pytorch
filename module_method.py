import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    # defines the computation performed at every call
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

    # extra representation of the module reimplement
    def extra_repr(self):
        return "for test model functions"

submodel = nn.Sequential(
    nn.Conv2d(20, 20, 5),
    nn.ReLU()
)
# model.add_module add submodule to current model
model = Model()
model.add_module('conv3', submodel)
#print(model)
'''
# init model weights by model.apply(function)
def init_weigths(m):
    print(m)
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(1.0)
        print(m.weight)

model.apply(init_weigths)

# model.buffers(recurse=True) current module and all submodules
for buf in model.buffers():
    print(type(buf.data), buf.size())

# return immediate children module
for children in model.children():
    print(children)
    
# casts all floating point parameters and buffers to double data type
# data type: double -> float -> half
model.double() 
model.float()
model.half()
# cast all parameters and buffers to dst_type
model.type(dst_type)


# load model
model.load_state_dict(state_dict, strict=True)
 
# 以递归的形式 输出全部module 重复出现的module只输出一次
# e.g. output: module -> submodule1, submodule2 ...
for module in model.modules():
    print(module)  
    
# named_buffer/parameters/children 用于进行输出和筛选
for name, buf in model.named_buffers():
    if name in ['running_var']:
        print(buf.size())

for name, module in model.named_children():
    print(name, module)
    
# conv1.weight parameter containing: ... .../ conv1.bias parameter containing: ... ...
for name, param in model.named_parameters():
    print(name, param)    
    
# display params of model without parameter_name
for param in model.parameters():
    print(param.data)
# named_parameters
for name, param in model.named_parameters():
    print(name, param.requires_grad)
'''
'''
# adds a persistent buffer to module
# This is typically used to register a buffer that should not to be considered
# a module parameters
model.register_buffer('running_mean', tensor=torch.zeors(100))
# add parameters to module
model.register_parameter(name, param)
'''


'''
#### ==== IMPORTANT ====####
# the hook will be called every time after forward()
model.register_forward_hook()
# The hook will be called every time the gradients with respect to module inputs are computed
model.register_backward_hook()
# the hook will be called every time before forward()
'''
'''
# set gradients of all model parameters to zero
model.zero_grad()

'''
'''
    #tensor.unfold(dim, size, step)
    size is slice.size
    step
x = torch.arange(1,17).float().view(1,1,4,4)
print(x)
kh, kw = 2, 2
dh, dw = 2, 2

input_windows = x.unfold(2, kh, dh).unfold(3, kw, dw)
output = input_windows.contiguous().view(x.size())
print(output)
'''





