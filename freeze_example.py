import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

net = Net()

print('fc2 weight before train:')
print(net.fc2.weight)
#print('fc2 grad before train:')
#print(net.fc2.weight.grad)
# here is None

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    #print(net.fc2.weight.grad)
    optimizer.step()

print('fc2 weight after train:')
print(net.fc2.weight)

torch.save(net.state_dict(), 'model.pth')

del net
net = Net()
net.load_state_dict(torch.load('model.pth'))
print('fc2 pretrained weight (same as the one above):')
print(net.fc2.weight)

random_input = torch.randn(10, requires_grad=True)
random_target = torch.randn(1, requires_grad=True)

net.fc2.weight.requires_grad=False
net.fc2.bias.requires_grad=False

criterion= nn.MSELoss()
# pytorch optimizer explicity accepts requires_grad parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()

print('fc2 weight (frozen) after retrain:')
print(net.fc2.weight)

# then unfreeze fc2 layers
net.fc2.weight.requires_grad=True
net.fc2.bias.requires_grad=True

optimizer.add_param_group({'params': net.fc2.parameters()})

# retrain
for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()


print('fc2 weight (unfrozen) after re-retrain:')
print(net.fc2.weight)




