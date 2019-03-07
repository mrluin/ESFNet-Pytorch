import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(5, 5, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(5, 3, 3, padding=1, bias=False)

    def forward(self, *input):

        out = F.relu(self.conv1(*input))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        return out

if __name__ == '__main__':

    random_input = torch.randn(1,3,10,10)
    random_output = torch.randn(3, 10, 10)

    model = Net()
    criterion = nn.MSELoss()
    #opt = optim.Adam(model.parameters(), lr=0.1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    print(opt.state)
    # lr_lambda可以是list 每一个对应param_groups中的一个group
    # lr_lambda是一个关于epoch的函数, new_lr = currently_lr * lr_lambda
    lr_lambda = lambda epoch: epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    #print(lr_scheduler.state_dict())
    """
        for {
            'last_epoch': -1
            'base_lrs': [0.1]
            'lr_lambdas': [None] from source is [None] * len(lr_lambda). just [None]
        }
    """
    for epoch in range(20):

        lr_scheduler.step(epoch)
        #print(lr_scheduler.state_dict())
        #print(lr_scheduler.get_lr())
        #print(opt.state_dict())
        opt.zero_grad()
        output = model(random_input)
        loss = criterion(output, random_output)
        loss.backward()
        opt.step()
        print(opt.state)
    start_epoch = lr_scheduler.last_epoch +1
    state = {
        'epoch': start_epoch,
        'opt': opt.state_dict(),
        'state_dict': model.state_dict(),
    }
    torch.save(state, 'model.pth')
    """
        不需要 lr_scheduler.state_dict()当已经有一个lr_scheduler 继续训练的时候
        当 from_scratch的时候需要进行load
    """

    # resume
    ckpt = torch.load('model.pth')
    model.load_state_dict(ckpt['state_dict'])
    opt.load_state_dict(ckpt['opt'])

    for epoch in range(ckpt['epoch'], ckpt['epoch']+20):

        lr_scheduler.step(epoch)
        #print(lr_scheduler.get_lr())
        opt.zero_grad()
        output = model(random_input)
        loss = criterion(output, random_output)
        loss.backward()
        opt.step()








