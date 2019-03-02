import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_adam = False


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.enc = nn.Linear(64, 10)
        self.dec1 = nn.Linear(10, 64)
        self.dec2 = nn.Linear(10, 64)

    def forward(self, x, decoder_idx):
        x = F.relu(self.enc(x))
        if decoder_idx == 1:
            print('Using dec1')
            x = self.dec1(x)
        elif decoder_idx == 2:
            print('Using dec2')
            x = self.dec2(x)
        else:
            print('Unknown decoder_idx')

        return x


x = torch.randn(1, 64)
y = x.clone()
model = MyModel()
criterion = nn.MSELoss()

if use_adam:
    # for all parameters
    optimizer = optim.Adam(model.parameters(), lr=1.)
else:
    optimizer = optim.SGD(model.parameters(), lr=1.)

# Save init values
old_state_dict = {}
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

# Training procedure for DEC1
optimizer.zero_grad()
output = model(x, 1)
loss = criterion(output, y)
loss.backward()

# Check for gradients in dec1, dec2
print('Dec1 grad: {}\nDec2 grad: {}'.format(
    model.dec1.weight.grad, model.dec2.weight.grad))

optimizer.step()

# Save new params
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

# Compare params
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print('Diff in {}'.format(key))

# Update
old_state_dict = {}
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

# Pass through dec2
optimizer.zero_grad()
output = model(x, 2)
loss = criterion(output, y)
loss.backward()

print('Dec1 grad: {}\nDec2 grad: {}'.format(
    model.dec1.weight.grad, model.dec2.weight.grad))

optimizer.step()

# Save new params
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

# Compare params
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print('Diff in {}'.format(key))

## Create separate optimizers for adam share the same encoder
model = MyModel()
dec1_params = list(model.enc.parameters()) + list(model.dec1.parameters())
optimizer1 = optim.Adam(dec1_params, lr=1.)
dec2_params = list(model.enc.parameters()) + list(model.dec2.parameters())
optimizer2 = optim.Adam(dec2_params, lr=1.)

# Save init values
old_state_dict = {}
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

# Training procedure
optimizer1.zero_grad()
output = model(x, 1)
loss = criterion(output, y)
loss.backward()

# Check for gradients in dec1, dec2
print('Dec1 grad: {}\nDec2 grad: {}'.format(
    model.dec1.weight.grad, model.dec2.weight.grad))

optimizer1.step()

# Save new params
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

# Compare params
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print('Diff in {}'.format(key))

# Update
old_state_dict = {}
for key in model.state_dict():
    old_state_dict[key] = model.state_dict()[key].clone()

# Pass through dec2
optimizer1.zero_grad()
output = model(x, 2)
loss = criterion(output, y)
loss.backward()

print('Dec1 grad: {}\nDec2 grad: {}'.format(
    model.dec1.weight.grad, model.dec2.weight.grad))

optimizer2.step()

# Save new params
new_state_dict = {}
for key in model.state_dict():
    new_state_dict[key] = model.state_dict()[key].clone()

# Compare params
for key in old_state_dict:
    if not (old_state_dict[key] == new_state_dict[key]).all():
        print('Diff in {}'.format(key))