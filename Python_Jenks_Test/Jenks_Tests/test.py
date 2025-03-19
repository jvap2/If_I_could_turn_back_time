import torch
from custom_optimizer import JenksSGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1
from networks import Vanilla_Test

input_data = torch.tensor([.5, .5, .5, .5])
targe_idx = torch.randint(0, 10, (1,))
print(targe_idx)
target = torch.zeros(10)  # Use the class index directly
target[targe_idx] = 1
model = Vanilla_Test(input_dim=4, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = JenksSGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
output = model(input_data)
print(output)
print(target)
loss = criterion(output, target)
loss.backward()
## print out the weights and gradients in matrix form
for name, param in model.named_parameters():
    print(name, param.data)
    print(name, param.grad)

for param_group in optimizer.param_groups:
    for param in param_group['params']:
        state = optimizer.state[param]
        print(f"State for {param}: {state}")
        if 'velocity' in state:
            print(f"Velocity for {param}: {state['velocity']}")

optimizer.step()
print("After the update")
for name, param in model.named_parameters():
    print(name, param.data)
    print(name, param.grad)

