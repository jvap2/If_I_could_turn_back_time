import torch
from custom_optimizer import JenksSGD
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet
from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian

from functions import hutchinson_trace_hmp,rademacher
from custom_optimizer import PruneWeights

torch.cuda.empty_cache()
# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1),

                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2),

                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),

                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),

                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2),

                nn.Flatten(),

                nn.Linear(9216, 4096),
                nn.ReLU(),

                nn.Linear(4096, 4096),
                nn.ReLU(),

                nn.Linear(4096, 10)
            ).to(device)

timestamp = datetime.now().strftime("%Y-%m-%d")
experiment_name = "MNIST"
model_name = "Lemodel5V1"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

transform = transforms.Compose(
    [transforms.Resize((227, 227)),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
os.makedirs("AlexNet_CIFAR10_output", exist_ok=True)
model = extend(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn = extend(loss_fn)
momentum = 0.99
loss_fn = nn.CrossEntropyLoss()
optimizer = JenksSGD(params=model.parameters(), lr=5e-3, scale=0.5e-4, momentum=0.99)
accuracy = Accuracy(task='multiclass', num_classes=10)
accuracy = accuracy.to(device)
EPOCHS = 2
train_loss, train_acc = 0.0, 0.0
count = 0
original_magnitude = sum(torch.norm(p)**2 for p in model.parameters())
lambda_ = 0.01

for epoch in range(EPOCHS):
    # Training loop
    print("Epoch: ", epoch)
    for X, y in trainloader:
        count += 1
        X, y = X.to(device), y.to(device)
        
        model.train()
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
        loss = loss.clone() + lambda_ * l2_reg  
        train_loss += loss.item()

        acc = accuracy(y_pred, y)
        train_acc += acc
        train_filename = f"AlexNet_CIFAR10_output/training_log_{timestamp}_{momentum}.txt"
        with open(train_filename,"a") as f:
            print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | L_2: {l2_reg/original_magnitude: .5f}", file=f)
        optimizer.zero_grad()
        with backpack(DiagHessian(), HMP()):
            loss.backward()
        optimizer.step()
        params = torch.cat([p.data.flatten() for p in model.parameters()])

        trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
        # trace = exact_trace(model_lenet5v1)
        # Calculate the trace
        trace_filename = f"AlexNet_CIFAR10_output/trace_log_{timestamp}_{momentum}.txt"
        with open(trace_filename,"a") as f:
            print(f"Iteration: {count}| Trace: {trace: .5f}", file=f)
        
    # train_loss /= len(train_dataloader)
    # train_acc /= len(train_dataloader)
    # Validation loop
val_loss, val_acc = 0.0, 0.0
count_val = 0
prunedmodel = PruneWeights(model)
'''Make sure the weights are back on the device'''
with open("output/output.txt","a") as f:
    print("Able to prune the weights", file=f)
model = prunedmodel.to(device)
# model.eval()
trace_val_filename = f"AlexNet_CIFAR10_output/trace_val_log_{timestamp}_{momentum}.txt"
non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
total_params = sum(p.numel() for p in model.parameters())
sparsity = 1 - non_zero_params / total_params
sparsity_filename = f"AlexNet_CIFAR10_output/sparisty_log_{timestamp}_{momentum}.txt"  
model.eval()
with open(sparsity_filename,"a") as f:
    print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
with torch.inference_mode():
    for X, y in testloader:
        count_val += 1
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        val_loss += loss.item()
        # optimizer.zero_grad()
        # with backpack(DiagHessian(), HMP()):
        # # keep graph for autodiff HVPs
        #     loss.backward()
        # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
        # with open(trace_val_filename,"a") as f:
        #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
        acc = accuracy(y_pred, y)
        val_acc += acc
        val_filename = f"output/validation_log_{timestamp}_{momentum}.txt"
        with open(val_filename,"a") as f:
            print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f}", file=f)
        
    val_loss /= len(testloader)
    val_acc /= len(testloader)
    
writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
with open("output/output.txt","a") as f:
    print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc/count: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)
## Save model
torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")