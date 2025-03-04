import torch
from custom_optimizer import JenksSGD,PruneWeights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet,lenet5v1
from torch.autograd.functional import hessian
from functions import hutchinson_trace_hmp,rademacher
from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian
from functions import exact_trace

# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'




train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./datasets", train=False, download=False, transform=transforms.ToTensor())

imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

mean = imgs.view(1, -1).mean(dim=1)    
std = imgs.view(1, -1).std(dim=1)     

mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)])

train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=mnist_transforms)
test_dataset = datasets.MNIST(root="./datasets/", train=False, download=False, transform=mnist_transforms)

train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model_lenet5v1 = LeNet5V1()
model = nn.Sequential(            
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
    
    #2
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),  

    nn.Flatten(),
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Tanh(),
    nn.Linear(in_features=120, out_features=84),
    nn.Tanh(),
    nn.Linear(in_features=84, out_features=10),
).to(device)

model = extend(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn = extend(loss_fn)
momentum = 0.99
optimizer = JenksSGD(params=model.parameters(), lr=5e-3, scale=0.5e-4, momentum=momentum)
accuracy = Accuracy(task='multiclass', num_classes=10)


# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d")
experiment_name = "MNIST"
model_name = "LeNet5V1"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

# device-agnostic setup
print(f"Using {device} device")
accuracy = accuracy.to(device)
os.makedirs("models", exist_ok=True)
EPOCHS = 2
train_loss, train_acc = 0.0, 0.0
count = 0
original_magnitude = sum(torch.norm(p) for p in model.parameters())

for epoch in range(EPOCHS):
    # Training loop
    print("Epoch: ", epoch)
    for X, y in train_dataloader:
        count += 1
        X, y = X.to(device), y.to(device)
        
        model.train()
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        mag = 0
        mag = sum(torch.norm(p) for p in model.parameters())
        acc = accuracy(y_pred, y)
        train_acc += acc
        train_filename = f"output/training_log_{timestamp}_{momentum}.txt"
        with open(train_filename,"a") as f:
            print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Sparsity: {mag/original_magnitude: .5f}", file=f)
        optimizer.zero_grad()
        with backpack(DiagHessian(), HMP()):
        # keep graph for autodiff HVPs
            loss.backward()
        optimizer.step()
        params = torch.cat([p.data.flatten() for p in model.parameters()])

        trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
        # trace = exact_trace(model_lenet5v1)
        # Calculate the trace
        trace_filename = f"output/trace_log_{timestamp}_{momentum}.txt"
        with open(trace_filename,"a") as f:
            print(f"Iteration: {count}| Trace: {trace: .5f}", file=f)
        
    # train_loss /= len(train_dataloader)
    # train_acc /= len(train_dataloader)
    # Validation loop
val_loss, val_acc = 0.0, 0.0
count_val = 0
prunedmodel = PruneWeights(model)
'''Make sure the weights are back on the device'''
model = prunedmodel.to(device)
# model.eval()
trace_val_filename = f"output/trace_val_log_{timestamp}_{momentum}.txt"
non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
total_params = sum(p.numel() for p in model.parameters())
sparsity = 1 - non_zero_params / total_params
sparsity_filename = f"output/sparisty_log_{timestamp}_{momentum}.txt"  
with open(sparsity_filename,"a") as f:
    print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
with torch.inference_mode():
    for X, y in val_dataloader:
        count_val += 1
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        val_loss += loss.item()
        optimizer.zero_grad()
        with backpack(DiagHessian(), HMP()):
        # keep graph for autodiff HVPs
            loss.backward()
        trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
        with open(trace_val_filename,"a") as f:
            print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
        acc = accuracy(y_pred, y)
        val_acc += acc
        val_filename = f"output/validation_log_{timestamp}_{momentum}.txt"
        with open(val_filename,"a") as f:
            print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f}", file=f)
        
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    
writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
with open("output/output.txt","a") as f:
    print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc/count: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)
## Save model
torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")