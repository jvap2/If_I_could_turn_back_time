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


timestamp = datetime.now().strftime("%Y-%m-%d")
experiment_name = "MNIST"
model_name = "LeNet5V1"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

transform = transforms.Compose(
    [transforms.ToTensor(),
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
loss_fn = nn.CrossEntropyLoss()
optimizer = JenksSGD(params=net.parameters(), lr=5e-3, scale=0.5e-4, momentum=0.99)
accuracy = Accuracy(task='multiclass', num_classes=10)
accuracy = accuracy.to(device)
net = net.to(device)
EPOCHS = 4

for epoch in range(EPOCHS):
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    print("Epoch: ", epoch)
    count = 0
    for X, y in trainloader:
        count += 1
        X, y = X.to(device), y.to(device)
        
        net.train()
        
        y_pred = net(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        acc = accuracy(y_pred, y)
        train_acc += acc

        if count % 100 == 0:
            print(f"Batch: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(trainloader)
    train_acc /= len(trainloader)
        
    # Validation loop
    val_loss, val_acc = 0.0, 0.0
    net.eval()
    with torch.inference_mode():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = net(X)
            
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            val_acc += acc
            
        val_loss /= len(testloader)
        val_acc /= len(testloader)
        
    writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
    with open("output.txt","a") as f:
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)
    ## Save model
    torch.save(net.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")