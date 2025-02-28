import torch
from custom_optimizer import JenksSGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet

# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)

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
model_lenet5v1 = LeNet5V1()
loss_fn = nn.CrossEntropyLoss()
optimizer = JenksSGD(params=model_lenet5v1.parameters(), lr=5e-3, scale=0.5e-4, momentum=0.99)
accuracy = Accuracy(task='multiclass', num_classes=10)


# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d")
experiment_name = "MNIST"
model_name = "LeNet5V1"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

# device-agnostic setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
accuracy = accuracy.to(device)
model_lenet5v1 = model_lenet5v1.to(device)
os.makedirs("models", exist_ok=True)
EPOCHS = 4

for epoch in range(EPOCHS):
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    print("Epoch: ", epoch)
    count = 0
    for X, y in train_dataloader:
        count += 1
        X, y = X.to(device), y.to(device)
        
        model_lenet5v1.train()
        
        y_pred = model_lenet5v1(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        acc = accuracy(y_pred, y)
        train_acc += acc

        if count % 100 == 0:
            print(f"Batch: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
        
    # Validation loop
    val_loss, val_acc = 0.0, 0.0
    model_lenet5v1.eval()
    with torch.inference_mode():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model_lenet5v1(X)
            
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            val_acc += acc
            
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        
    writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
    with open("output.txt","a") as f:
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)
    ## Save model
    torch.save(model_lenet5v1.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")