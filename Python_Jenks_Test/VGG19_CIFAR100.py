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



torch.cuda.empty_cache()
train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} device")


train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())

imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

mean = imgs.view(1, -1).mean(dim=1)    
std = imgs.view(1, -1).std(dim=1)     

mnist_transforms_train = transforms.Compose([transforms.ToTensor(),
                                       transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                       transforms.Normalize(mean=mean, std=std)])

mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])



train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

train_dataset.dataset.transform = mnist_transforms_train
val_dataset.dataset.transform = mnist_transforms
BATCH_SIZE = 256

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)