import torch
from custom_optimizer import JenksSGD,PruneWeights, JenksSGD_Noise, SAM, JenksSGD_Test, PruneWeights_Test, train_one_step, Prune_Score_Mag, train_one_step_prune_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet,lenet5v1
from torch.autograd.functional import hessian
from functions import hutchinson_trace_hmp,rademacher
from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian
from functions import exact_trace
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from time import time
from cuda_helpers import get_memory_free_MiB
from custom_optimizer import Prune_Score,train_one_step_prune,Prune_Score_Select, InfiniteDataLoader, Prune_Score_v2, init_network, Prune_Score_v3
from custom_schedulers import WarmupMultiStepLR, init_lr_weight_decay,WarmupMultiStepJenks

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from datetime import datetime
import os
import torch.nn as nn
# from networks import LeNet5V1,alexnet,lenet5v1
from torch.autograd.functional import hessian
# from functions import hutchinson_trace_hmp,rademacher
from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian
from rcnet import create_lenet5
from rcnet import ConvBuilder
from training_loop import train_val_loop
one_shot = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
prune_ratio = .95
model  = create_lenet5().to(device)
'''Go through the model and find the name of the first and last layer'''


names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
name_first = names[0]
name_last = names[-1]
imp_names = [name_first, name_last]
print(f"First layer name: {name_first}")
print(f"Last layer name: {name_last}")
## Print the number of parameters in the model
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

# model_compare = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
#     nn.Tanh(),
#     nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
    
#     #2
#     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
#     nn.Tanh(),
#     nn.AvgPool2d(kernel_size=2, stride=2),  

#     nn.Flatten(),
#     nn.Linear(in_features=16*5*5, out_features=120),
#     nn.Tanh(),
#     nn.Linear(in_features=120, out_features=84),
#     nn.Tanh(),
#     nn.Linear(in_features=84, out_features=10),
# ).to(device)

kill_velocity = False
train_lr_decay_factor = 0.25
BATCH_SIZE = 256
gsm_lr_base_value = 1e-2
gsm_lr_boundaries = [160, 200, 240]
gsm_momentum = 0.99
gsm_max_epochs = 280
mask = True
torch.cuda.empty_cache()
train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset =  datasets.MNIST("./datasets/", train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))
val_dataset= datasets.MNIST("./datasets/", train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))




train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# model_lenet5v1 = LeNet5V1()

min_epochs = 200
model = extend(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn = extend(loss_fn)
momentum = 0.99
learning_rate = 1e-2
weight_decay = 5e-4
warmup_epochs = 10
nestrov = False
params = []
bias_lr = True
optimizer = init_lr_weight_decay(model, learning_rate, weight_decay, momentum=momentum, nestrov=nestrov, bias_lr=bias_lr)
init_network(optimizer)
# scheduler = WarmupMultiStepLR(optimizer, milestones=[80, 120, 140], warmup_factor=0.1, warmup_iters=10, warmup_method="linear")
scheduler = WarmupMultiStepJenks(optimizer, milestones=gsm_lr_boundaries, warmup_factor=0.1, warmup_iters=warmup_epochs, warmup_method="linear")
accuracy = Accuracy(task='multiclass', num_classes=10)
top5accuracy = MulticlassAccuracy(num_classes=10, top_k=5)


# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "MNIST"
model_name = "LeNet5"
log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
writer = SummaryWriter(log_dir)

# device-agnostic setup
print(f"Using {device} device")
accuracy = accuracy.to(device)
top5accuracy = top5accuracy.to(device)
os.makedirs("models", exist_ok=True)
train_loss, train_acc = 0.0, 0.0
train_top5acc = 0.0
count = 0
original_magnitude = sum(torch.norm(p)**2 for p in model.parameters())
lambda_ = 1e-4


train_dir = "LeNet5_MNIST_output_new/"
os.makedirs(train_dir, exist_ok=True)  # Create directory if it doesn't exist
name =  "SGD_Agg"
EPOCHS = 280
log_filename = os.path.join(train_dir, f"log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
train_filename = os.path.join(train_dir, f"training_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_filename = os.path.join(train_dir, f"trace_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
sparsity_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
trace_val_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
val_filename = os.path.join(train_dir,f"validation_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
test_filename = os.path.join(train_dir,f"test_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
jenks_filename = os.path.join(train_dir,f"jenks_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
debug_filename = os.path.join(train_dir,f"debug_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
prune_filename = os.path.join(train_dir,f"prune_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
master_count = 0
epoch = 0
prune_epoch = 160
prune_epoch_list = [prune_epoch, prune_epoch + 20]
no_jenks =False
l2 = True
mag_prune = True
prune_between = 10
with open(log_filename,"a") as f:
    print(f"Starting Learning rate: {learning_rate}", file=f)
    print(f"Momentum: {momentum}", file=f)
    print(f"Weight decay: {weight_decay}", file=f)
    print(f"Batch size: {BATCH_SIZE}", file=f)
    print(f"Epochs: {EPOCHS}", file=f)
    print(f"Epoch to start pruning: {prune_epoch}", file=f)
    print(f"Warmup epochs: {warmup_epochs}", file=f)
    if nestrov:
        print(f"Opt type is Nesterov", file=f)
    else:
        print(f"Opt type is Jenks SGD", file=f)
    if kill_velocity:
        print(f"Velocity is killed", file=f)
    else:
        print(f"Velocity is not killed", file=f)
    if mask:
        print(f"Mask is used", file=f)
    else:
        print(f"Mask is not used", file=f)
    if bias_lr:
        print(f"Bias LR is used", file=f)
    else:
        print(f"Bias LR is not used", file=f)
    if no_jenks:
        print(f"No Jenks is used", file=f)
    else:
        print(f"Jenks is used", file=f)
    if mag_prune:
        print(f"Mag prune is used", file=f)
    else:
        print(f"Mag prune is not used", file=f)
prune_count = 0
sparsity = 0.0
one_update = True
bias_prune = False

train_val_loop(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device,
               experiment_name, model_name, timestamp,
               train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
               prune_count=prune_count, one_update=one_update, EPOCHS=EPOCHS, sparsity=sparsity,
               prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch, prune_between=prune_between,
               prune_ratio=prune_ratio, one_shot=one_shot, mask=mask, mag_prune=mag_prune,
               bias_prune=bias_prune, kill_velocity=kill_velocity, l2=l2, lambda_=lambda_, warmup_epochs=warmup_epochs,
               min_epochs=min_epochs)