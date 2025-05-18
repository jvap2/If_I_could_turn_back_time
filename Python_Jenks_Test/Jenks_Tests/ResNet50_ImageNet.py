from models import ResNet56
import torch
from custom_optimizer import JenksSGD,PruneWeights, JenksSGD_Noise, SAM, JenksSGD_Test, PruneWeights_Test, train_one_step, Prune_Score_Mag
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
from custom_optimizer import Prune_Score,train_one_step_prune,Prune_Score_Select
from custom_schedulers import WarmupMultiStepLR, init_lr_weight_decay,WarmupMultiStepJenks
from torchvision.models import resnet50

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

one_shot = True

torch.cuda.empty_cache()
# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

kill_velocity = False
train_lr_decay_factor = 0.25

gsm_lr_base_value = 1e-2
gsm_lr_boundaries = [200, 230, 260]
gsm_momentum = 0.99
gsm_max_epochs = 280
mask = True



train_val_dataset = datasets.ImageNet(root="./datasets/", train=True, download=True, transform=transforms.ToTensor())


imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

mean = imgs.view(1, -1).mean(dim=1)    
std = imgs.view(1, -1).std(dim=1)     

mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((256, 256)),
                                       transforms.Normalize(mean=mean, std=std)])

train_val_dataset = datasets.ImageNet(root="./datasets/", train=True, download=False, transform=mnist_transforms)


train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size


train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
fin_val_dataset, test_dataset = torch.utils.data.random_split(dataset=val_dataset, lengths=[int(0.5 * len(val_dataset)), int(0.5 * len(val_dataset))])

train_dataset.dataset.transform = mnist_transforms
fin_val_dataset.dataset.transform = mnist_transforms
test_dataset.dataset.transform = mnist_transforms

BATCH_SIZE = 16

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=fin_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model_lenet5v1 = LeNet5V1()
model = resnet50(weights=None,progress=True).to(device)
model = extend(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn = extend(loss_fn)
momentum = 0.9
learning_rate = .1e-2
weight_decay = 1e-3
warmup_epochs = 20
nestrov = True
params = []
bias_lr = False
optimizer = init_lr_weight_decay(model, learning_rate, weight_decay, momentum=momentum, nestrov=nestrov, bias_lr=bias_lr)
# scheduler = WarmupMultiStepLR(optimizer, milestones=[80, 120, 140], warmup_factor=0.1, warmup_iters=10, warmup_method="linear")
scheduler = WarmupMultiStepJenks(optimizer, milestones=gsm_lr_boundaries, warmup_factor=0.1, warmup_iters=warmup_epochs, warmup_method="linear")
accuracy = Accuracy(task='multiclass', num_classes=1000)
top5accuracy = MulticlassAccuracy(num_classes=1000, top_k=5)


# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "ImageNet"
model_name = "ResNet50"
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
lambda_ = 0.01


train_dir = "ResNet50_ImageNet_output/"
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
master_count = 0
epoch = 0
prune_epoch = 100
no_jenks =True
prune_count = 0
for epoch in range(EPOCHS):
    # Training loop
    print("Epoch: ", epoch)
    epoch += 1
    model.train()
    #print the epoch and learning rate
    with open(train_filename,"a") as f:
        print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
    count = 0
    train_loss, train_acc = 0.0, 0.0
    train_top5acc = 0.0
    start = time()
    print(f"Memory free: {get_memory_free_MiB(0)} MiB")
    for X, y in train_dataloader:
        # print(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        count += 1
        # loss = loss.clone() + lambda_ * l2_reg
        X, y = X.to(device), y.to(device)
        master_count += 1
        acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch, mask=mask,)
        if mask and epoch>prune_epoch:
            ## Go through all the parameters and set the pruned ones to zero
            for name, param in model.named_parameters():
                param.data = param.data *optimizer.state[param]['mask']
        # acc = accuracy(y_pred, y)
        # acc_5 = top5accuracy(y_pred, y)
        train_loss += loss.item()
        train_top5acc += acc5.item()
        train_acc += acc.item()
        l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
        # print("Train loss type : ", type(train_loss))
        # print("Train Acc type : ", type(train_acc))
        # print("Train Top5Acc type : ", type(train_top5acc))
        # print("Loss type : ", type(loss))
        # print("l2_reg type : ", type(l2_reg))
        with open(train_filename, "a") as f:
            print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg/original_magnitude: .5f}", file=f)
    stop = time()
    print(f"Time taken for epoch: {stop-start}")
    # if epoch < 151:
    scheduler.step()
    with open (log_filename,"a") as f:
        print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
    if epoch >= prune_epoch and epoch % 10 == 0:
        # if kill_velocity and epoch==prune_epoch:
        #     Prune_Score(optimizer, kill_velocity=True)
        if one_shot and epoch==prune_epoch and mask:
            print("Pruning the weights")
            Prune_Score(optimizer, mask=True)
            prune_count += 1
        elif not one_shot and epoch>=prune_epoch and epoch % 20 == 0 and prune_count<10:
            print("Pruning the weights")
            Prune_Score(optimizer, mask=True)
            prune_count += 1
        # if not kill_velocity or not mask:
        #     Prune_Score(optimizer)
        '''Make sure the weights are back on the device'''
        # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
        #     print("Able to prune the weights", file=f)
        # model = prunedmodel.to(device)
        non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        sparsity = 1 - non_zero_params / total_params
        with open(sparsity_filename,"a") as f:
            print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
    if epoch == warmup_epochs:
        '''Change the learning rate to the base value'''
        for group in optimizer.param_groups:
            group['lr'] = 3e-3
        # for param_group in optimizer.param_groups:
        #     param_group['momentum'] = 0.99
        
    model.eval()
    with torch.inference_mode():
        with open(val_filename,"a") as f:
            print(f"Epoch: {epoch}", file=f)
        val_loss, val_acc = 0.0, 0.0
        val_top5acc = 0.0
        count_val = 0
        for X, y in val_dataloader:
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
            top5_acc = top5accuracy(y_pred, y)
            val_top5acc += top5_acc
            val_acc += acc
            with open(val_filename,"a") as f:
                print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

        # val_loss /= len(test_dataloader)
        # val_acc /= len(test_dataloader)

    writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
    writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
    with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)
    ## Save model
    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

val_loss, val_acc = 0.0, 0.0
val_top5acc = 0.0
count_val = 0
'''Make sure the weights are back on the device'''
non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
# non_zero_params_2 = sum(torch.count_nonzero(p) for p in model_2.parameters())
total_params = sum(p.numel() for p in model.parameters())
sparsity = 1 - non_zero_params / total_params
# sparsity_2 = 1 - non_zero_params_2 / total_params
model.eval()
with open(sparsity_filename,"a") as f:
    print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)

model.eval()
with torch.inference_mode():
    with open(test_filename,"a") as f:
        print(f"Epoch: {epoch}", file=f)
    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    for X, y in test_dataloader:
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
        top5_acc = top5accuracy(y_pred, y)
        val_top5acc += top5_acc
        val_acc += acc
        with open(test_filename,"a") as f:
            print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

    val_loss /= len(test_dataloader)
    val_acc /= len(test_dataloader)