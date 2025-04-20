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
from custom_optimizer import Prune_Score,train_one_step_prune
from custom_schedulers import WarmupMultiStepLR

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
# from functions import exact_trace

kill_velocity = False
train_lr_decay_factor = 0.1

gsm_lr_base_value = 1e-2
gsm_lr_boundaries = [160, 200, 240]
gsm_momentum = 0.99
gsm_max_epochs = 280
mask = True
torch.cuda.empty_cache()
train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} device")


train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())

imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)

mean = imgs.view(1, -1).mean(dim=1)
std = imgs.view(1, -1).std(dim=1)

# mnist_transforms_train = transforms.Compose([transforms.ToTensor(),
#                                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
#                                        transforms.Normalize(mean=mean, std=std)])

mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])



train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size


train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])
fin_val_dataset, test_dataset = torch.utils.data.random_split(dataset=val_dataset, lengths=[int(0.5 * len(val_dataset)), int(0.5 * len(val_dataset))])

train_dataset.dataset.transform = mnist_transforms
fin_val_dataset.dataset.transform = mnist_transforms
test_dataset.dataset.transform = mnist_transforms
BATCH_SIZE = 256

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# model_lenet5v1 = LeNet5V1()



model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=300),
    nn.ReLU(),
    nn.Linear(in_features=300, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
).to(device)

model_2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=784, out_features=300),
    nn.ReLU(),
    nn.Linear(in_features=300, out_features=100),
    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
).to(device)

model = extend(model)
loss_fn = nn.CrossEntropyLoss()
loss_fn = extend(loss_fn)
momentum = 0.9
learning_rate = .5e-2
weight_decay = 1e-3
warmup_epochs = 20
nestrov = True
params = []
bias_lr = True
if bias_lr:
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = learning_rate
        weight_decay = weight_decay
        # if "bias" in key or "bn" in key or "BN" in key:
        #     # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        #     weight_decay = cfg.weight_decay_bias
        #     print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum, nesterov=nestrov)
else:
    optimizer = SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nestrov)
# optimizer = AdamW(params=model.parameters(), lr=5e-3, weight_decay=1e-3)
# optimizer = JenksSGD_Test(params=model.parameters(),warmup_epochs=warmup_epochs, lr=.02, scale=1e-3, momentum=momentum, nestrov=False, bias = True)
# optimizer = SAM(params=model.parameters(), base_optimizer=JenksSGD_Test, lr=5e-3, momentum=momentum)
# scheduler = StepLR(optimizer, step_size = 50, gamma = 0.1)
scheduler = WarmupMultiStepLR(optimizer, milestones=[80, 120, 140], warmup_factor=0.1, warmup_iters=10, warmup_method="linear")
accuracy = Accuracy(task='multiclass', num_classes=10)
top5accuracy = MulticlassAccuracy(num_classes=10, top_k=5)


# Experiment tracking
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = "MNIST"
model_name = "LeNet300V100"
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


train_dir = "LeNet300_100_MNIST_output/"
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
        acc, acc5, loss = train_one_step(model,X, y, optimizer, loss_fn, epoch, warmup_epochs)
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
        if mask and epoch==prune_epoch:
            print("Pruning the weights")
            Prune_Score(optimizer, mask=True)
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
# model_2.load_state_dict(model.state_dict())
# model_2 = model_2.to(device)
# model_2.eval()
# Prune_Score(optimizer)
# prunedmodel_2 = optimizer.PruneWeights_Test(model_2)
'''Make sure the weights are back on the device'''
# with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
#     print("Able to prune the weights", file=f)
# model = prunedmodel.to(device)
# model_2 = prunedmodel_2.to(device)
# model.eval()
# trace_val_filename = f"LeNet5_MNIST_output/trace_val_log_{timestamp}_{momentum}.txt"
non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters())
non_zero_params_2 = sum(torch.count_nonzero(p) for p in model_2.parameters())
total_params = sum(p.numel() for p in model.parameters())
sparsity = 1 - non_zero_params / total_params
sparsity_2 = 1 - non_zero_params_2 / total_params
model.eval()
with open(sparsity_filename,"a") as f:
    print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f} | Sparsity from Test: {sparsity_2: .5f}", file=f)

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


# model_2.eval()
# with torch.inference_mode():
#     with open(test_filename,"a") as f:
#         print(f"Epoch: {epoch}", file=f)
#     val_loss, val_acc = 0.0, 0.0
#     val_top5acc = 0.0
#     for X, y in test_dataloader:
#         count_val += 1
#         X, y = X.to(device), y.to(device)

#         y_pred = model_2(X)

#         loss = loss_fn(y_pred, y)
#         val_loss += loss.item()
#         # optimizer.zero_grad()
#         # with backpack(DiagHessian(), HMP()):
#         # # keep graph for autodiff HVPs
#         #     loss.backward()
#         # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
#         # with open(trace_val_filename,"a") as f:
#         #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
#         acc = accuracy(y_pred, y)
#         top5_acc = top5accuracy(y_pred, y)
#         val_top5acc += top5_acc
#         val_acc += acc
#         with open(test_filename,"a") as f:
#             print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

#     val_loss /= len(test_dataloader)
#     val_acc /= len(test_dataloader)