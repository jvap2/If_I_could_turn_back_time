import torch
from custom_optimizer import JenksSGD,PruneWeights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from datetime import datetime
import os
import torch.nn as nn
from networks import LeNet5V1,alexnet
from torch.autograd.functional import hessian
from functions import hutchinson_trace_hmp,rademacher

# train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True)
# test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True)

train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="./datasets", train=False, download=False, transform=transforms.ToTensor())

imgs = torch.stack([img for img, _ in train_val_dataset], dim=0)
def hutchinson_trace_hmp(V, V_batch=1):
    """Hessian trace estimate using BackPACK's HMP extension.

    Perform `V_batch` Hessian multiplications at a time.
    """
    V_count = 0
    trace = 0

    while V_count < V:
        V_missing = V - V_count
        V_next = min(V_batch, V_missing)

        for param in model.parameters():
            v = rademacher((V_next, *param.shape))
            Hv = param.hmp(v).detach()
            vHv = torch.einsum("i,i->", v.flatten(), Hv.flatten().detach())
            trace += vHv / V

        V_count += V_next

    return trace
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
momentum = 0.99
optimizer = JenksSGD(params=model_lenet5v1.parameters(), lr=5e-3, scale=0.5e-4, momentum=momentum)
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
EPOCHS = 2
train_loss, train_acc = 0.0, 0.0
count = 0
original_magnitude = sum(torch.norm(p) for p in model_lenet5v1.parameters())

for epoch in range(EPOCHS):
    # Training loop
    print("Epoch: ", epoch)
    for X, y in train_dataloader:
        count += 1
        X, y = X.to(device), y.to(device)
        
        model_lenet5v1.train()
        
        y_pred = model_lenet5v1(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        mag = 0
        mag = sum(torch.norm(p) for p in model_lenet5v1.parameters())
        acc = accuracy(y_pred, y)
        train_acc += acc
        train_filename = f"output/training_log_{timestamp}_{momentum}.txt"
        with open(train_filename,"a") as f:
            print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Sparsity: {mag/original_magnitude: .5f}", file=f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        params = torch.cat([p.data.flatten() for p in model_lenet5v1.parameters()])

        # # Calculate the Hessian matrix
        # def loss_fn_wrapper(params):
        #     model_lenet5v1.zero_grad()
        #     model_lenet5v1.load_state_dict({name: param.view_as(p) for (name, p), param in zip(model_lenet5v1.named_parameters(), params.split([p.numel() for p in model_lenet5v1.parameters()]))})
        #     y_pred = model_lenet5v1(X)
        #     return loss_fn(y_pred, y)
        
        hessian_matrix = hessian(nn.CrossEntropyLoss(), params)
        # Calculate the trace
        trace = torch.trace(hessian_matrix)
        trace_filename = f"output/trace_log_{timestamp}_{momentum}.txt"
        with open(trace_filename,"a") as f:
            print(f"Iteration: {count}| Trace: {trace: .5f}", file=f)
        
    # train_loss /= len(train_dataloader)
    # train_acc /= len(train_dataloader)
    # Validation loop
val_loss, val_acc = 0.0, 0.0
count_val = 0
prunedmodel = PruneWeights(model_lenet5v1)
'''Make sure the weights are back on the device'''
model_lenet5v1 = prunedmodel.to(device)
model_lenet5v1.eval()
non_zero_params = sum(torch.count_nonzero(p) for p in model_lenet5v1.parameters())
total_params = sum(p.numel() for p in model_lenet5v1.parameters())
sparsity = 1 - non_zero_params / total_params
sparsity_filename = f"output/sparisty_log_{timestamp}_{momentum}.txt"  
with open(sparsity_filename,"a") as f:
    print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
with torch.inference_mode():
    for X, y in val_dataloader:
        count_val += 1
        X, y = X.to(device), y.to(device)
        
        y_pred = model_lenet5v1(X)
        
        loss = loss_fn(y_pred, y)
        val_loss += loss.item()
        
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
torch.save(model_lenet5v1.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")