import brevitas
import torch
from torch import log, nn
import os
from datetime import datetime


from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import Accuracy
import sys
net = 'LeNet300'  # Change this to the desired model
dataset = 'tiny_imagenet'  # Change this to the desired dataset
choice = "LeNet300"  # Change this to the desired model
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
root = "../Best_Results_HPO/"+choice

'''WE need to check the log dynamic range or the layers in each network to help decide bit-widths,
mixed precision boundaries, uniform quantization etc.'''

if choice == "DenseNet40":
    from densenet import create_densenet40 as create_model
    folder_name = "DenseNet40_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth")
    csv_file = "DenseNet40_data.csv"
elif choice == "LeNet5":
    from rcnet import create_lenet5 as create_model
    folder_name = "LeNet5_MNIST"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-12-08_15-03-41_MNIST_LeNet5.pth")
    csv_file = "LeNet5_data.csv"
elif choice == "LeNet300":
    def create_model():
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
        )
        return model
    folder_name = "LeNet300_MNIST"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-12-09_18-58-19_MNIST_LeNet300.pth")
    csv_file = "LeNet300_data.csv"
    dataset_root = "../data/MNIST"
    last_layer = "5.weight"
    transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=dataset_root, train=False, download=True,
                                transform=transform)
elif choice == "ResNet56":
    from resnet import resnet56 as create_model
    folder_name = "ResNet56_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth")
    csv_file = "ResNet56_data.csv"
elif choice == "ResNet56e":
    from resnet import resnet56 as create_model
    folder_name = "ResNet56e_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    print(os.getcwd())
    state_dict = torch.load(root+"/best_2025-11-16_18-44-00_CIFAR10_ResNet56.pth")
    csv_file = "ResNet56e_data.csv"
elif choice == "DenseNet40e":
    from densenet import create_densenet40 as create_model
    folder_name = "DenseNet40e_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-16_18-44-04_CIFAR10_DenseNet40.pth")
    csv_file = "DenseNet40e_data.csv"
elif choice == "VGG19_Test":
    from models import vgg19 as create_model
    folder_name = "VGG19_CIFAR10_Test"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2026-01-22_17-17-51_tiny_imagenet_vgg19.pth")
    csv_file = "VGG19_Test_data.csv"
elif choice == "VGG-19/CIFAR-10":
    from models import vgg19 as create_model
    folder_name = "VGG19_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-25_06-06-15_cifar10_vgg19.pth")
    csv_file = "VGG19_C10_data.csv"
elif choice == "VGG-19/CIFAR-100/90_sparsity":
    from models import vgg19 as create_model
    folder_name = "VGG19_CIFAR100_a"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-21_12-14-31_cifar100_vgg19.pth")
    csv_file = "VGG19_C100_a_data.csv"
elif choice == "VGG-19/CIFAR-100/98_sparsity":
    from models import vgg19 as create_model
    folder_name = "VGG19_CIFAR100_b"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-11-23_20-24-12_cifar100_vgg19.pth")
    csv_file = "VGG19_C100_b_data.csv"
elif choice == "VGG-19/TinyImageNet":
    from models import vgg19 as create_model
    folder_name = "VGG19_TinyImageNet"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-12-31_11-17-06_tiny_imagenet_vgg19.pth")
    csv_file = "VGG19_TinyImageNet_data.csv"
elif choice == "ResNet32/CIFAR-10":
    from resnet import resnet32 as create_model
    folder_name = "ResNet32_CIFAR10"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-12-05_00-10-54_cifar10_ResNet_32.pth")
    csv_file = "ResNet32_C10_data.csv"
elif choice == "ResNet32/TinyImageNet":
    from resnet import resnet32 as create_model
    folder_name = "ResNet32_TinyImageNet"
    os.makedirs(f"models/{folder_name}", exist_ok=True)
    state_dict = torch.load(root+"/best_2025-12-15_10-04-01_tiny_imagenet_ResNet_32.pth")
    csv_file = "ResNet32_TinyImageNet_data.csv"



'''Make a folder for this network'''

if net == 'VGG19':
    model = create_model(dataset=dataset)
elif net == 'ResNet32':
    model = create_model(num_classes=200)
else:
    model = create_model()
# 2. Load the state_dict
data = pd.DataFrame(columns=["name", "sparsity"])
model.load_state_dict(state_dict)
'''Now let us look at the layers and make bar graphs of the weights of each layer'''
log_dynamic_ranges = []
def plot_weights(model):
    for name, param in model.named_parameters():
        if param.dim() in [2,4]:  # Only consider weight matrices and convolutional layers
            weights = param.data.cpu().numpy().flatten()
            ''' Drop the weights that are zero'''
            print(f"Before: zeros={np.sum(weights == 0)}")
            weights_no = param.flatten().numel()
            sparsity = np.sum(weights == 0) / weights_no
            data.loc[len(data)] = [name, sparsity]
            weights = weights[weights != 0]
            max_weight = np.max(np.abs(weights))
            min_weight = np.min(np.abs(weights))
            log_dynamic_range = np.log10(max_weight) - np.log10(min_weight)
            log_dynamic_ranges.append((name, log_dynamic_range))

            print(f"After: zeros={np.sum(weights == 0)}")
            plt.figure(figsize=(10, 5))
            plt.hist(weights, bins=100)
            plt.title(f'Weight Histogram of {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.savefig(f"models/{folder_name}/{name}_weights_histogram.png")
            plt.close()
            print(f"File exists after save: {os.path.exists(f'models/{folder_name}/{name}_weights_histogram.png')}")
plot_weights(model)
# Save the sparsity data to a CSV file
data.to_csv(f"models/{folder_name}/{csv_file}", index=False)
plt.figure(figsize=(10, 5))
## plot the log dynamic ranges
names = [item[0] for item in log_dynamic_ranges]
ranges = [item[1] for item in log_dynamic_ranges]
plt.bar(names, ranges)
plt.xticks(rotation=90)
plt.title(f'Log Dynamic Ranges for {choice}')
plt.xlabel('Layer Name')
plt.ylabel('Log10 Dynamic Range')
plt.savefig(f"models/{folder_name}/log_dynamic_ranges_{folder_name}.png")

## Save the log dynamic ranges to a CSV file
log_data = pd.DataFrame(log_dynamic_ranges, columns=["name", "log_dynamic_range"])
log_data.to_csv(f"models/{folder_name}/log_dynamic_ranges_{folder_name}.csv", index=False)
