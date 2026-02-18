import os
import numpy as np
import torch
import random
import torchvision.transforms.functional as F

def calculate_normalisation_params(train_loader, test_loader):
    """
    Calculate the mean and standard deviation of each channel
    for all observations in training and test datasets. The
    results can then be used for normalisation
    """ 
    chan0 = np.array([])
    chan1 = np.array([])
    chan2 = np.array([])
    
    for i, data in enumerate(train_loader, 0):
        images, _ = data
        chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
        chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
        chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
        
    for i, data in enumerate(test_loader, 0):
        images, _ = data
        chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
        chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
        chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
        
    means = [np.mean(chan0), np.mean(chan1), np.mean(chan2)]
    stds  = [np.std(chan0), np.std(chan1), np.std(chan2)]
    
    return means, stds


class RandomContrast(object):
    def __init__(self, scale_range=(0.9, 1.08)):
        self.scale_range = scale_range

    def __call__(self, img):
        factor = random.uniform(*self.scale_range)
        img = F.adjust_contrast(img, factor)
        # clip back to [0, 1] (we will convert uint8 to float anyway)
        return torch.clamp(img, 0.0, 1.0)


class RandomGamma(object):
    def __init__(self, gamma_range=(0.9, 1.08)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        img = F.adjust_gamma(img, gamma)
        return torch.clamp(img, 0.0, 1.0)

        from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import glob
from PIL import Image
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
VALID_DIR = './datasets/tiny-imagenet-200/val'
TRAIN_DIR = './datasets/tiny-imagenet-200/train'

class TinyImageNetDataset(Dataset):
    def __init__(self, root, id, transform=None, train=False):
        self.transform = transform
        self.id_dict = id
        self.train = train
        if self.train:
            self.filenames = glob.glob(os.path.join(root, "train/*/*/*.JPEG")) 
        else:
            self.filenames = glob.glob(os.path.join(root,"val/images/*.JPEG"))
            self.cls_dic = {}
            for i, line in enumerate(open(os.path.join(root,'val/val_annotations.txt'), 'r')):
                a = line.split('\t')
                img, cls_id = a[0],a[1]
                self.cls_dic[img] = id[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')

        if self.train:
            label = self.id_dict[img_path.split('/')[4]]
        else:
            label = self.cls_dic[img_path.split('/')[-1]]

        if self.transform:
            image = self.transform(image)
        return image, label

