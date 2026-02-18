from datasets import load_dataset



ds_train = load_dataset("zh-plus/tiny-imagenet", split="train")
ds_valid = load_dataset("zh-plus/tiny-imagenet", split="valid")

'''Now we want to save the images to disk in a folder structure that can be read by ImageFolder'''
import os
from PIL import Image
import numpy as np
import shutil
base_path = "./datasets/TinyImageNet/"
train_path = os.path.join(base_path, "Train")
valid_path = os.path.join(base_path, "Val")
if os.path.exists(base_path):
    shutil.rmtree(base_path)
os.makedirs(train_path)
os.makedirs(valid_path)
# Save training images
for i, item in enumerate(ds_train):
    img = Image.fromarray(np.array(item['image']))
    label = item['label']
    label_path = os.path.join(train_path, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    img.save(os.path.join(label_path, f"{i}.png"))

# Save validation images
for i, item in enumerate(ds_valid):
    img = Image.fromarray(np.array(item['image']))
    label = item['label']
    label_path = os.path.join(valid_path, str(label))
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    img.save(os.path.join(label_path, f"{i}.png"))