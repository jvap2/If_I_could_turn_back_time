import os, sys, subprocess, tempfile

def _setup_vs_env():
    vcvarsall = (r"C:\Program Files (x86)\Microsoft Visual Studio"
                 r"\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat")
    if not os.path.exists(vcvarsall):
        return
    bat = tempfile.NamedTemporaryFile(suffix=".bat", delete=False, mode="w")
    bat.write(f'@echo off\ncall "{vcvarsall}" x64\nset\n')
    bat.close()
    out = subprocess.check_output(["cmd", "/c", bat.name],
                                  stderr=subprocess.DEVNULL).decode("utf-8", errors="replace")
    os.unlink(bat.name)
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip()

if sys.platform == "win32":
    _setup_vs_env()

_CUDA_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
os.environ.setdefault("CUDA_HOME", _CUDA_ROOT)
os.environ.setdefault("CUDA_PATH", _CUDA_ROOT)

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import random
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.models import resnet50
from datasets import load_dataset

from torch.nn.init import orthogonal_
from custom_optimizer import Prune_Score, train_one_step_prune, Prune_Score_Select, init_network
from custom_schedulers import init_lr_weight_decay, WarmupAutoJenks, singular_value
from training_loop import train_val_loop_HPO
from cuda_helpers import get_memory_free_MiB
import custom_optimizer

# ── Transforms (top-level so workers can pickle them) ────────────────────────

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


class HFStreamingImageNet(IterableDataset):
    """Streams ILSVRC/imagenet-1k from HuggingFace without local download."""

    def __init__(self, split, transform=None, shuffle=False, buffer_size=2000):
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        dataset = load_dataset(
            "ILSVRC/imagenet-1k", split=self.split,
            streaming=True,
        )
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )
        if self.shuffle:
            dataset = dataset.shuffle(
                seed=random.randint(0, 99999), buffer_size=self.buffer_size
            )
        for sample in dataset:
            img = sample["image"].convert("RGB")
            if self.transform:
                img = self.transform(img)
            yield img, sample["label"]


if __name__ == "__main__":

    # ── Config ───────────────────────────────────────────────────────────────

    device        = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE    = 64
    NUM_WORKERS   = 2
    one_shot      = True
    prune_ratio   = 0.5
    mask          = True
    kill_velocity = False
    bias_prune    = False
    one_update    = True
    accum_steps   = 8
    USE_EMA       = True
    EMA_DECAY     = 0.9999

    label_smoothing   = 0.1
    momentum          = 0.99
    learning_rate     = 5e-2
    weight_decay      = 5e-4
    bias_weight_decay = 2e-4
    warmup_epochs     = 10
    nestrov           = True
    bias_lr           = True
    prune_epoch       = 350
    prune_between     = 5
    reset             = False
    rewind_epoch      = None

    EPOCHS            = 400
    min_epochs        = 300
    gsm_lr_boundaries = [200, 230, 260]

    custom_optimizer.MIXUP           = True
    custom_optimizer.MIXUP_OFF_EPOCH = EPOCHS - 20

    # ── Datasets & DataLoaders ────────────────────────────────────────────────

    train_dataset = HFStreamingImageNet("train", transform=train_transform,
                                        shuffle=True, buffer_size=500)
    val_dataset   = HFStreamingImageNet("validation", transform=val_transform,
                                        shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    model = resnet50(weights=None, progress=True)
    model = model.to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = init_lr_weight_decay(
        model, learning_rate, weight_decay,
        bias_weight_decay=bias_weight_decay,
        momentum=momentum, nestrov=nestrov,
        bias_lr=bias_lr, elem_bias=True,
        warmup_epochs=warmup_epochs, prune_epoch=prune_epoch,
    )
    init_network(optimizer)

    scheduler = WarmupAutoJenks(
        optimizer, milestones=gsm_lr_boundaries,
        warmup_factor=1/2, warmup_iters=warmup_epochs,
        prune_epochs=prune_epoch, reset=reset, rewind_epoch=rewind_epoch,
    )

    num_classes  = 1000
    accuracy     = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    top5accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)

    # ── EMA ───────────────────────────────────────────────────────────────────

    if USE_EMA:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    else:
        ema_model = None

    # ── Logging ───────────────────────────────────────────────────────────────

    timestamp       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = "ImageNet"
    model_name      = "ResNet50"
    log_dir         = os.path.join("runs", timestamp, experiment_name, model_name)
    writer          = SummaryWriter(log_dir)
    train_dir       = "ResNet50_ImageNet_output/"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    name              = "SGD_Agg"
    log_filename      = os.path.join(train_dir, f"log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    train_filename    = os.path.join(train_dir, f"training_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    sparsity_filename = os.path.join(train_dir, f"sparisty_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    val_filename      = os.path.join(train_dir, f"validation_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    debug_filename    = os.path.join(train_dir, f"debug_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    jenks_filename    = os.path.join(train_dir, f"jenks_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")
    prune_filename    = os.path.join(train_dir, f"prune_log_{timestamp}_{momentum}_{name}_{EPOCHS}.txt")

    total_params        = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_pruned_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    print(f"Using {device} device")
    print(f"Total trainable parameters: {total_params}")
    print(f"Total prunable parameters:  {total_pruned_params}")

    with open(log_filename, "a") as f:
        print(f"Starting Learning rate: {learning_rate}", file=f)
        print(f"Momentum: {momentum}", file=f)
        print(f"Weight decay: {weight_decay}", file=f)
        print(f"Batch size: {BATCH_SIZE} (accum_steps={accum_steps}, effective={BATCH_SIZE*accum_steps})", file=f)
        print(f"Epochs: {EPOCHS}", file=f)
        print(f"Prune epoch: {prune_epoch}", file=f)
        print(f"Warmup epochs: {warmup_epochs}", file=f)
        print(f"Nesterov: {nestrov}", file=f)
        print(f"Label smoothing: {label_smoothing}", file=f)
        print(f"EMA: {USE_EMA} (decay={EMA_DECAY})", file=f)
        print(f"MixUp: on (off after epoch {EPOCHS - 20})", file=f)

    # ── Training ──────────────────────────────────────────────────────────────

    prune_epoch_list = [prune_epoch]
    prune_count      = 0
    sparsity         = 0.0
    lambda_          = 0

    train_val_loop_HPO(
        model, train_dataloader, val_dataloader,
        optimizer, loss_fn, scheduler,
        accuracy, top5accuracy, writer, device,
        experiment_name, model_name, timestamp,
        train_filename=train_filename, val_filename=val_filename,
        log_filename=log_filename, sparsity_filename=sparsity_filename,
        prune_filename=prune_filename, debug_filename=debug_filename,
        jenks_filename=jenks_filename,
        prune_count=prune_count, one_update=one_update,
        EPOCHS=EPOCHS, sparsity=sparsity,
        prune_epoch_list=prune_epoch_list, prune_epoch=prune_epoch,
        prune_between=prune_between, prune_ratio=prune_ratio,
        one_shot=one_shot, mask=mask,
        mag_prune=True, bias_prune=bias_prune, kill_velocity=kill_velocity,
        l2=False, lambda_=lambda_,
        warmup_epochs=warmup_epochs, min_epochs=min_epochs,
        elem_bias=True, accum_steps=accum_steps,
        ema_model=ema_model,
    )

    if ema_model is not None:
        torch.save(
            ema_model.module.state_dict(),
            f"models/ema_final_{timestamp}_{experiment_name}_{model_name}.pth",
        )
        print("EMA model saved.")
