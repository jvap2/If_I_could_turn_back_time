# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
# from dataset import num_iters_per_epoch
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        final_lr,
        final_iters,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        assert final_iters > warmup_iters
        self.final_lr = final_lr
        self.final_iters = final_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_method = warmup_method
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    #   last_epoch == 0:            base_lr * warmup_factor
    #   last_epoch == warmup_iters: base_lr
    #   last_epoch == final_iters:  final_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError(
                    "Only 'constant' or 'linear' warmup_method accepted"
                    "got {}".format(self.warmup_method)
                )
            return [
                base_lr
                * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr - (base_lr - self.final_lr) * float(self.last_epoch - self.warmup_iters) / (
                            self.final_iters - self.warmup_iters)
                for base_lr in self.base_lrs
            ]

#   LR scheduler should work according the number of iterations
# def get_lr_scheduler(cfg, optimizer):
#     it_ep = num_iters_per_epoch(cfg)
#     if cfg.linear_final_lr is None:
#         lr_iter_boundaries = [it_ep * ep for ep in cfg.lr_epoch_boundaries]
#         return WarmupMultiStepLR(
#             optimizer, lr_iter_boundaries, cfg.lr_decay_factor,
#             warmup_factor=cfg.warmup_factor,
#             warmup_iters=cfg.warmup_epochs * it_ep,
#             warmup_method=cfg.warmup_method, )
#     else:
#         return WarmupLinearLR(optimizer, final_lr=cfg.linear_final_lr,
#                               final_iters=cfg.max_epochs * it_ep,
#                               warmup_factor=cfg.warmup_factor,
#                               warmup_iters=cfg.warmup_epochs * it_ep,
#                               warmup_method=cfg.warmup_method,)



def compute_layer_lr(base_lr, percent_pruned, std_saliency, alpha=1.0, beta=0.1):
    # alpha controls pruning sensitivity; beta controls saliency sensitivity
    scale = 1.0 + alpha * percent_pruned + beta * std_saliency
    return base_lr * scale


class LayerwiseAdaptiveLRScheduler:
    def __init__(self, optimizer, base_lrs, get_saliency_stats_fn):
        self.optimizer = optimizer
        self.base_lrs = base_lrs  # Dict[layer_name] = base_lr
        self.get_saliency_stats_fn = get_saliency_stats_fn

    def step(self):
        saliency_stats = self.get_saliency_stats_fn()  # Should return layerwise dict
        for param_group in self.optimizer.param_groups:
            layer_name = param_group['name']
            stats = saliency_stats[layer_name]
            percent_pruned = stats['percent_pruned']
            std_saliency = stats['std_saliency']
            param_group['lr'] = compute_layer_lr(
                self.base_lrs[layer_name],
                percent_pruned,
                std_saliency
            )




def init_lr_weight_decay(model, learning_rate, weight_decay, momentum=0.9, nestrov=False, bias_lr=False):
    base_lrs = {}
    layerwise_lr_stats = {}
    params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Set learning rate and weight decay
        lr = 2 * learning_rate if (bias_lr and 'bias' in name) else learning_rate
        wd = weight_decay

        # Store base LR and initialize stats
        base_lrs[name] = lr
        layerwise_lr_stats[name] = {
            'percent_pruned': 0.0,
            'saliency_std': 0.0,
        }

        # Add to param group
        params.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
            "name": name  # Tag group with param name
        })

    # Create optimizer
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=nestrov)

    # 🧠 Embed layerwise stats and base LRs directly into the optimizer
    optimizer.layerwise_lr_stats = layerwise_lr_stats
    optimizer.base_lrs_by_name = base_lrs

    return optimizer

