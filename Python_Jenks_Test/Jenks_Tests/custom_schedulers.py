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

class WarmupMultiStepJenks(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        alpha=0.5,     # Custom scaling factor for pruning-based adjustment
        beta=0.0,      # Optionally add saliency std as another factor
        last_epoch=-1,
        adjustable = True,
        cosine = False
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.alpha = alpha
        self.beta = beta
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        super(WarmupMultiStepJenks, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
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
        else:
            scaled_lrs = []
            for group in self.optimizer.param_groups:
                name = group.get("name", None)
                base_lr = group['initial_lr'] if 'initial_lr' in group else group['lr']
                milestone_scale = self.gamma ** bisect_right(self.milestones, self.last_epoch)

                # Fetch param group name
                name = group.get("name", None)
                if self.adjustable:
                    if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                        stats = self.optimizer.layerwise_lr_stats.get(name, {})
                        percent_pruned = stats.get('percent_pruned', 0.0)
                        saliency_std = stats.get('saliency_std', 0.0)

                        # Custom scaling logic
                        dynamic_scale = 1.0 + self.alpha * percent_pruned + self.beta * saliency_std
                        if "weight_decay" in group:
                            group["weight_decay"] *= (1-percent_pruned)**self.alpha
                else:
                    dynamic_scale = 1.0  # fallback

                scaled_lr = base_lr * warmup_factor * milestone_scale * dynamic_scale
                scaled_lrs.append(scaled_lr)


            return scaled_lrs
        

class WarmupMultiStepJenksBias(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones_weights,
        milestones_bias,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        alpha=0.5,     # Custom scaling factor for pruning-based adjustment
        beta=0.0,      # Optionally add saliency std as another factor
        last_epoch=-1,
        adjustable = True,
        cosine = False
    ):
        if not list(milestones_weights) == sorted(milestones_weights):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones_weights}"
            )
        if not list(milestones_bias) == sorted(milestones_bias):
            raise ValueError(
                f"Milestones should be a list of increasing integers. Got {milestones_bias}"
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(warmup_method)
            )

        self.milestones_weights = milestones_weights
        self.milestones_bias = milestones_bias
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.alpha = alpha
        self.beta = beta
        self.adjustable = adjustable  # Whether to adjust learning rate based on pruning/saliency
        super(WarmupMultiStepJenksBias, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones_weights, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            scaled_lrs = []
        for group in self.optimizer.param_groups:
            name = group.get("name", None)
            base_lr = group['initial_lr'] if 'initial_lr' in group else group['lr']
            if (name is not None) and ("bias" in name or "bn" in name):
                milestone_scale = self.gamma ** bisect_right(self.milestones_bias, self.last_epoch)
            else:
                milestone_scale = self.gamma ** bisect_right(self.milestones_weights, self.last_epoch)

            if self.adjustable:
                if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                    stats = self.optimizer.layerwise_lr_stats.get(name, {})
                    percent_pruned = stats.get('percent_pruned', 0.0)
                    saliency_std = stats.get('saliency_std', 0.0)
                    dynamic_scale = 1.0 + self.alpha * percent_pruned + self.beta * saliency_std
                    if "weight_decay" in group:
                        group["weight_decay"] *= (1-percent_pruned)**self.alpha
            else:
                dynamic_scale = 1.0  # fallback

            scaled_lr = base_lr * warmup_factor * milestone_scale * dynamic_scale
            scaled_lrs.append(scaled_lr)

        return scaled_lrs
    
class SequentialJenksScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedulers, milestones):
        assert len(schedulers) == len(milestones) + 1, "schedulers should be one more than milestones"
        self.schedulers = schedulers
        self.milestones = milestones
        self.current_idx = 0
        self.current_scheduler = self.schedulers[self.current_idx]
        self.last_epoch = -1
        super().__init__(optimizer)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None, metric=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Switch scheduler if at milestone
        while self.current_idx < len(self.milestones) and epoch >= self.milestones[self.current_idx]:
            self.current_idx += 1
            self.current_scheduler = self.schedulers[self.current_idx]

        # Step the current scheduler
        import inspect
        params = inspect.signature(self.current_scheduler.step).parameters
        if "metrics" in params:
            self.current_scheduler.step(metric)
        elif "epoch" in params:
            self.current_scheduler.step(epoch)
        else:
            self.current_scheduler.step()

        # Update optimizer lrs
        super().step(epoch)

    def get_lr(self):
        return self.current_scheduler.get_last_lr()
        


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
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
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

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
                base_lr * warmup_factor for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr + (self.final_lr - base_lr) * (
                    1 + torch.cos(torch.tensor(self.last_epoch - self.warmup_iters) / (self.final_iters - self.warmup_iters) * 3.141592653589793)) / 2
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


class LayerwiseAdaptiveLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, alpha=1.0, beta=0.1, gamma=0.1,warmup_factor=1.0 / 3,warmup_iters=500,warmup_method="linear",last_epoch=-1):
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta

    def step(self):
        for group in self.optimizer.param_groups:
            name = group.get("name", None)
            if name and hasattr(self.optimizer, "layerwise_lr_stats"):
                stats = self.optimizer.layerwise_lr_stats.get(name, {})
                base_lr = self.optimizer.base_lrs_by_name.get(name, group['lr'])

                # Custom scaling rule
                scale = 1.0 + self.alpha * stats.get('percent_pruned', 0.0) + self.beta * stats.get('saliency_std', 0.0)
                group['lr'] = base_lr * scale




def init_lr_weight_decay(model, learning_rate, weight_decay, bias_weight_decay=0, momentum=0.9, nestrov=False, bias_lr=False):
    base_lrs = {}
    layerwise_lr_stats = {}
    params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Set learning rate and weight decay
        lr = 2 * learning_rate if (bias_lr and 'bias' in name) else learning_rate
        wd = weight_decay
        if "bias" in name or "bn" in name or "BN" in name:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            wd = bias_weight_decay
            print('set weight_decay_bias={} for {}'.format(wd, name))

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

    # Embed layerwise stats and base LRs directly into the optimizer
    optimizer.layerwise_lr_stats = layerwise_lr_stats
    optimizer.base_lrs_by_name = base_lrs

    return optimizer

