import numpy as np
from jenkspy import JenksNaturalBreaks
import torch
from torch.optim import Optimizer
from cuda_helpers import module_weights, module_bias

class JenksSGD(Optimizer):
    def __init__(self, params, lr=5e-3, scale=5e-4, momentum=0.99):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)

                velocity = self.state[param]['velocity']

                # Check if the parameter is a weight matrix or bias vector
                if len(param.shape) > 1:  # Assuming weight matrices have more than 1 dimension
                    # Custom weight update: Scale gradients before applying update
                    # print(torch.mul(param, param.grad).cpu().numpy())
                    s_W = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_W = torch.abs(s_W)
                    unique_values = torch.unique(s_W)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_W)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]

                        # # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)
                        WB_cuda_flatten = s_W.flatten()
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)

                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                        var_min = var.argmin().item()

                        indices_ = WB_cuda_indices[:var_min]
                        indices = WB_cuda_indices[var_min:]

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_B = torch.abs(s_B)
                    unique_values = torch.unique(s_B)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_B)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]
                        B_cuda_sorted, B_cuda_indices = s_B.sort()
                        var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                        var_min = var.argmin().item()
                        # Print the output
                        indices_ = B_cuda_indices[:var_min]
                        indices = B_cuda_indices[var_min:]
                        # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                        
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
        return loss
    
class JenksSGD_Test(Optimizer):
    def __init__(self, params, warmup_epochs, lr=5e-3, scale=5e-4, momentum=0.9, nestrov=False, bias = True):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)
        self.warmup_epochs = warmup_epochs
        self.nestrov = nestrov
        self.bias = bias
        if self.bias:
            self.name = "JenksSGD_Test"
        else:
            self.name = "JenksSGD_Test_Weights"
        ##Define agg scores which should be the same size as the parameters
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['agg_score'] = torch.zeros_like(param.data)
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['lookahead'] = torch.zeros_like(param.data, requires_grad=True)
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['prev_weights'] = torch.zeros_like(param.data, requires_grad=True)

    # @torch.no_grad()
    def step(self, epoch, closure=None):
        torch.cuda.empty_cache()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)
                if 'agg_score' not in self.state[param]:
                    print("Still no agg_score")
                    self.state[param]['agg_score'] = torch.zeros_like(param.data)
                # self.state[param]['velocity'] = param.data - self.state[param]['prev_weights']
                # self.state[param]['prev_weights'] = param.data
                self.state[param]['agg_score']
                # agg_score = self.state[param]['agg_score']

                if self.nestrov:
                    # print(f"self.state[param]['lookahead'] requires_grad: {self.state[param]['lookahead'].requires_grad}")
                    # print(f"param requires_grad: {param.requires_grad}")
                    self.state[param]['lookahead'] = (param - momentum * self.state[param]['velocity']).requires_grad_(True)
                    if param.grad is None:
                        raise RuntimeError("param.grad is None. Ensure gradients are computed before calling step().")
                    computed_grad = torch.autograd.grad(self.state[param]['lookahead'], param, grad_outputs=param.grad, retain_graph=True, allow_unused=True)[0]
                    param.grad = computed_grad.detach()
                if epoch < self.warmup_epochs:
                    # if self.nestrov:
                    #     # velocity = momentum * velocity - lr * (scale * self.state[param]['lookahead'] + param.grad)
                    #     self.state[param]['velocity'].mul_(momentum).add_(lr * (scale * param.data + param.grad))
                    #     param.data.sub_(self.state[param]['velocity'])
                    # else:
                    # velocity = momentum * velocity + scale * param.data + param.grad
                    self.state[param]['velocity'].mul_(momentum).add_(scale * param.data + param.grad)
                    # param.data -= lr * velocity
                    param.data.sub_(lr * self.state[param]['velocity'])
                    # self.state[param]['velocity'] = velocity
                    score = torch.abs(param.grad * param.data)
                    # agg_score += score
                    self.state[param]['agg_score'] += score
                else:
                    # Check if the parameter is a weight matrix or bias vector
                    if param.dim() in [2, 4]:  # Assuming weight matrices have more than 1 dimension
                        # Custom weight update: Scale gradients before applying update
                        # print(torch.mul(param, param.grad).cpu().numpy())
                        s_W = torch.abs(param.data * param.grad)  # Move to CPU, convert to NumPy, and flatten
                        # agg_score += s_W
                        self.state[param]['agg_score'] += s_W
                        # s_W = -s_W
                        unique_values = torch.unique(s_W)
                        n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                        if n_classes > 1:
                            # # Update velocity
                            velocity_flat = self.state[param]['velocity'].view(-1)
                            param_data_flat = param.data.view(-1)
                            param_grad_flat = param.grad.data.view(-1)
                            WB_cuda_flatten = s_W.flatten()
                            WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                            WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)
                            # lookahead = self.state[param]['lookahead']
                            # lookahead_flat = self.state[param]['lookahead'].view(-1)
                            var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                            var_min = var.argmin().item()
                            # print(f"Weight")
                            # print(f"var_min: {var_min}")
                            # print(f"WB_cuda_indices size: {WB_cuda_indices.size()}")

                            indices_ = WB_cuda_indices[:var_min]
                            del WB_cuda_sorted, WB_cuda_indices, var
                            torch.cuda.empty_cache()
                            param_grad_flat[indices_] = 0
                            # if self.nestrov:
                            #     # velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                            #     velocity_flat[indices].mul_(momentum).add_(lr * (scale * param_data_flat[indices] + param_grad_flat[indices]))
                            #     # velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                            #     velocity_flat[indices_].mul_(momentum).add_(lr * (scale * param_data_flat[indices_]))
                            #     # param_data_flat[indices] += velocity_flat[indices]
                            #     param_data_flat.sub_(velocity_flat)
                            #     # param_data_flat[indices_] += velocity_flat[indices_]
                            # else:
                            velocity_flat.mul_(momentum).add_(scale * param_data_flat + param_grad_flat)
                            # velocity_flat[indices_].mul_(momentum).add_(scale * param_data_flat[indices_])
                            # velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                            param_data_flat -= lr * velocity_flat
                            # param_data_flat[indices_] -= lr * velocity_flat[indices_]

                            # Update parameters

                            param.data = param_data_flat.view(param.data.shape)
                            self.state[param]['velocity'] = velocity_flat.view(self.state[param]['velocity'].shape)
                            del param_data_flat, param_grad_flat, velocity_flat
                            torch.cuda.empty_cache()
                        else:
                            velocity = momentum * velocity + scale * param.data + param.grad
                            param.data -= lr * velocity
                            self.state[param]['velocity'] = velocity
                    else:  # Assuming bias vectors have 1 dimension
                        if self.bias:
                            s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                            s_B = torch.abs(s_B)
                            # agg_score += s_B
                            self.state[param]['agg_score'] = s_B
                            unique_values = torch.unique(s_B)
                            n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                            # jnb = JenksNaturalBreaks(n_classes)
                            # jnb.fit(s_B)
                            # labels = jnb.labels_
                            # indices = np.where(labels == 1)[0]
                            # indices_ = np.where(labels == 0)[0]
                            B_cuda_sorted, B_cuda_indices = s_B.sort()
                            var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                            var_min = var.argmin().item()
                            # print(f"Bias")
                            # print(f"var_min: {var_min}")
                            # print(f"B_cuda_indices size: {B_cuda_indices.size()}")
                            # Print the output
                            indices_ = B_cuda_indices[:var_min]
                            indices = B_cuda_indices[var_min:]
                            del B_cuda_sorted, B_cuda_indices, var
                            torch.cuda.empty_cache()
                            # lookahead = self.state[param]['lookahead']
                            # lookahead_flat = lookahead.view(-1)
                            # Update velocity
                            velocity_flat = self.state[param]['velocity'].view(-1)
                            param_data_flat = param.data.view(-1)
                            param_grad_flat = param.grad.data.view(-1)
                            param_grad_flat[indices_] = 0
                            # if self.nestrov:
                            #     # velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                            #     velocity_flat[indices].mul_(momentum).add_(lr * (scale * param_data_flat[indices] + param_grad_flat[indices]))
                            #     # velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                            #     velocity_flat[indices_].mul_(momentum).add_(lr * (scale * param_data_flat[indices_]))
                            #     # param_data_flat[indices] += velocity_flat[indices]
                            #     param_data_flat.sub_(velocity_flat)
                            # else:
                            # velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                            velocity_flat.mul_(momentum).add_(scale * param_data_flat + param_grad_flat)
                            # velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                            param_data_flat -= lr * velocity_flat
                            # Update parameters
                            param.data = param_data_flat.view(param.data.shape)
                            self.state[param]['velocity'] = velocity_flat.view(self.state[param]['velocity'].shape)
                            del param_data_flat, param_grad_flat, velocity_flat
                            torch.cuda.empty_cache()
                        else:
                            self.state[param]['velocity'].mul_(momentum).add_(scale * param.data + param.grad)
                            param.data.sub_(lr * self.state[param]['velocity'])
        return loss
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
    def PruneWeights_Test(self, model):    
        jnb = JenksNaturalBreaks(2)
        for param in model.parameters():
            if param.dim() in [2, 4]: 
                layer = param.data.flatten()
                if 'agg_score' not in self.state[param]:
                    print("agg_score not found for param")
                    break
                score = self.state[param]['agg_score']
                # print(f"agg_score for param: {score}")
                # print(f"agg_score shape: {score.shape}")    
                WB_cuda_flatten = score.flatten()
                # print(f"WB_cuda_flatten shape: {WB_cuda_flatten.shape}")

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                # print(f"WB_cuda_sorted shape: {WB_cuda_sorted.shape}")
                # print(f"WB_cuda_indices shape: {WB_cuda_indices.shape}")
                WB_cuda_sorted = WB_cuda_sorted.reshape(score.shape)

                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()

                # Validate var_min
                if var_min <= 0 or var_min > WB_cuda_indices.size(0):
                    print(f"Invalid var_min: {var_min}")
                    continue

                indices_ = WB_cuda_indices[:var_min]
                layer[indices_] = 0
                layer = layer.reshape(param.data.shape)
                param.data = layer
            elif param.dim() == 1 and self.bias:
                layer = param.data
                if 'agg_score' not in self.state[param]:
                    print("agg_score not found for param")
                    break
                score = self.state[param]['agg_score']
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                # Print the output
                indices_ = B_cuda_indices[:var_min]
                layer[indices_] = 0
                param.data = layer
            else:
                print("Invalid parameter dimension")
                continue
        return model
    

def torch_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        is_correct = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Fix applied here
        res.append(is_correct.mul_(100.0 / batch_size))
    return res

def train_one_step(net, data, label, optimizer, criterion, epoch, warmup_epochs):
    ## Check if the agg score is already defined
    # if epoch == 0:
    for group in optimizer.param_groups:
        for param in group['params']:
            if 'agg_score' not in optimizer.state[param]:
                optimizer.state[param]['agg_score'] = torch.zeros_like(param.data)
            if 'exp_avg' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param.data)
            if 'exp_avg_sq' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            if 'step' not in optimizer.state[param]:
                optimizer.state[param]['step'] = torch.tensor(0, dtype = torch.float32, device = 'cpu')
            if 'mask' not in optimizer.state[param]:
                optimizer.state[param]['mask'] = torch.ones_like(param.data, requires_grad=False)
    optimizer.zero_grad()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    # to_concat_g = []
    # to_concat_v = []
    if epoch > warmup_epochs:
        for name, param in net.named_parameters():
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                for kernel_idx in range(param.shape[0]):
                    kernel = param[kernel_idx]  # Access the kernel (3D tensor)
                    kernel_grad = param.grad[kernel_idx]  # Access the gradient of the kernel

                    # Flatten the kernel and its gradient
                    kernel_data_flat = kernel.view(-1)
                    kernel_grad_flat = kernel_grad.view(-1)

                    # Perform the calculations from lines 386-395
                    WB_cuda_flatten = torch.abs(kernel_data_flat * kernel_grad_flat)
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    kernel_grad_flat[indices_] = 0
                    optimizer.state[param]['agg_score'][kernel_idx] += WB_cuda_flatten.view(kernel.shape)

            elif param.dim() == 2:  # Fully connected layer weights
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += param_grad_flat.view(param.data.shape)

    optimizer.step()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss

def train_one_step_prune(net, data, label, optimizer, criterion, epoch, warmup_epochs, prune_epochs, mask = False):
    ## Check if the agg score is already defined
    # if epoch == 0:
    for group in optimizer.param_groups:
        for param in group['params']:
            if 'agg_score' not in optimizer.state[param]:
                optimizer.state[param]['agg_score'] = torch.zeros_like(param.data)
            if 'exp_avg' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg'] = torch.zeros_like(param.data)
            if 'exp_avg_sq' not in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param.data)
            if 'step' not in optimizer.state[param]:
                optimizer.state[param]['step'] = torch.tensor(0, dtype = torch.float32, device = 'cpu')
            if 'mask' not in optimizer.state[param]:
                optimizer.state[param]['mask'] = torch.ones_like(param.data, requires_grad=False)
    optimizer.zero_grad()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    # to_concat_g = []
    # to_concat_v = []

    if epoch > warmup_epochs and epoch <= prune_epochs:
        for name, param in net.named_parameters():
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                for kernel_idx in range(param.shape[0]):
                    kernel = param.data[kernel_idx]  # Access the kernel (3D tensor)
                    kernel_grad = param.grad[kernel_idx]  # Access the gradient of the kernel

                    # Flatten the kernel and its gradient
                    kernel_data_flat = kernel.view(-1)
                    kernel_grad_flat = kernel_grad.view(-1)

                    # Perform the calculations from lines 386-395
                    WB_cuda_flatten = torch.abs(kernel_data_flat * kernel_grad_flat)
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    kernel_grad_flat[indices_] = 0
                    optimizer.state[param]['agg_score'][kernel_idx] += WB_cuda_flatten.view(kernel.shape)

            elif param.dim() == 2:  # Fully connected layer weights
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                param_data_flat = param.data.view(-1)
                param_grad_flat = param.grad.data.view(-1)
                B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]
                param.grad.view(-1)[indices_] = 0
                optimizer.state[param]['agg_score'] += param_grad_flat.view(param.data.shape)
    if epoch > prune_epochs:
        if not mask:
            for name, param in net.named_parameters():
                if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                    # Iterate over each kernel (slice along the output channel dimension)
                    for kernel_idx in range(param.shape[0]):
                        kernel = param.data[kernel_idx]  # Access the kernel (3D tensor)
                        kernel_grad = param.grad[kernel_idx]  # Access the gradient of the kernel

                        # Flatten the kernel and its gradient
                        kernel_data_flat = kernel.view(-1)
                        kernel_grad_flat = kernel_grad.view(-1)

                        # Perform the calculations from lines 386-395
                        WB_cuda_flatten = torch.abs(kernel_data_flat * kernel_grad_flat)
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                        var_min = var.argmin().item()
                        indices_ = WB_cuda_indices[:var_min]
                        kernel_grad_flat[indices_] = 0
                        optimizer.state[param]['agg_score'][kernel_idx] += WB_cuda_flatten.view(kernel.shape)

                elif param.dim() == 2:  # Fully connected layer weights
                    param_data_flat = param.data.view(-1)
                    param_grad_flat = param.grad.data.view(-1)
                    WB_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]
                    param.grad.view(-1)[indices_] = 0
                    optimizer.state[param]['agg_score'] += WB_cuda_flatten.view(param.data.shape)

                elif param.dim() == 1:  # Bias terms
                    param_data_flat = param.data.view(-1)
                    param_grad_flat = param.grad.data.view(-1)
                    B_cuda_flatten = torch.abs(param_data_flat * param_grad_flat)
                    B_cuda_sorted, B_cuda_indices = B_cuda_flatten.sort()
                    var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                    var_min = var.argmin().item()
                    indices_ = B_cuda_indices[:var_min]
                    param.grad.view(-1)[indices_] = 0
                    optimizer.state[param]['agg_score'] += param_grad_flat.view(param.data.shape)
                
    optimizer.step()
    if epoch > prune_epochs:
        if mask:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data * optimizer.state[param]['mask']
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss    


def Prune_Score(optimizer, kill_velocity=False, mask=False):
    ## Pass through the network and decide which weights to prune based on optimizer.state[param]['agg_score']
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() == 4:  # Convolutional layer weights (4D tensor)
                # Iterate over each kernel (slice along the output channel dimension)
                for kernel_idx in range(param.shape[0]):
                    kernel = param.data[kernel_idx]  # Access the kernel (3D tensor)
                    if 'agg_score' not in optimizer.state[param]:
                        print("agg_score not found for param")
                        break
                    score = optimizer.state[param]['agg_score'][kernel_idx]  # Access the agg_score for this kernel
                    WB_cuda_flatten = score.flatten()

                    # Check for invalid values
                    if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                        print("Invalid values in WB_cuda_flatten")
                        continue

                    WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                    WB_cuda_sorted = WB_cuda_sorted.reshape(kernel.shape)
                    var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                    if torch.isnan(var).any() or torch.isinf(var).any() or var.numel() == 0 or var.shape[0] == 0:
                        print("Invalid values in var")
                        return

                    var_min = var.argmin().item()
                    indices_ = WB_cuda_indices[:var_min]

                    # Prune the kernel
                    with torch.no_grad():  # Disable gradient tracking
                        kernel_flat = kernel.view(-1)
                        kernel_flat[indices_] = 0
                        param[kernel_idx].data = kernel_flat.view(kernel.shape)

                    # Update velocity and mask if required
                        if kill_velocity:
                            if 'velocity' in optimizer.state[param]:
                                optimizer.state[param]['velocity'][kernel_idx].view(-1)[indices_] = 0
                        if mask:
                            if 'mask' in optimizer.state[param]:
                                mask_dummy = torch.ones_like(kernel_flat, requires_grad=False)
                                mask_dummy[indices_] = 0
                                optimizer.state[param]['mask'][kernel_idx] = mask_dummy.view(kernel.shape)
                            else:
                                print("Mask not found in optimizer state")

                        # Update the kernel in the parameter tensor
                        param[kernel_idx] = kernel

            elif param.dim() == 2:  # Fully connected layer weights
                layer = param.data.flatten()
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                score = optimizer.state[param]['agg_score']
                WB_cuda_flatten = score.flatten()

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)
                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = WB_cuda_indices[:var_min]

                # Prune the layer
                layer[indices_] = 0
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'].view(-1)[indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        mask_dummy = torch.ones_like(layer, requires_grad=False)
                        mask_dummy[indices_] = 0
                        optimizer.state[param]['mask'] = mask_dummy.view(param.data.shape)
                    else:
                        print("Mask not found in optimizer state")
                param.data = layer.view(param.data.shape)

            elif param.dim() == 1:  # Bias terms
                layer = param.data
                if 'agg_score' not in optimizer.state[param]:
                    print("agg_score not found for param")
                    break
                score = optimizer.state[param]['agg_score']
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                indices_ = B_cuda_indices[:var_min]

                # Prune the bias
                layer[indices_] = 0
                if kill_velocity:
                    if 'velocity' in optimizer.state[param]:
                        optimizer.state[param]['velocity'][indices_] = 0
                if mask:
                    if 'mask' in optimizer.state[param]:
                        optimizer.state[param]['mask'][indices_] = 0
                    else:
                        print("Mask not found in optimizer state")
                param.data = layer
            else:
                print("Invalid parameter dimension")
                continue




def Prune_Score_Mag(optimizer):
    ## Pass through the network and decide which weights to prune based on optimizer.state[param]['agg_score']
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.dim() in [2, 4]:   
                WB_cuda_flatten = param.data.flatten()
                # print(f"WB_cuda_flatten shape: {WB_cuda_flatten.shape}")

                # Check for invalid values
                if torch.isnan(WB_cuda_flatten).any() or torch.isinf(WB_cuda_flatten).any():
                    print("Invalid values in WB_cuda_flatten")
                    continue

                WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                # print(f"WB_cuda_sorted shape: {WB_cuda_sorted.shape}")
                # print(f"WB_cuda_indices shape: {WB_cuda_indices.shape}")
                WB_cuda_sorted = WB_cuda_sorted.reshape(param.data.shape)

                var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                var_min = var.argmin().item()

                # Validate var_min
                if var_min <= 0 or var_min > WB_cuda_indices.size(0):
                    print(f"Invalid var_min: {var_min}")
                    continue

                indices_ = WB_cuda_indices[:var_min]
                param.data.view(-1)[indices_] = 0
            elif param.dim() == 1:
                score = param.data
                B_cuda_sorted, B_cuda_indices = score.sort()
                var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                var_min = var.argmin().item()
                # Print the output
                indices_ = B_cuda_indices[:var_min]
                param.data.view(-1)[indices_] = 0
            else:
                print("Invalid parameter dimension")
                continue
        




class JenksSGD_Noise(Optimizer):
    def __init__(self, params, lr=0.01, scale=0.9, momentum=0.9):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            momentum = group['momentum']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Initialize velocity if not already done
                if 'velocity' not in self.state[param]:
                    self.state[param]['velocity'] = torch.zeros_like(param.data)

                velocity = self.state[param]['velocity']

                # Check if the parameter is a weight matrix or bias vector
                if len(param.shape) > 1:  # Assuming weight matrices have more than 1 dimension
                    # Custom weight update: Scale gradients before applying update
                    # print(torch.mul(param, param.grad).cpu().numpy())
                    s_W = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_W = torch.abs(s_W)
                    unique_values = torch.unique(s_W)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_W)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]

                        # # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)
                        WB_cuda_flatten = s_W.flatten()
                        WB_cuda_sorted, WB_cuda_indices = WB_cuda_flatten.sort()
                        WB_cuda_sorted = WB_cuda_sorted.reshape(s_W.shape)

                        var = module_weights.jenks_optimization_cuda(WB_cuda_sorted)
                        var_min = var.argmin().item()

                        indices_ = WB_cuda_indices[:var_min]
                        indices = WB_cuda_indices[var_min:]

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        n_params = len(param_data_flat)
                        noise = torch.randn(n_params).to('cuda')
                        param_data_flat.add(noise)
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                    s_B = torch.abs(s_B)
                    unique_values = torch.unique(s_B)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        # jnb = JenksNaturalBreaks(n_classes)
                        # jnb.fit(s_B)
                        # labels = jnb.labels_
                        # indices = np.where(labels == 1)[0]
                        # indices_ = np.where(labels == 0)[0]
                        B_cuda_sorted, B_cuda_indices = s_B.sort()
                        var = module_bias.jenks_optimization_biases_cuda(B_cuda_sorted)
                        var_min = var.argmin().item()
                        # Print the output
                        indices_ = B_cuda_indices[:var_min]
                        indices = B_cuda_indices[var_min:]
                        # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)

                        velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                        velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

                        # Update parameters
                        n_params = len(param_data_flat)
                        noise = torch.randn(n_params).to('cuda')
                        param_data_flat.add(noise)
                        param_data_flat[indices] -= lr * velocity_flat[indices]
                        param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                        
                    else:
                        velocity = momentum * velocity + scale * param.grad
                        param.data -= lr * velocity
        return loss


# import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None: print("Warning: gradient is None")
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


'''Now we need to also have a function in which we take the network and prune based upon magnitude of the weights.'''





def PruneWeights(model):
    # Get the weights of the model, save in different layers
    jnb = JenksNaturalBreaks(2)
    for param in model.parameters():
        layer = param.data.cpu().numpy()
        layer = layer.flatten()
        layer_abs = np.abs(layer)
        jnb.fit(layer_abs)
        labels = jnb.labels_
        indices_ = np.where(labels == 0)[0]
        indices = np.where(labels == 1)[0]
        layer[indices_] = 0
        layer = layer.reshape(param.data.shape)
        param.data = torch.from_numpy(layer)
        param = param.to('cuda')
    return model

def PruneWeights_Test(model, optimizer):
    jnb = JenksNaturalBreaks(2)
    for param in model.parameters():
        layer = param.data.cpu().numpy()
        layer = layer.flatten()
        if 'agg_score' in optimizer.state[param]:
            print(f"agg_score for param: {optimizer.state[param]['agg_score']}")
        else:
            print("agg_score not found for param")
            break
        score = optimizer.state[param]['agg_score']
        layer_abs = np.abs(score.cpu().numpy())
        jnb.fit(layer_abs)
        labels = jnb.labels_
        indices_ = np.where(labels == 0)[0]
        indices = np.where(labels == 1)[0]
        layer[indices_] = 0
        layer = layer.reshape(param.data.shape)
        param.data = torch.from_numpy(layer)
        param = param.to('cuda')
    return model

