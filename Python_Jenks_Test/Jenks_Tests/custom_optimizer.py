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
    def __init__(self, params, warmup_epochs, lr=5e-3, scale=5e-4, momentum=0.9, nestrov=False):
        defaults = dict(lr=lr, scale=scale, momentum=momentum)
        super().__init__(params, defaults)
        self.warmup_epochs = warmup_epochs
        self.nestrov = nestrov
        ##Define agg scores which should be the same size as the parameters
        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['agg_score'] = torch.zeros_like(param.data)
        for group in self.param_groups:
            for param in group['params']:
                self.lookahead_param = torch.zeros_like(param.data)

    @torch.no_grad()
    def step(self, epoch, closure=None):
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
                velocity = self.state[param]['velocity']
                agg_score = self.state[param]['agg_score']

                if self.nestrov:
                    self.lookahead_param = param.data + momentum * velocity
                    param.grad = torch.autograd.grad(self.lookahead_velocity, param, grad_outputs=param.grad, retain_graph=True)[0]
                if epoch < self.warmup_epochs:
                    if self.nestrov:
                        velocity = momentum * velocity - lr * (scale * self.lookahead_param + param.grad)
                        param.data += velocity
                    else:
                        velocity = momentum * velocity + scale * param.data + param.grad
                        param.data -= lr * velocity
                    self.state[param]['velocity'] = velocity
                    score = torch.abs(param.grad * param.data)
                    agg_score += score
                    self.state[param]['agg_score'] = agg_score
                else:
                    # Check if the parameter is a weight matrix or bias vector
                    if param.dim() in [2, 4]:  # Assuming weight matrices have more than 1 dimension
                        # Custom weight update: Scale gradients before applying update
                        # print(torch.mul(param, param.grad).cpu().numpy())
                        s_W = torch.abs(param * param.grad)  # Move to CPU, convert to NumPy, and flatten
                        agg_score += s_W
                        self.state[param]['agg_score'] = agg_score
                        # s_W = -s_W
                        unique_values = torch.unique(s_W)
                        n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                        if n_classes > 1:
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
                            if self.nestrov:
                                velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                                velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                                param_data_flat[indices] += velocity_flat[indices]
                                param_data_flat[indices_] += velocity_flat[indices_]
                            else:
                                velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                                velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                                param_data_flat[indices] -= lr * velocity_flat[indices]
                                param_data_flat[indices_] -= lr * velocity_flat[indices_]

                            # Update parameters

                            param.data = param_data_flat.view(param.data.shape)
                            self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
                        else:
                            velocity = momentum * velocity + scale * param.data + param.grad
                            param.data -= lr * velocity
                            self.state[param]['velocity'] = velocity
                    else:  # Assuming bias vectors have 1 dimension
                        s_B = torch.mul(param, param.grad)  # Move to CPU, convert to NumPy, and flatten
                        s_B = torch.abs(s_B)
                        agg_score += s_B
                        self.state[param]['agg_score'] = agg_score
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
                        # Print the output
                        indices_ = B_cuda_indices[:var_min]
                        indices = B_cuda_indices[var_min:]
                        # Update velocity
                        velocity_flat = velocity.view(-1)
                        param_data_flat = param.data.view(-1)
                        param_grad_flat = param.grad.data.view(-1)
                        if self.nestrov:
                            velocity_flat[indices] = momentum * velocity_flat[indices] - lr * (scale * param_data_flat[indices] + param_grad_flat[indices])
                            velocity_flat[indices_] = momentum * velocity_flat[indices_] - lr * (scale * param_data_flat[indices_] + param_grad_flat[indices_])
                            param_data_flat[indices] += velocity_flat[indices]
                            param_data_flat[indices_] += velocity_flat[indices_]
                        else:
                            velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
                            velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]
                            param_data_flat[indices] -= lr * velocity_flat[indices]
                            param_data_flat[indices_] -= lr * velocity_flat[indices_]
                        # Update parameters
                        param.data = param_data_flat.view(param.data.shape)
                        self.state[param]['velocity'] = velocity_flat.view(velocity.shape)
        return loss
    def PruneWeights_Test(self, model):    
        jnb = JenksNaturalBreaks(2)
        for param in model.parameters():
            if param.dim() in [2, 4]: 
                layer = param.data.flatten()
                if 'agg_score' not in self.state[param]:
                    print("agg_score not found for param")
                    break
                score = self.state[param]['agg_score']
                print(f"agg_score for param: {score}")
                print(f"agg_score shape: {score.shape}")    
                WB_cuda_flatten = score.flatten()
                print(f"WB_cuda_flatten shape: {WB_cuda_flatten.shape}")

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
            elif param.dim() == 1:
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


import torch


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

