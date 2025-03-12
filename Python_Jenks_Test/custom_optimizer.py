import numpy as np
from jenkspy import JenksNaturalBreaks
import torch
from torch.optim import Optimizer
from cuda_helpers import module_weights, module_bias

class JenksSGD(Optimizer):
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


