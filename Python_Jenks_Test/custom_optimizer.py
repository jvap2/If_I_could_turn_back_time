import numpy as np
from jenkspy import JenksNaturalBreaks
import torch
from torch.optim import Optimizer

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
                    s_W = torch.mul(param, param.grad).cpu().numpy().flatten()  # Move to CPU, convert to NumPy, and flatten
                    s_W = np.abs(s_W)
                    unique_values = np.unique(s_W)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        jnb = JenksNaturalBreaks(n_classes)
                        jnb.fit(s_W)
                        labels = jnb.labels_
                        indices = np.where(labels == 1)[0]
                        indices_ = np.where(labels == 0)[0]

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
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad).cpu().numpy().flatten()  # Move to CPU, convert to NumPy, and flatten
                    s_B = np.abs(s_B)
                    unique_values = np.unique(s_B)
                    n_classes = min(2, len(unique_values))  # Ensure n_classes is valid
                    if n_classes > 1:
                        jnb = JenksNaturalBreaks(n_classes)
                        jnb.fit(s_B)
                        labels = jnb.labels_
                        indices = np.where(labels == 1)[0]
                        indices_ = np.where(labels == 0)[0]

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

