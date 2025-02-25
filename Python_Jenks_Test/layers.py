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
                    s_W = torch.mul(param, param.grad)
                    jnb = JenksNaturalBreaks(2)
                    jnb.fit(s_W)
                    labels = jnb.labels_
                    indices = np.where(labels == 1)
                    indices_ = np.where(labels == 0)

                    # Update velocity
                    velocity[indices] = momentum * velocity[indices] + param.grad.data[indices] + scale * param.data[indices]
                    velocity[indices_] = momentum * velocity[indices_] + scale * param.data[indices]

                    # Update parameters
                    param.data[indices] -= lr * velocity[indices]
                    param.data[indices_] -= lr * velocity[indices_]
                else:  # Assuming bias vectors have 1 dimension
                    s_B = torch.mul(param, param.grad)
                    jnb = JenksNaturalBreaks(2)
                    jnb.fit(s_B)
                    labels = jnb.labels_
                    indices = np.where(labels == 1)
                    indices_ = np.where(labels == 0)

                    # Update velocity
                    velocity[indices] = momentum * velocity[indices] + param.grad.data[indices] + scale * param.data[indices]
                    velocity[indices_] = momentum * velocity[indices_] + scale * param.data[indices]

                    # Update parameters
                    param.data[indices] -= lr * velocity[indices]
                    param.data[indices_] -= lr * velocity[indices_]

        return loss

# Example usage:
model = torch.nn.Linear(10, 2)  # Example model
optimizer = JenksSGD(model.parameters(), lr=0.01, scale=0.9, momentum=0.9)

# Dummy input and target
input = torch.randn(10)
target = torch.randn(2)

# Example training loop
for _ in range(100):
    optimizer.zero_grad()
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()