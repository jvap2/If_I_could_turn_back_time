import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import sigmoid
import copy
import gc
from math import sqrt


class QuantLinear(nn.Linear):
    def __init__(self, linear_module, bitwidth=4):
        super().__init__(
            linear_module.in_features,
            linear_module.out_features,
            bias=(linear_module.bias is not None)
        )

        self.weight.data = linear_module.weight.data.clone()
        if linear_module.bias is not None:
            self.bias.data = linear_module.bias.data.clone()

        w = self.weight.data
        qmax = 2**(bitwidth-1) - 1
        self.s = w.abs().amax(dim=1, keepdim=True) / qmax + 1e-12
        self.Q = torch.floor(w / self.s)

        self.Theta = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        Theta = torch.sigmoid(self.Theta)
        W_bar = self.s * (self.Q + Theta)
        return F.linear(x, W_bar, self.bias)
    

class QuantConv2d(nn.Conv2d):
    def __init__(self, conv_module, bitwidth=4):
        super().__init__(
            conv_module.in_channels,
            conv_module.out_channels,
            conv_module.kernel_size,
            conv_module.stride,
            conv_module.padding,
            conv_module.dilation,
            conv_module.groups,
            bias=(conv_module.bias is not None),
        )

        self.weight.data = conv_module.weight.data.clone()
        if conv_module.bias is not None:
            self.bias.data = conv_module.bias.data.clone()

        # symmetric per-output-channel scaling
        w = self.weight.data
        qmax = 2**(bitwidth-1) - 1
        self.s = w.abs().amax(dim=(1,2,3), keepdim=True) / qmax + 1e-12
        self.Q = torch.floor(w / self.s)

        self.Theta = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        Theta = torch.sigmoid(self.Theta)
        W_bar = self.s * (self.Q + Theta)
        return F.conv2d(
            x, W_bar, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )