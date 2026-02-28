import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import sigmoid
import copy
import gc
from math import sqrt


class QuantLinear(nn.Linear):
    def __init__(self, linear_module, bitwidth=4, mask = None):
        super().__init__(
            linear_module.in_features,
            linear_module.out_features,
            bias=(linear_module.bias is not None)
        )
        self.weight.requires_grad = False
        self.weight.data = linear_module.weight.data.clone()
        if linear_module.bias is not None:
            self.bias.data = linear_module.bias.data.clone()

        w = self.weight.data
        qmax = 2**(bitwidth-1) - 1
        s = w.abs().amax(dim=1, keepdim=True) / qmax + 1e-12
        Q = torch.floor(w / s)

        self.register_buffer("s", s)
        self.register_buffer("Q", Q)
        self.Theta = nn.Parameter(torch.randn_like(self.Q)*0.01, requires_grad=True)
        self.register_buffer('mask', mask if mask is not None else torch.ones_like(self.weight.data))
        self.hard_quant = False
    def forward(self, x):
        if self.hard_quant:
            return F.linear(x, self.weight, self.bias)
        Theta = torch.clip(torch.sigmoid(self.Theta)*1.2-.1, 0, 1)
        W_bar = self.mask * self.s * (self.Q + Theta)
        return F.linear(x, W_bar, self.bias)
    def commit(self):
        with torch.no_grad():
            Theta = (torch.sigmoid(self.Theta)*1.2-.1 > 0.5).float()
            Q_round = torch.round(self.Q + Theta)
            W_q = self.mask * self.s * Q_round
            self.weight.data.copy_(W_q)
            self.hard_quant = True
    

class QuantConv2d(nn.Conv2d):
    def __init__(self, conv_module, bitwidth=4, mask = None):
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
        self.weight.requires_grad = False
        self.weight.data = conv_module.weight.data.clone()
        if conv_module.bias is not None:
            self.bias.data = conv_module.bias.data.clone()

        # symmetric per-output-channel scaling
        w = self.weight.data
        qmax = 2**(bitwidth-1) - 1
        s = w.abs().amax(dim=1, keepdim=True) / qmax + 1e-12
        Q = torch.floor(w / s)

        self.register_buffer("s", s)
        self.register_buffer("Q", Q)
        self.Theta = nn.Parameter(torch.randn_like(self.Q)*0.01, requires_grad=True)
        self.register_buffer('mask', mask if mask is not None else torch.ones_like(self.weight.data))
        self.hard_quant = False
    def forward(self, x):
        if self.hard_quant:
            return F.conv2d(
                x, self.weight, self.bias,
                self.stride, self.padding,
                self.dilation, self.groups
            )
        Theta = torch.clip(torch.sigmoid(self.Theta)*1.2-.1, 0, 1)
        W_bar = self.mask * self.s * (self.Q + Theta)
        return F.conv2d(
            x, W_bar, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
    def commit(self):
        with torch.no_grad():
            self.hard_quant = True
            ## Set theta to 1 or 0
            Theta = (torch.sigmoid(self.Theta)*1.2-.1 > 0.5).float()
            Q_round = torch.round(self.Q + Theta)
            W_q = self.mask * self.s * Q_round
            self.weight.data.copy_(W_q)