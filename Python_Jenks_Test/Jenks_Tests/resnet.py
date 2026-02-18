'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

_AFFINE = True
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)

        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        if num_blocks == [9,9,9]:
            _outputs = [16, 32, 64]
        else:
            _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]

        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(_outputs[2], num_classes)

        self.apply(weights_init)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else:
                module.do_prune = False
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, **kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
    test_lastlayer = kwargs.get('test_lastlayer', False)
    test_firstlayer = kwargs.get('test_firstlayer', False)
    if test_lastlayer:
        model.linear.do_prune = False
    if test_firstlayer:
        ## Turn off pruning for first conv layer
        first_conv = True
        for name, module in model.named_modules():
            if first_conv and isinstance(module, nn.Conv2d):
                module.do_prune = False
                first_conv = False
    return model



def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    model = ResNet(BasicBlock, [9, 9, 9])
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
            module.do_prune = True
        else:
            module.do_prune = False
    return model


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


# def test(net):
#     import numpy as np
#     total_params = 0

#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()
