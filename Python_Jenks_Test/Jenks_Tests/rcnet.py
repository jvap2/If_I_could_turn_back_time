from torch import nn
from torch.nn import functional as F
import numpy as np

from collections import namedtuple

BaseConfigByEpoch = namedtuple('BaseConfigByEpoch', ['network_type', 'dataset_name', 'dataset_subset', 'global_batch_size', 'num_node', 'device',
                                       'weight_decay', 'weight_decay_bias', 'optimizer_type', 'momentum',
                                       'bias_lr_factor', 'max_epochs', 'base_lr', 'lr_epoch_boundaries', 'lr_decay_factor', 'linear_final_lr',
                                       'warmup_epochs', 'warmup_method', 'warmup_factor',
                                       'ckpt_iter_period', 'tb_iter_period',
                                       'output_dir',  'tb_dir',
                                       'init_weights', 'save_weights',
                                       'val_epoch_period', 'grad_accum_iters',
                                                     'deps',
                                                     'se_reduce_scale'])


def get_baseconfig_by_epoch(network_type, dataset_name, dataset_subset, global_batch_size, num_node,
                    weight_decay, optimizer_type, momentum,
                    max_epochs, base_lr, lr_epoch_boundaries, lr_decay_factor, linear_final_lr,
                    warmup_epochs, warmup_method, warmup_factor,
                    ckpt_iter_period, tb_iter_period,
                    output_dir, tb_dir, save_weights,
                    device='cuda', weight_decay_bias=0, bias_lr_factor=2, init_weights=None, val_epoch_period=-1, grad_accum_iters=1,
                            deps=None,
                            se_reduce_scale=0):
    print('----------------- show lr schedule --------------')
    print('base_lr:', base_lr)
    print('max_epochs:', max_epochs)
    print('lr_epochs:', lr_epoch_boundaries)
    print('lr_decay:', lr_decay_factor)
    print('linear_final_lr:', linear_final_lr)
    print('-------------------------------------------------')

    return BaseConfigByEpoch(network_type=network_type,dataset_name=dataset_name,dataset_subset=dataset_subset,global_batch_size=global_batch_size,num_node=num_node, device=device,
                      weight_decay=weight_decay,weight_decay_bias=weight_decay_bias,optimizer_type=optimizer_type,momentum=momentum,bias_lr_factor=bias_lr_factor,
                      max_epochs=max_epochs, base_lr=base_lr, lr_epoch_boundaries=lr_epoch_boundaries,lr_decay_factor=lr_decay_factor, linear_final_lr=linear_final_lr,
                             warmup_epochs=warmup_epochs,warmup_method=warmup_method,warmup_factor=warmup_factor,
                      ckpt_iter_period=int(ckpt_iter_period),tb_iter_period=int(tb_iter_period),
                      output_dir=output_dir, tb_dir=tb_dir,
                      init_weights=init_weights, save_weights=save_weights,
                             val_epoch_period=val_epoch_period, grad_accum_iters=grad_accum_iters, deps=deps, se_reduce_scale=se_reduce_scale)

network_type = 'rc56'
dataset_name = 'cifar10'
batch_size = 128
base_log_dir = 'gsm_exps/{}_base_train'.format(network_type)
gsm_log_dir = 'gsm_exps/{}_gsm'.format(network_type)

base_train_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                            global_batch_size=batch_size, num_node=1, weight_decay=1e-4, optimizer_type='sgd',
                                            momentum=0.9, max_epochs=500, base_lr=0.1, lr_epoch_boundaries=[100, 200, 300, 400],
                                            lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                            warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                            output_dir=base_log_dir, tb_dir=base_log_dir, save_weights=None,
                                            val_epoch_period=2)

gsm_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                            global_batch_size=batch_size, num_node=1, weight_decay=1e-4, optimizer_type='sgd',
                                            momentum=0.98, max_epochs=600, base_lr=5e-3, lr_epoch_boundaries=[400, 500],     # Note this line
                                            lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                            warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                            output_dir=gsm_log_dir, tb_dir=gsm_log_dir, save_weights=None,
                                            val_epoch_period=2)

class FlattenLayer(nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)
    

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = F.sigmoid(x)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        return inputs * x

class ConvBuilder(nn.Module):

    def __init__(self, base_config):
        super(ConvBuilder, self).__init__()
        print('ConvBuilder initialized.')
        self.BN_eps = 1e-5
        self.BN_momentum = 0.1
        self.BN_affine = True
        self.BN_track_running_stats = True
        self.base_config = base_config

    def set_BN_config(self, eps, momentum, affine, track_running_stats):
        self.BN_eps = eps
        self.BN_momentum = momentum
        self.BN_afine = affine
        self.BN_track_running_stats = track_running_stats


    def Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_original_conv=False):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    # The running estimates are kept with a default momentum of 0.1.
    # By default, the elements of \gammaγ are sampled from \mathcal{U}(0, 1)U(0,1) and the elements of \betaβ are set to 0.
    # If track_running_stats is set to False, this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well.
    def BatchNorm2d(self, num_features, eps=None, momentum=None, affine=None, track_running_stats=None):
        if eps is None:
            eps = self.BN_eps
        if momentum is None:
            momentum = self.BN_momentum
        if affine is None:
            affine = self.BN_affine
        if track_running_stats is None:
            track_running_stats = self.BN_track_running_stats
        return nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    # def _succeedingBN2d(self, num_features, eps=None, momentum=None, affine=None, track_running_stats=None):
    #     return self._BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    #
    # def SeparateBN2d(self, num_features, eps=None, momentum=None, affine=None, track_running_stats=None):
    #     return self._BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def Sequential(self, *args):
        return nn.Sequential(*args)

    def ReLU(self):
        return nn.ReLU()

    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        conv_layer = self.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, use_original_conv=use_original_conv)
        bn_layer = self.BatchNorm2d(num_features=out_channels)
        se = self.Sequential()
        se.add_module('conv', conv_layer)
        se.add_module('bn', bn_layer)
        if self.base_config is not None and self.base_config.se_reduce_scale is not None and self.base_config.se_reduce_scale > 0:
            se.add_module('se', SEBlock(input_channels=out_channels, internal_neurons=out_channels // self.base_config.se_reduce_scale))
        return se

    def Conv2dBNReLU(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        conv = self.Conv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, use_original_conv=use_original_conv)
        conv.add_module('relu', self.ReLU())
        return conv

    def BNReLUConv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_original_conv=False):
        bn_layer = self.BatchNorm2d(num_features=in_channels)
        conv_layer = self.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        se = self.Sequential()
        se.add_module('bn', bn_layer)
        se.add_module('relu', self.ReLU())
        se.add_module('conv', conv_layer)
        return se

    def Linear(self, in_features, out_features, bias=True):
        return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def Identity(self):
        return nn.Identity()

    def ResIdentity(self, num_channels):
        return nn.Identity()

    def Dropout(self, keep_prob):
        return nn.Dropout(p=1-keep_prob)

    def Maxpool2d(self, kernel_size, stride=None):
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def Avgpool2d(self, kernel_size, stride=None):
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def Flatten(self):
        return FlattenLayer()

    def GAP(self, kernel_size):
        gap = nn.Sequential()
        gap.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size))
        gap.add_module('flatten', FlattenLayer())
        return gap



    def relu(self, in_features):
        return F.relu(in_features)

    def max_pool2d(self, in_features, kernel_size, stride, padding):
        return F.max_pool2d(in_features, kernel_size=kernel_size, stride=stride, padding=padding)

    def avg_pool2d(self, in_features, kernel_size, stride, padding):
        return F.avg_pool2d(in_features, kernel_size=kernel_size, stride=stride, padding=padding)

    def flatten(self, in_features):
        result = in_features.view(in_features.size(0), -1)
        return result

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, builder:ConvBuilder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = builder.Conv2dBN(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = builder.ResIdentity(num_channels=in_planes)

        self.conv1 = builder.Conv2dBNReLU(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = builder.Conv2dBN(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder:ConvBuilder, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bd = builder

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = builder.Conv2dBN(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = builder.ResIdentity(num_channels=in_planes)

        self.conv1 = builder.Conv2dBNReLU(in_planes, planes, kernel_size=1)
        self.conv2 = builder.Conv2dBNReLU(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = builder.Conv2dBN(planes, self.expansion*planes, kernel_size=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class RCBlock(nn.Module):

    def __init__(self, in_channels, conv1_channels, conv2_channels, stride=1, builder=None):
        super(RCBlock, self).__init__()

        if stride != 1:
            self.shortcut = builder.Conv2dBN(in_channels, conv2_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = builder.ResIdentity(num_channels=conv2_channels)

        self.conv1 = builder.Conv2dBNReLU(in_channels, conv1_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = builder.Conv2dBN(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)

        self.relu = builder.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

def rc_origin_deps_flattened(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    stage1 = [16] * filters_in_each_stage
    stage2 = [32] * filters_in_each_stage
    stage3 = [64] * filters_in_each_stage
    return np.array(stage1 + stage2 + stage3)

def rc_pacesetter_idxes(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    pacesetters = [0, int(filters_in_each_stage), int(2 * filters_in_each_stage)]
    return pacesetters

def rc_convert_flattened_deps(flattened):
    filters_in_each_stage = len(flattened) / 3
    n = int((filters_in_each_stage - 1) // 2)
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = [flattened[0]]
    for ps in pacesetters:
        assert flattened[ps] == flattened[ps+2]
        stage_deps = []
        for i in range(n):
            stage_deps.append([flattened[ps + 1 + 2 * i], flattened[ps + 2 + 2 * i]])
        result.append(stage_deps)
    return result

class RCNet(nn.Module):

    def __init__(self, block_counts, num_classes, builder:ConvBuilder, deps):
        super(RCNet, self).__init__()
        self.bd = builder
        assert block_counts[0] == block_counts[1]
        assert block_counts[1] == block_counts[2]
        if deps is None:
            deps = rc_origin_deps_flattened(block_counts[0])
        deps = rc_convert_flattened_deps(deps)

        self.conv1 = self.bd.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=3, stride=1, padding=1)
        self.stage1 = self._build_stage(stage_in_channels=deps[0], stage_channels=deps[1], num_blocks=block_counts[0], stride=1)
        self.stage2 = self._build_stage(stage_in_channels=deps[1][-1][-1], stage_channels=deps[2], num_blocks=block_counts[1], stride=2)
        self.stage3 = self._build_stage(stage_in_channels=deps[2][-1][-1], stage_channels=deps[3], num_blocks=block_counts[2], stride=2)
        self.linear = self.bd.Linear(in_features=deps[3][-1][-1], out_features=num_classes)

    def _build_stage(self, stage_in_channels, stage_channels, num_blocks, stride):
        layers = []
        assert num_blocks == len(stage_channels)
        in_channels = stage_in_channels
        for i in range(num_blocks):
            layers.append(RCBlock(in_channels=in_channels,
                                    conv1_channels=stage_channels[i][0],
                                  conv2_channels=stage_channels[i][1],
                                  stride=stride if i == 0 else 1, builder=self.bd))
            in_channels = stage_channels[i][1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.bd.avg_pool2d(in_features=out, kernel_size=8, stride=1, padding=0)
        out = self.bd.flatten(out)
        out = self.linear(out)
        return out


def create_RC56():
    return RCNet(block_counts=[9,9,9], num_classes=10, builder=ConvBuilder(gsm_config), deps=gsm_config.deps)


class ResNet(nn.Module):
    def __init__(self, builder:ConvBuilder, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.bd = builder
        self.in_planes = 64
        self.conv1 = builder.Conv2dBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.stage1 = self._make_stage(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, 512, num_blocks[3], stride=2)
        self.linear = self.bd.Linear(512*block.expansion, num_classes)

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(builder=self.bd, in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bd.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.bd.avg_pool2d(out, 7, 1, 0)
        out = self.bd.flatten(out)
        out = self.linear(out)
        return out

def create_ResNet18():
    return ResNet(builder=ConvBuilder(gsm_config), block=BasicBlock, num_blocks=[2,2,2,2], num_classes=1000)