import brevitas
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantTanh, TruncAvgPool2d, BatchNorm2dToQuantScaleBias
from brevitas.utils.torch_utils import TupleSequential
from torch import nn
import torch.nn.functional as F
import torch
import math
from collections import OrderedDict


class FlattenLayer(nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = QuantConv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = QuantConv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = F.sigmoid(x)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        return inputs * x

class QuantConvBuilder(nn.Module):

    def __init__(self, base_config):
        super(QuantConvBuilder, self).__init__()
        print('QuantConvBuilder initialized.')
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
        return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
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
        return QuantLinear(in_features=in_features, out_features=out_features, bias=bias)

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

LENET5_DEPS = [20, 50, 500]
gsm_lr_base_value = 1e-2
gsm_lr_boundaries = [160, 200, 240]
gsm_momentum = 0.99
gsm_max_epochs = 280
weight_decay_strength = 5e-4
batch_size = 256
train_lr_decay_factor = 0.1
warmup_epochs = 5

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

network_type = 'LeNet5'
dataset_name = 'cifar10'
batch_size = 128
base_log_dir = 'gsm_exps/{}_base_train'.format(network_type)
gsm_log_dir = 'gsm_exps/{}_gsm'.format(network_type)

gsm_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=gsm_momentum,
                                     max_epochs=gsm_max_epochs, base_lr=gsm_lr_base_value, lr_epoch_boundaries=gsm_lr_boundaries,
                                     lr_decay_factor=train_lr_decay_factor, linear_final_lr=None,
                                     warmup_epochs=warmup_epochs, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=gsm_log_dir,
                                     tb_dir=gsm_log_dir, save_weights=None, val_epoch_period=2)

class QuantLeNet5(nn.Module):

    def __init__(self, builder:QuantConvBuilder):
        super(QuantLeNet5, self).__init__()
        self.bd = builder
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2d(in_channels=1, out_channels=LENET5_DEPS[0], kernel_size=5, bias=True))
        stem.add_module('relu1', builder.ReLU())
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
        stem.add_module('conv2', builder.Conv2d(in_channels=LENET5_DEPS[0], out_channels=LENET5_DEPS[1], kernel_size=5, bias=True))
        stem.add_module('relu2', builder.ReLU())
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.Linear(in_features=LENET5_DEPS[1] * 16, out_features=LENET5_DEPS[2])
        self.relu1 = builder.ReLU()
        self.linear2 = builder.Linear(in_features=LENET5_DEPS[2], out_features=10)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else:
                module.do_prune = False

    def forward(self, x):
        out = self.stem(x)
        # print(out.size())
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out
    
def quantlenet5():
    return QuantLeNet5(builder=QuantConvBuilder(gsm_config))
    

class QuantLeNet300(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
                nn.Flatten(),
                QuantLinear(in_features=784, out_features=300),
                nn.ReLU(),
                QuantLinear(in_features=300, out_features=100),
                nn.ReLU(),
                QuantLinear(in_features=100, out_features=10),
            )
    def forward(self, x):
        return self.classifier(x)



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


_AFFINE = True
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)

        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                QuantConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
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


class BasicQuantConvBlock(nn.Module):
  ''' The BasicConvBlock takes an input with in_channels, applies sone blocks of convolutional layers
  to reduce it to out_channels and sum it up to the original input
  If their sizes mismatch then the input goes into as identity

  Basically the basic CovnBlock will implement the regular basic Conv Block +
  the shortcut block that doens the dimension matching job ( optionA or B with reference to research paper) when dimension changes between 2 blocks
  '''

  def __init__( self, in_channels, out_channels, stride=1 , option ='A'):
    super(BasicQuantConvBlock, self).__init__()

    self.features = nn.Sequential( OrderedDict([
        ('conv1', QuantConv2d( in_channels, out_channels, kernel_size=3, stride= stride, padding = 1, bias = False )),
        ('bn1', nn.BatchNorm2d(out_channels)),
        ('act1', nn.ReLU(inplace = False)),
        ('conv2', QuantConv2d( out_channels, out_channels, kernel_size =3, stride = 1, padding = 1, bias = False)),
        ('bn2', nn.BatchNorm2d(out_channels))
    ]))

    self.shortcut = nn.Sequential()

    '''When input and output spatial dimension dont match we have 2 opions , with stride:
      - A Use identity shortcuts with zero padding to increase channel dimension.
      - B Use 1x1 convolution to increase channels dimensions ( projection shortcut)
    '''

    if stride != 1 or in_channels != out_channels:
      if option =="A":
        # use identity shortcuts with zero padding to increase channel dimension
        pad_to_add= out_channels//4
        ''' ::2 is doing the job of strid =2
        F.pad apply padding to ( W,H, C,N)

        The padding lengths are specified in reverse order of the dimensions,
        F.pad( x[:, :, ::2, ::2], (0,0, 0,0, pad,pad, 0,0))

        [ width_beginning, widht_end, height_beginning, height_end, channel_beginning, channel_end, batch_beginning, batch_end]
        '''
        self.shortcut = LambdaLayer( lambda x: F.pad(x[:, :, ::2, ::2], (0,0, 0,0, pad_to_add,pad_to_add, 0,0)) )

      if option =="B":
        self.shortcut = nn.Sequential( OrderedDict([
            ( 's_conv1', QuantConv2d( in_channels, 2*out_channels, kernel_size = 1, stride = stride, padding = 0,  bias = False)),
            ( 's_bn1', nn.BatchNorm2d(2*out_channels))
        ]))

  def forward(self, x):
    out = self.features(x)
    shortcut = self.shortcut(x) # the shortcut layer is applied to the input x
    out = out + shortcut # adding the shortcut layer to the current layer output ( the residual is added here )
    out = F.relu(out) # applying the activation functions
    return out

class QuantRCNet(nn.Module):
  def __init__(self, block_type, num_blocks):
    super(QuantRCNet,self).__init__()

    self.in_channels = 16

    self.conv0 = QuantConv2d( 3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False) # first simple conv layer with # of filter = 16
    self.bn0 = nn.BatchNorm2d(16) # batchnormalizaton
    # num_blocks = value of n , here it is 9 and 6n+2 =56
    # _build_layer will implement the 2n layers for each block before pooling into lesser dimension with strides  with n residual connections
    self.block1 = self.__build_layer( block_type, 16 , num_blocks[0], starting_stride = 1) #
    self.block2 = self.__build_layer( block_type, 32 , num_blocks[1], starting_stride = 2)
    self.block3 = self.__build_layer( block_type, 64 , num_blocks[2], starting_stride = 2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.linear = QuantLinear(64,10)

    # building the n layers for each block in resnet
  def __build_layer( self, block_type, out_channels, num_blocks, starting_stride):

    # stride for the first layer is 2 as to decrease the mapping output size to compensate the calculation complexity bcoz of increase the number of channels
    stride_list_for_current_block = [starting_stride] + [1]*(num_blocks-1)
    ''' Above line will generate an array whose first element is starting_stride and it wil have (num_blocks-1) more elements each of value 1'''

    layers = []

    for stride in stride_list_for_current_block:
      layers.append( block_type( self.in_channels, out_channels, stride)) # appending all the convolution layers for a block
      self.in_channels = out_channels

    return nn.Sequential(*layers) # return all the convolution layers stacked in sequence

  def forward(self, x):
    out = F.relu( self.bn0(self.conv0(x))) # need to understand this
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1) # flattening the avpool layer which of size batch_size x channel x height x width i.e. batch_size x 64x1x1
    out = self.linear(out)
    return out
  

class QuantResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QuantResNet, self).__init__()
        if num_blocks == [9,9,9]:
            _outputs = [16, 32, 64]
        else:
            _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]

        self.conv1 = QuantConv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)

        self.linear = nn.Linear(_outputs[2], num_classes)

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
    

def Quantresnet32(num_classes=10, **kwargs):
    model = QuantResNet(QuantBasicBlock, [5, 5, 5], num_classes=num_classes)
    return model



def Quantresnet44():
    return QuantResNet(QuantBasicBlock, [7, 7, 7])


def Quantresnet56():
    model = QuantResNet(QuantBasicBlock, [9, 9, 9])
    return model

class QuantDenseNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(QuantDenseNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = QuantConv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class QuantBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(QuantBottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = QuantConv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = QuantConv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class QuantTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(QuantTransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class QuantDenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(QuantDenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = QuantBottleneckBlock
        else:
            block = QuantDenseNetBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = QuantDenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = QuantTransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = QuantDenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = QuantTransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = QuantDenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = QuantLinear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else:
                module.do_prune = False
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
    
def Quantdensenet40():
    return DenseNet3(depth=40, num_classes=10, growth_rate=12, reduction=1.0, bottleneck=False, dropRate=0.0)


defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class QuantVGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, affine=True, batchnorm=True, test_lastlayer=False, test_firstlayer=False):
        super(QuantVGG, self).__init__()
        
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'cinic-10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = QuantLinear(cfg[-1], num_classes)
        # self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = QuantConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
def quantvgg19(dataset='tiny_imagenet', **kwargs):
    model = QuantVGG(dataset=dataset, depth=19, **kwargs)
    return model


def Torch_to_Brevitas(model, state_dict):
    '''
    Load state_dict into Brevitas model, matching by parameter names and handling shape mismatches.
    
    model: Brevitas model
    state_dict: path to .pth file or dict of saved PyTorch model weights
    '''
    sd = torch.load(state_dict)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('Missing keys when loading state_dict:', missing)
    print('Unexpected keys when loading state_dict:', unexpected)
    return model