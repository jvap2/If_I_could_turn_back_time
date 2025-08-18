import os
import shutil

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms , datasets
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor, device
import sys
from densenet import BasicBlock, DenseBlock, TransitionBlock, BottleneckBlock
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Guiding-Neural-Collapse')))
from nc.models import ddn_modules
from nc.models.ddn_modules import ClosestETFGeometryLayer, FeaturesMovingAverageLayer



class Args:
    def __init__(self, inference=False):
        self.inference = inference

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd

  def forward(self, x):
    return self.lambd(x)

"""In this ResNet implementation we will implement a 56 layer architecture. The total number of stack layers are 6*n+2 ( here n=9 ) thus 56 layers. \
The architecture goes as: \
1. simple Convolutions layers with filter 7x7 and Maxpooling layers reducing the dimensions while increaseing the number of channels. \
2. Then 3x3 convolution are implemented as the ConvBlock containing a stack of 2 layers to which the skip connections will be added i.e. \
a. 3x3 conv + batchnorm + relu \
b. 3x3 conv2d + batchnorm \
c. Then the skip connection (identity) is added to the last output ofcourse the dimensions are matched \
This small architecture is the Residual Block
3. such n block form the first part with feature map size preserved as 32
4. this feature map size is subsampled to 16 with stride of 2
5. then 2 times the process is repeated with number of filters as 32, and 64 repestively

"""

class BasicConvBlock(nn.Module):
  ''' The BasicConvBlock takes an input with in_channels, applies sone blocks of convolutional layers
  to reduce it to out_channels and sum it up to the original input
  If their sizes mismatch then the input goes into as identity

  Basically the basic CovnBlock will implement the regular basic Conv Block +
  the shortcut block that doens the dimension matching job ( optionA or B with reference to research paper) when dimension changes between 2 blocks
  '''

  def __init__( self, in_channels, out_channels, stride=1 , option ='A'):
    super(BasicConvBlock, self).__init__()

    self.features = nn.Sequential( OrderedDict([
        ('conv1', nn.Conv2d( in_channels, out_channels, kernel_size=3, stride= stride, padding = 1, bias = False )),
        ('bn1', nn.BatchNorm2d(out_channels)),
        ('act1', nn.ReLU(inplace = False)),
        ('conv2', nn.Conv2d( out_channels, out_channels, kernel_size =3, stride = 1, padding = 1, bias = False)),
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
            ( 's_conv1', nn.Conv2d( in_channels, 2*out_channels, kernel_size = 1, stride = stride, padding = 0,  bias = False)),
            ( 's_bn1', nn.BatchNorm2d(2*out_channels))
        ]))

  def forward(self, x):
    out = self.features(x)
    shortcut = self.shortcut(x) # the shortcut layer is applied to the input x
    out = out + shortcut # adding the shortcut layer to the current layer output ( the residual is added here )
    out = F.relu(out) # applying the activation functions
    return out

class ResNet(nn.Module):
  '''ResNet-56 Architecture for CIFAR 10 Dataset of shape 32x32x3
  '''
  def __init__(self, block_type, num_blocks):
    super(ResNet,self).__init__()

    self.in_channels = 16

    self.conv0 = nn.Conv2d( 3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False) # first simple conv layer with # of filter = 16
    self.bn0 = nn.BatchNorm2d(16) # batchnormalizaton
    # num_blocks = value of n , here it is 9 and 6n+2 =56
    # _build_layer will implement the 2n layers for each block before pooling into lesser dimension with strides  with n residual connections
    self.block1 = self.__build_layer( block_type, 16 , num_blocks[0], starting_stride = 1) #
    self.block2 = self.__build_layer( block_type, 32 , num_blocks[1], starting_stride = 2)
    self.block3 = self.__build_layer( block_type, 64 , num_blocks[2], starting_stride = 2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.linear = nn.Linear(64,10)

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

def ResNet56():
  return ResNet(block_type = BasicConvBlock, num_blocks= [ 9,9,9])

class ResNetETF(nn.Module):
  '''ResNet-56 Architecture for CIFAR 10 Dataset of shape 32x32x3
  '''
  def __init__(self, block_type, num_blocks, affine=True, **kwargs):
    super(ResNetETF,self).__init__()

    self.in_channels = 16
    self.affine = affine
    self.conv0 = nn.Conv2d( 3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False) # first simple conv layer with # of filter = 16
    self.bn0 = nn.BatchNorm2d(16, affine=affine) # batchnormalizaton
    # num_blocks = value of n , here it is 9 and 6n+2 =56
    # _build_layer will implement the 2n layers for each block before pooling into lesser dimension with strides  with n residual connections
    self.block1 = self.__build_layer( block_type, 16 , num_blocks[0], starting_stride = 1) #
    self.block2 = self.__build_layer( block_type, 32 , num_blocks[1], starting_stride = 2)
    self.block3 = self.__build_layer( block_type, 64 , num_blocks[2], starting_stride = 2)
    self.decl_ETF = kwargs.get('decl_ETF', False)
    self.ETF_fc = kwargs.get('ETF_fc', False)
    self.stand_ETF = kwargs.get('stand_ETF', False)
    self.bias = kwargs.get('bias', True)
    self.device = kwargs.get('device', 'cpu')
    self.K = kwargs['num_classes']
    print(f"decl_ETF: {self.decl_ETF}, ETF_fc: {self.ETF_fc}, stand_ETF: {self.stand_ETF}, bias: {self.bias}")

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    if self.decl_ETF:
        self.W = torch.zeros((kwargs['num_classes'], kwargs['num_features']))
        self.b = torch.zeros(kwargs['num_classes'])
        if not kwargs['args'].inference:
            self.FeaturesMovingAverage = FeaturesMovingAverageLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
            self.ClosestETFGeometry = ClosestETFGeometryLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
    elif self.ETF_fc:
        self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=True)
        self.classifier.bias.requires_grad_(False)
    elif self.stand_ETF:
        self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
    else:
        self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if self.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if self.ETF_fc:
                P = kwargs['P'] if 'P' in kwargs else None
                weight = torch.sqrt(torch.tensor(kwargs['num_classes']/(kwargs['num_classes']-1))) * (
                    torch.eye(kwargs['num_classes'])-(1/kwargs['num_classes'])*torch.ones((kwargs['num_classes'], kwargs['num_classes'])))
                if P is not None:
                    print('P is not None')
                    print(P.shape)
                    m.weight = nn.Parameter(torch.mm(weight, P.T))
                else:
                    m.weight = nn.Parameter(
                        torch.mm(weight, torch.eye(kwargs['num_classes'], kwargs['num_features']))
                    )
                m.weight.requires_grad_(False)
    # building the n layers for each block in resnet
    # Set the layers with param.dim() in [2,4] to have do_prune=True
    for name, module in self.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
            module.do_prune = True
        else:
            module.do_prune = False
    if self.decl_ETF:
        self.FeaturesMovingAverage.do_prune = False
        self.ClosestETFGeometry.do_prune = False
  def __build_layer( self, block_type, out_channels, num_blocks, starting_stride):

    # stride for the first layer is 2 as to decrease the mapping output size to compensate the calculation complexity bcoz of increase the number of channels
    stride_list_for_current_block = [starting_stride] + [1]*(num_blocks-1)
    ''' Above line will generate an array whose first element is starting_stride and it wil have (num_blocks-1) more elements each of value 1'''

    layers = []

    for stride in stride_list_for_current_block:
      layers.append( block_type( self.in_channels, out_channels, stride)) # appending all the convolution layers for a block
      self.in_channels = out_channels

    return nn.Sequential(*layers) # return all the convolution layers stacked in sequence

  def forward(self, x:Tensor, y:Tensor, training)->Tensor:
    self.training = training
    out = F.relu( self.bn0(self.conv0(x))) # need to understand this
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.avgpool(out)
    features = out.view(out.size(0), -1) # flattening the avpool layer which of size batch_size x channel x height x width i.e. batch_size x 64x1x1
    # out = self.linear(out)
    if self.decl_ETF:
        if self.training:
            feature_means, mu_G = self.FeaturesMovingAverage(features, y)
            # with torch.no_grad():
            P = self.ClosestETFGeometry(feature_means)
            weight = torch.sqrt(torch.tensor(self.K/(self.K-1))) * (torch.eye(self.K)-(1/self.K)*torch.ones((self.K, self.K)))
            weight = weight.to(self.device)
            
            self.W = torch.mm(weight, P.T).to(self.device)
            self.b = - 1.0 * torch.mv(self.W, mu_G)
                
            self.W.requires_grad_(True)
            self.b.requires_grad_(True)

        x = F.linear(features, self.W, self.b)

        return x
    elif self.ETF_fc:
        self.classifier.bias.data = - 1.0 * torch.mv(self.classifier.weight, torch.mean(features, dim=0))
        x = F.linear(features, self.classifier.weight, self.classifier.bias)
        return x
    elif self.stand_ETF:
        return self.classifier(features)
    else:
        return self.classifier(features)

def ResNet56ETF(**kwargs):
  return ResNetETF(block_type = BasicConvBlock, num_blocks= [ 9,9,9], **kwargs)


class DenseNet3ETF(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, **kwargs):
        super(DenseNet3ETF, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        self.decl_ETF = kwargs.get('decl_ETF', False)
        self.ETF_fc = kwargs.get('ETF_fc', False)
        self.stand_ETF = kwargs.get('stand_ETF', False)
        self.bias = kwargs.get('bias', True)
        self.device = kwargs.get('device', 'cpu')
        self.K = kwargs['num_classes']

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes
        if self.decl_ETF:
          self.W = torch.zeros((kwargs['num_classes'], kwargs['num_features']))
          self.b = torch.zeros(kwargs['num_classes'])
          if not kwargs['args'].inference:
              self.FeaturesMovingAverage = FeaturesMovingAverageLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
              self.ClosestETFGeometry = ClosestETFGeometryLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
        elif self.ETF_fc:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=True)
            self.classifier.bias.requires_grad_(False)
        elif self.stand_ETF:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        else:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if self.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.ETF_fc:
                    P = kwargs['P'] if 'P' in kwargs else None
                    weight = torch.sqrt(torch.tensor(kwargs['num_classes']/(kwargs['num_classes']-1))) * (
                        torch.eye(kwargs['num_classes'])-(1/kwargs['num_classes'])*torch.ones((kwargs['num_classes'], kwargs['num_classes'])))
                    if P is not None:
                        print('P is not None')
                        print(P.shape)
                        m.weight = nn.Parameter(torch.mm(weight, P.T))
                    else:
                        m.weight = nn.Parameter(
                            torch.mm(weight, torch.eye(kwargs['num_classes'], kwargs['num_features']))
                        )
                    m.weight.requires_grad_(False)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else:
                module.do_prune = False
        if self.decl_ETF:
            self.FeaturesMovingAverage.do_prune = False
            self.ClosestETFGeometry.do_prune = False
    def forward(self, x: Tensor, y: Tensor, training) -> Tensor:
        self.training = training
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(-1, self.in_planes)
        if self.decl_ETF:
            if self.training:
                feature_means, mu_G = self.FeaturesMovingAverage(features, y)
                # with torch.no_grad():
                P = self.ClosestETFGeometry(feature_means)
                weight = torch.sqrt(torch.tensor(self.K/(self.K-1))) * (torch.eye(self.K)-(1/self.K)*torch.ones((self.K, self.K)))
                weight = weight.to(self.device)
                
                self.W = torch.mm(weight, P.T).to(self.device)
                self.b = - 1.0 * torch.mv(self.W, mu_G)
                    
                self.W.requires_grad_(True)
                self.b.requires_grad_(True)

            x = F.linear(features, self.W, self.b)

            return x
        elif self.ETF_fc:
            self.classifier.bias.data = - 1.0 * torch.mv(self.classifier.weight, torch.mean(features, dim=0))
            x = F.linear(features, self.classifier.weight, self.classifier.bias)
            return x
        elif self.stand_ETF:
            return self.classifier(features)
        else:
            return self.classifier(features)


def densenet40ETF(**kwargs):
    return DenseNet3ETF(depth=40, num_classes=10, growth_rate=12, reduction=1.0, bottleneck=False, dropRate=0.0, **kwargs)

class LeNet5V1ETF(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.decl_ETF = kwargs.get('decl_ETF', False)
        self.ETF_fc = kwargs.get('ETF_fc', False)
        self.stand_ETF = kwargs.get('stand_ETF', False)
        self.bias = kwargs.get('bias', True)
        self.device = kwargs.get('device', 'cpu')
        self.K = kwargs['num_classes']

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)   # 28*28->32*32-->28*28
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 14*14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # 10*10
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 5*5
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.classifier = nn.Linear(in_features=84, out_features=10)
        if self.decl_ETF:
          self.W = torch.zeros((kwargs['num_classes'], kwargs['num_features']))
          self.b = torch.zeros(kwargs['num_classes'])
          if not kwargs['args'].inference:
              self.FeaturesMovingAverage = FeaturesMovingAverageLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
              self.ClosestETFGeometry = ClosestETFGeometryLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
        elif self.ETF_fc:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=True)
            self.classifier.bias.requires_grad_(False)
        elif self.stand_ETF:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        else:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if self.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.ETF_fc:
                    P = kwargs['P'] if 'P' in kwargs else None
                    weight = torch.sqrt(torch.tensor(kwargs['num_classes']/(kwargs['num_classes']-1))) * (
                        torch.eye(kwargs['num_classes'])-(1/kwargs['num_classes'])*torch.ones((kwargs['num_classes'], kwargs['num_classes'])))
                    if P is not None:
                        print('P is not None')
                        print(P.shape)
                        m.weight = nn.Parameter(torch.mm(weight, P.T))
                    else:
                        m.weight = nn.Parameter(
                            torch.mm(weight, torch.eye(kwargs['num_classes'], kwargs['num_features']))
                        )
                    m.weight.requires_grad_(False)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else:
                module.do_prune = False
        if self.decl_ETF:
            self.FeaturesMovingAverage.do_prune = False
            self.ClosestETFGeometry.do_prune = False
    def forward(self, x: Tensor, y: Tensor, training) -> Tensor:
        self.training = training
        out = self.conv1(x)
        out = self.tanh1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.tanh2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.tanh3(out)
        out = self.fc2(out)
        out = self.tanh4(out)
        features = out
        if self.decl_ETF:
            if self.training:
                feature_means, mu_G = self.FeaturesMovingAverage(features, y)
                # with torch.no_grad():
                P = self.ClosestETFGeometry(feature_means)
                weight = torch.sqrt(torch.tensor(self.K/(self.K-1))) * (torch.eye(self.K)-(1/self.K)*torch.ones((self.K, self.K)))
                weight = weight.to(self.device)
                
                self.W = torch.mm(weight, P.T).to(self.device)
                self.b = - 1.0 * torch.mv(self.W, mu_G)
                    
                self.W.requires_grad_(True)
                self.b.requires_grad_(True)

            x = F.linear(features, self.W, self.b)

            return x
        elif self.ETF_fc:
            self.classifier.bias.data = - 1.0 * torch.mv(self.classifier.weight, torch.mean(features, dim=0))
            x = F.linear(features, self.classifier.weight, self.classifier.bias)
            return x
        elif self.stand_ETF:
            return self.classifier(features)
        else:
            return self.classifier(features)
        
def LeNet5V1ETFModel(**kwargs):
    return LeNet5V1ETF(**kwargs)


class LeNet300V100ETF(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.decl_ETF = kwargs.get('decl_ETF', False)
        self.ETF_fc = kwargs.get('ETF_fc', False)
        self.stand_ETF = kwargs.get('stand_ETF', False)
        self.bias = kwargs.get('bias', True)
        self.device = kwargs.get('device', 'cpu')
        self.K = kwargs['num_classes']

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(in_features=100, out_features=10)
        if self.decl_ETF:
          self.W = torch.zeros((kwargs['num_classes'], kwargs['num_features']))
          self.b = torch.zeros(kwargs['num_classes'])
          if not kwargs['args'].inference:
              self.FeaturesMovingAverage = FeaturesMovingAverageLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
              self.ClosestETFGeometry = ClosestETFGeometryLayer(kwargs['num_features'], kwargs['num_classes'], device=self.device)
        elif self.ETF_fc:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=True)
            self.classifier.bias.requires_grad_(False)
        elif self.stand_ETF:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        else:
            self.classifier = nn.Linear(kwargs['num_features'], kwargs['num_classes'], bias=self.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if self.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.ETF_fc:
                    P = kwargs['P'] if 'P' in kwargs else None
                    weight = torch.sqrt(torch.tensor(kwargs['num_classes']/(kwargs['num_classes']-1))) * (
                        torch.eye(kwargs['num_classes'])-(1/kwargs['num_classes'])*torch.ones((kwargs['num_classes'], kwargs['num_classes'])))
                    if P is not None:
                        print('P is not None')
                        print(P.shape)
                        m.weight = nn.Parameter(torch.mm(weight, P.T))
                    else:
                        m.weight = nn.Parameter(
                            torch.mm(weight, torch.eye(kwargs['num_classes'], kwargs['num_features']))
                        )
                    m.weight.requires_grad_(False)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.dim() in [2, 4]:
                module.do_prune = True
            else: 
                module.do_prune = False
        if self.decl_ETF:
            self.FeaturesMovingAverage.do_prune = False
            self.ClosestETFGeometry.do_prune = False
    def forward(self, x: Tensor, y:Tensor, training)->Tensor:
        self.training = training
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        features = out
        if self.decl_ETF:
            if self.training:
                feature_means, mu_G = self.FeaturesMovingAverage(features, y)
                # with torch.no_grad():
                P = self.ClosestETFGeometry(feature_means)
                weight = torch.sqrt(torch.tensor(self.K/(self.K-1))) * (torch.eye(self.K)-(1/self.K)*torch.ones((self.K, self.K)))
                weight = weight.to(self.device)
                
                self.W = torch.mm(weight, P.T).to(self.device)
                self.b = - 1.0 * torch.mv(self.W, mu_G)
                    
                self.W.requires_grad_(True)
                self.b.requires_grad_(True)

            x = F.linear(features, self.W, self.b)

            return x
        elif self.ETF_fc:
            self.classifier.bias.data = - 1.0 * torch.mv(self.classifier.weight, torch.mean(features, dim=0))
            x = F.linear(features, self.classifier.weight, self.classifier.bias)
            return x
        elif self.stand_ETF:
            return self.classifier(features)
        else:
            return self.classifier(features)

def LeNet300ETFModel(**kwargs):
    return LeNet300V100ETF(**kwargs)
