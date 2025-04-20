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