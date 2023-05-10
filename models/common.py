import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler
from ceconv.ceconv2d import CEConv2D
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Concat_ce(nn.Module):
    def __init__(self, dim, *args):
        super(Concat_ce, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        # print("inputs", [i.shape for i in inputs])

        inputs_shapes2 = [x.shape[2] for x in inputs] # groups
        inputs_shapes3 = [x.shape[3] for x in inputs] # H
        inputs_shapes4 = [x.shape[4] for x in inputs] # W

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and \
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3)) and \
                np.all(np.array(inputs_shapes4) == min(inputs_shapes4)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            target_shape4 = min(inputs_shapes4)

            inputs_ = []
            for inp in inputs:
                diff3 = (inp.size(3) - target_shape3) // 2
                diff4 = (inp.size(4) - target_shape4) // 2
                inputs_.append(inp[:, :, :target_shape2, diff3: diff3 + target_shape3, diff4:diff4 + target_shape4])

        # print(len(inputs_), inputs_[0].shape, inputs_[1].shape, self.dim)

        return torch.cat(inputs_, dim=self.dim)


    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


class View_to_5d(nn.Module):
    """
        Reshape tensor from 4d to 5d
    """
    def __init__(self, channels, rotations):
        super(View_to_5d, self).__init__()
        self.channels = channels
        self.rotations = rotations

    def forward(self, x):
        return x.view(
            x.shape[0],
            self.channels,
            self.rotations,
            x.shape[-2],
            x.shape[-1],
        )

class View_to_4d(nn.Module):
    """
        Reshape tensor from 5d to 4d
    """
    def __init__(self, channels, rotations):
        super(View_to_4d, self).__init__()
        self.channels = channels
        self.rotations = rotations

    def forward(self, x):
        return x.view(
            x.shape[0],
            self.channels * self.rotations,
            x.shape[-2],
            x.shape[-1],
        )

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)

def bn_3d(num_features):
    return nn.BatchNorm3d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def conv_ce(in_rotations,
            out_rotations,
            in_f, out_f,
            kernel_size,
            stride=1,
            bias=True,
            pad='zero',
            downsample_mode='stride',
            groupcosetmaxpool: bool = False):

    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
                                      preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0


    convolver = CEConv2D(in_rotations, out_rotations, in_f, out_f, kernel_size, stride=stride, padding=to_pad, bias=bias)

    if groupcosetmaxpool is True:
        gmp = GroupCosetMaxPool()
    else:
        gmp = None

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler, gmp])
    return nn.Sequential(*layers)