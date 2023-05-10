'''
Code for "Inpainting" figures $6$, $8$ and 7 (top) from the main paper.
'''


from __future__ import print_function

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models.ce_skip import ce_skip

from utils.inpainting_utils import *

import os
import argparse

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main():
    # net = ce_skip(num_input_channels=3, num_output_channels=3)

    net = ce_skip(in_rotations=1, out_rotations=3,
            num_input_channels=3,
            num_output_channels=10,
            num_channels_down=[128] * 5,
            num_channels_up=[128] * 5,
            num_channels_skip=[128] * 5,
            filter_size_up=3, filter_size_down=3,
            upsample_mode='nearest', filter_skip_size=1,
            need_sigmoid=True, need_bias=True, pad='zeros',
            act_fun='LeakyReLU')


    input = torch.rand((1, 3, 224, 224))

    output = net(input)

    print(output.shape)


if __name__ == '__main__':
    main()