"""Color Equivariant ResNet in PyTorch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ceconv.ceconv2d import CEConv2D
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, rotations=1, separable=False
    ) -> None:
        super(BasicBlock, self).__init__()

        bnlayer = nn.BatchNorm2d if rotations == 1 else nn.BatchNorm3d
        self.bn1 = bnlayer(planes)
        self.bn2 = bnlayer(planes)

        self.shortcut = nn.Sequential()

        self.stride = stride
        self.kernel_size = 3
        self.padding = 2

        if rotations == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bnlayer(self.expansion * planes),
                )
        else:
            self.conv1 = CEConv2D(
                rotations,
                rotations,
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                separable=separable,
            )
            self.conv2 = CEConv2D(
                rotations,
                rotations,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                separable=separable,
            )
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    CEConv2D(
                        rotations,
                        rotations,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        separable=False,
                    ),
                    bnlayer(self.expansion * planes),
                )

    def forward(self, x) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, rotations=1, separable=False):
        super(Bottleneck, self).__init__()
        bnlayer = nn.BatchNorm2d if rotations == 1 else nn.BatchNorm3d
        self.bn1 = bnlayer(planes)
        self.bn2 = bnlayer(planes)
        self.bn3 = bnlayer(self.expansion * planes)

        self.shortcut = nn.Sequential()

        if rotations == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    bnlayer(self.expansion * planes),
                )
        else:
            self.conv1 = CEConv2D(
                rotations,
                rotations,
                in_planes,
                planes,
                kernel_size=1,
                bias=False,
                separable=separable,
            )
            self.conv2 = CEConv2D(
                rotations,
                rotations,
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                separable=separable,
            )
            self.conv3 = CEConv2D(
                rotations,
                rotations,
                planes,
                self.expansion * planes,
                kernel_size=1,
                bias=False,
                separable=separable,
            )

            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    CEConv2D(
                        rotations,
                        rotations,
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        separable=False,
                    ),
                    bnlayer(self.expansion * planes),
                )

    def forward(self, x) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        rotations=1,
        groupcosetmaxpool=False,
        learnable=False,
        width=64,
        separable=False,
    ) -> None:
        super(ResNet, self).__init__()

        assert rotations > 0, "rotations must be greater than 0"

        # Scale network width to keep number of parameters constant.
        if separable and rotations > 1:
            channels = [
                math.floor(math.sqrt(9 * width**2 / (9 + rotations)) * 2**i)
                for i in range(len(num_blocks))
            ]
        else:
            channels = [
                int(width / math.sqrt(rotations) * 2**i)
                for i in range(len(num_blocks))
            ]
        self.in_planes = channels[0]
        strides = [1, 2, 2, 2]

        # Adjust 3-stage architectures for low-res input, e.g. cifar.
        low_resolution = True if len(num_blocks) == 3 else False
        conv1_kernelsize = 3 if low_resolution else 7
        conv1_stride = 1 if low_resolution else 2
        self.maxpool = nn.Identity()

        # Use CEConv2D for rotations > 1.
        if rotations > 1:
            self.conv1 = CEConv2D(
                1,  # in_rotations
                rotations,
                3,  # in_channels
                channels[0],
                kernel_size=conv1_kernelsize,
                stride=conv1_stride,
                padding=1,
                bias=False,
                learnable=learnable,
                separable=separable,
            )
            self.bn1 = nn.BatchNorm3d(channels[0])
            if not low_resolution:
                self.maxpool = GroupMaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                3,
                channels[0],
                kernel_size=conv1_kernelsize,
                stride=conv1_stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(channels[0])
            if not low_resolution:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Build resblocks
        self.layers = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.layers.append(
                self._make_layer(
                    block,
                    channels[i],
                    num_blocks[i],
                    stride=strides[i],
                    rotations=rotations,
                    separable=separable,
                )
            )
        # Pooling layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cosetpoollayer = None
        if groupcosetmaxpool:
            self.linear = nn.Linear(channels[-1] * block.expansion, num_classes)
            self.cosetpoollayer = GroupCosetMaxPool()
        else:
            self.linear = nn.Linear(
                channels[-1] * rotations * block.expansion, num_classes
            )

    def _make_layer(self, block, planes, num_blocks, stride, rotations, separable):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rotations, separable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)

        # Pool over group dimension or flatten
        if len(out.shape) == 5:
            if self.cosetpoollayer:
                out = self.cosetpoollayer(out)
            else:
                outs = out.size()
                out = out.view(outs[0], outs[1] * outs[2], outs[3], outs[4])

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet44(**kwargs):
    kwargs["width"] = kwargs.get("width", 32)  # If width not in kwargs, set to 32.
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    from torchinfo import summary

    summary(
        ResNet18(rotations=3, separable=True, groupcosetmaxpool=True),
        (2, 3, 224, 224),
        device="cpu",
    )
