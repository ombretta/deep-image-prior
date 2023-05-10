"""
Hybrid Color Equivariant ResNet in PyTorch.

Not all stages are color equivariant, only the first `ce_stages` stages are.

Network configs to keep number of parameters close to baseline:

ResNet18 baseline (width=64, num_classes=1000) = 11,689,512
    Separable TRUE:
        - ce_stages=1 --> width=63
        - ce_stages=2 --> width=63
        - ce_stages=3 --> width=61
        - ce_stages=4 --> width=55
    Separable FALSE:
        - ce_stages=1 --> width=63
        - ce_stages=2 --> width=60
        - ce_stages=3 --> width=52
        - ce_stages=4 --> width=37 (check!)

ResNet44 baseline (width=32, num_classes=10) = 2,636,458
    Separable TRUE:
        - ce_stages=1 --> width=31
        - ce_stages=2 --> width=30
        - ce_stages=3 --> width=27
    Separable FALSE:
        - ce_stages=1 --> width=30
        - ce_stages=2 --> width=26
        - ce_stages=3 --> width=18

"""

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


class HybridResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        ce_stages=99,
        num_classes=10,
        rotations=1,
        groupcosetmaxpool=False,
        learnable=False,
        width=64,
        separable=False,
    ) -> None:
        super(HybridResNet, self).__init__()

        assert rotations > 0, "rotations must be greater than 0"
        assert ce_stages > 0, "ce_stages must be greater than 0"

        if groupcosetmaxpool == False and ce_stages < len(num_blocks):
            raise NotImplementedError(
                "Intermediate flattening not implemented, use GroupCosetMaxPool"
            )

        channels = [width * 2**i for i in range(len(num_blocks))]
        self.ce_stages = [i < ce_stages for i in range(len(num_blocks))] + [False]

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
            block_rotations = rotations if self.ce_stages[i] else 1
            self.layers.append(
                self._make_layer(
                    block,
                    channels[i],
                    num_blocks[i],
                    stride=strides[i],
                    rotations=block_rotations,
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
        for i, layer in enumerate(self.layers):
            out = layer(out)

            # Pool or flatten between CE and non-CE stages.
            outs = out.shape
            if self.ce_stages[i + 1] is False and len(out.shape) == 5:
                if self.cosetpoollayer is not None:
                    out = self.cosetpoollayer(out)
                else:
                    out = out.view(outs[0], -1, outs[-2], outs[-1])

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def HybridResNet18(**kwargs):
    return HybridResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def HybridResNet34(**kwargs):
    return HybridResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def HybridResNet44(**kwargs):
    # If width not in kwargs, set to 32.
    kwargs["width"] = kwargs.get("width", 32)
    return HybridResNet(BasicBlock, [7, 7, 7], **kwargs)


def HybridResNet50(**kwargs):
    return HybridResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def HybridResNet101(**kwargs):
    return HybridResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def HybridResNet152(**kwargs):
    return HybridResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    from torchinfo import summary

    summary(
        HybridResNet44(
            rotations=3,
            width=26,
            ce_stages=2,
            num_classes=10,
            separable=False,
            groupcosetmaxpool=True,
        ),
        (2, 3, 224, 224),
        device="cpu",
    )
