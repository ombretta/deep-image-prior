"""Model definitions for the color MNIST experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ceconv.ceconv2d import CEConv2D
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d


_DROPOUT_FACTOR = 0.3


class CNN(nn.Module):
    """Vanilla Convolutional Neural Network with 7 layers."""

    def __init__(
        self,
        planes: int,
        separable: bool = False,
        num_classes: int = 10,
    ) -> None:

        super().__init__()

        self.conv1 = nn.Conv2d(3, planes, kernel_size=3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv6 = nn.Conv2d(planes, planes, kernel_size=3)
        self.conv7 = nn.Conv2d(planes, planes, kernel_size=4)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.bn5 = nn.BatchNorm2d(planes)
        self.bn6 = nn.BatchNorm2d(planes)
        self.bn7 = nn.BatchNorm2d(planes)

        self.fc = nn.Linear(planes, num_classes)

        self.mp = nn.MaxPool2d(2)
        self.do = nn.Dropout2d(_DROPOUT_FACTOR)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.do(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = self.do(F.relu(self.bn3(self.conv3(x))))
        x = self.do(F.relu(self.bn4(self.conv4(x))))
        x = self.do(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class CECNN(nn.Module):
    """Color Equivariant Convolutional Neural Network (CECNN) with 7 layers."""

    def __init__(
        self,
        planes: int,
        rotations: int,
        groupcosetmaxpool: bool = False,
        num_classes: int = 10,
        learnable: bool = False,
    ) -> None:

        super().__init__()

        assert rotations >= 2, "Rotations must be >= 2."

        self.conv1 = CEConv2D(
            1, rotations, 3, planes, kernel_size=3, learnable=learnable
        )
        self.conv2 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=3, learnable=learnable
        )
        self.conv3 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=3, learnable=learnable
        )
        self.conv4 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=3, learnable=learnable
        )
        self.conv5 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=3, learnable=learnable
        )
        self.conv6 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=3, learnable=learnable
        )
        self.conv7 = CEConv2D(
            rotations, rotations, planes, planes, kernel_size=4, learnable=learnable
        )

        if groupcosetmaxpool is True:
            rotations = 1
            self.gmp = GroupCosetMaxPool()
        else:
            self.gmp = None

        self.mp = GroupMaxPool2d(2)
        self.do = nn.Dropout3d(_DROPOUT_FACTOR)

        self.fc = nn.Linear(rotations * planes, num_classes)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn3 = nn.BatchNorm3d(planes)
        self.bn4 = nn.BatchNorm3d(planes)
        self.bn5 = nn.BatchNorm3d(planes)
        self.bn6 = nn.BatchNorm3d(planes)
        self.bn7 = nn.BatchNorm3d(planes)

    def forward(self, x) -> torch.Tensor:
        x = self.do(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = self.do(F.relu(self.bn3(self.conv3(x))))
        x = self.do(F.relu(self.bn4(self.conv4(x))))
        x = self.do(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        if self.gmp is not None:
            x = self.gmp(x)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    from torchinfo import summary

    summary(CNN(planes=20), (8, 3, 28, 28), device="cpu")
    summary(CNN(planes=56, separable=True), (8, 3, 28, 28), device="cpu")

    summary(CECNN(planes=10, rotations=4), (8, 3, 28, 28), device="cpu")
