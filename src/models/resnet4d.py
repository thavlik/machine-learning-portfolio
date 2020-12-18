import torch
import torch.nn as nn
import torch.nn.functional as F
from .batchnorm4d import BatchNorm4d
from .conv4d import Conv4d


class BasicBlock4d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock4d, self).__init__()
        self.conv1 = Conv4d(in_planes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn1 = BatchNorm4d(planes)
        self.conv2 = Conv4d(planes,
                            planes,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.bn2 = BatchNorm4d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv4d(in_planes,
                       self.expansion * planes,
                       kernel_size=1,
                       stride=stride,
                       bias=False),
                BatchNorm4d(self.expansion * planes)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class TransposeBasicBlock4d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TransposeBasicBlock4d, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = F.relu(out)
        return out
