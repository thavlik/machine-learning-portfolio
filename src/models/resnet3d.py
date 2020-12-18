import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm3d(self.expansion * planes)
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


class TransposeBasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TransposeBasicBlock3d, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_planes,
                                        planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.ConvTranspose3d(planes,
                                        planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(in_planes,
                                   self.expansion * planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False),
                nn.BatchNorm3d(planes)
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
