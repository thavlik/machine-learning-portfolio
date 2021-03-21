import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1d(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(planes)
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


class TransposeBasicBlock1d(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(TransposeBasicBlock1d, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_planes,
                                        planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes,
                                        planes,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_planes,
                                   planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False),
                nn.BatchNorm1d(planes)
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
