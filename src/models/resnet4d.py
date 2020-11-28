import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock4d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock4d, self).__init__()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut != None:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class TransposeBasicBlock4d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(TransposeBasicBlock4d, self).__init__()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut != None:
            out += self.shortcut(x)
        out = F.relu(out)
        return out
