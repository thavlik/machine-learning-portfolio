from torch import nn
from torch.nn import functional as F


class MaxPool4d(nn.MaxPool3d):

    def __init__(self) -> None:
        super(MaxPool4d, self).__init__()

    def forward(self, input):
        raise NotImplementedError
