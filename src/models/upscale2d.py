import torch
from torch import nn


class Upscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):
        super(Upscale2d, self).__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3],
                       1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2],
                                    self.factor * shape[3])
        return x


if __name__ == '__main__':
    assert Upscale2d()(torch.randn(32, 1, 28,
                                   28)).shape == torch.Size([32, 1, 56, 56])
