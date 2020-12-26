import torch
from torch import nn, Size
from torch.nn import functional as F
from .classifier import Classifier
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from torch import nn, Tensor
from torch.distributions import Normal
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3
from .localizer import Localizer
from .util import get_pooling2d, get_activation
from .resnet_localizer2d import ResNetLocalizer2d


class ResNetGaussianLocalizer2d(ResNetLocalizer2d):
    def __init__(self,
                 kappa: float = 0.05,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.kappa = kappa
        std_dev = [nn.Linear(self.prediction.in_features, 4)]
        if self.batch_normalize:
            std_dev.append(nn.BatchNorm1d(4))
        std_dev.append(self.activation)
        self.std_dev = nn.Sequential(*std_dev)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        mu = self.output(self.prediction(y))
        std_dev = self.std_dev(y)
        pred = reparameterize_normal(mu, std_dev * self.kappa)
        return pred


def reparameterize_normal(mu: Tensor, std_dev: Tensor) -> Tensor:
    return Normal(mu, std_dev).rsample(mu.shape)
