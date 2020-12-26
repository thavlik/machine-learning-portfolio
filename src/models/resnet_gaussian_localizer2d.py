import torch
from torch import nn, Size
from torch.nn import functional as F
from .classifier import Classifier
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3
from .localizer import Localizer
from .util import get_pooling2d, get_activation, reparameterize
from .resnet_localizer2d import ResNetLocalizer2d


class ResNetGaussianLocalizer2d(ResNetLocalizer2d):
    def __init__(self,
                 kappa: float = 0.05,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.kappa = kappa
        log_var = [nn.Linear(self.prediction.in_features, 4)]
        if self.batch_normalize:
            log_var.append(nn.BatchNorm1d(4))
        log_var.append(self.activation)
        self.log_var = nn.Sequential(*log_var)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        mu = self.output(self.prediction(y))
        log_var = self.log_var(y)
        pred = reparameterize(mu, log_var)
        pred = torch.lerp(mu, pred, self.kappa)
        return pred
