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
from .util import get_pooling2d, get_activation


class ResNetLocalizer2d(Localizer):
    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 dropout: float = 0.4,
                 pooling: str = None,
                 output_activation: str = 'sigmoid',
                 logvar_scale: float = 0.05) -> None:
        super().__init__(name=name,
                         logvar_scale=logvar_scale)
        self.width = input_shape[2]
        self.height = input_shape[1]
        self.channels = input_shape[0]
        self.hidden_dims = hidden_dims.copy()
        if pooling is not None:
            pool_fn = get_pooling2d(pooling)
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features,
                                        h_dim))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * self.width * self.height
        if pooling is not None:
            in_features /= 4**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.mu = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.BatchNorm1d(4),
        )
        self.log_var = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.BatchNorm1d(4),
        )
        self.activation = get_activation(output_activation)

    def predict(self, input: Tensor) -> List[Tensor]:
        y = self.layers(input)
        mu = self.activation(self.mu(y))
        log_var = self.activation(self.log_var(y))
        return [mu, log_var]
