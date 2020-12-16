import torch
from torch import nn
from torch.nn import functional as F
from .classifier import Classifier
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3
from .regression import Regression
from .util import get_pooling2d, get_activation

class ResNetRegression2d(Regression):
    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 num_output_features: int,
                 dropout: float = 0.4,
                 pooling: str = None,
                 output_activation: str = 'relu') -> None:
        super().__init__(name=name,
                         num_output_features=num_output_features)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()
        if pooling != None:
            pool_fn = get_pooling2d(pooling)
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features,
                                        h_dim))
            if pooling != None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * width * height
        if pooling != None:
            in_features /= 4**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.output = nn.Sequential(
            nn.Linear(in_features, num_output_features),
            nn.BatchNorm1d(num_output_features),
        )
        self.activation = get_activation(output_activation)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = self.layers(input)
        y = self.output(y)
        y = self.activation(y)
        return y
