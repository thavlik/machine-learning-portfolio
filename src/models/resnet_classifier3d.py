from math import ceil
import torch
from torch import nn, Tensor, Size
from torch.nn import functional as F
from typing import List
from .classifier import Classifier
from .resnet3d import BasicBlock3d
from .util import get_pooling3d


class ResNetClassifier3d(Classifier):
    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 num_classes: int,
                 dropout: float = 0.4,
                 pooling: str = None) -> None:
        super().__init__(name=name,
                         num_classes=num_classes)
        self.width = input_shape[3]
        self.height = input_shape[2]
        self.depth = input_shape[1]
        self.channels = input_shape[0]
        self.dropout = dropout
        self.hidden_dims = hidden_dims.copy()
        if pooling is not None:
            pool_fn = get_pooling3d(pooling)
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock3d(in_features, h_dim))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.layers = nn.Sequential(*modules)
        in_features = hidden_dims[-1] * self.width * self.height * self.depth
        if pooling is not None:
            in_features /= 8**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.output = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = self.layers(input)
        y = y.reshape((input.shape[0], -1))
        y = F.dropout(y, p=self.dropout)
        y = self.output(y)
        return y
