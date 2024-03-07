from math import ceil
from torch import Size, Tensor, nn
from typing import List

from .localizer import Localizer
from .resnet2d import BasicBlock2d
from .util import get_activation, get_pooling2d


class ResNetPGLocalizer2d(Localizer):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 dropout: float = 0.4,
                 pooling: str = None,
                 batch_normalize: bool = False,
                 output_activation: str = 'sigmoid') -> None:
        super().__init__(name=name)
        self.width = input_shape[2]
        self.height = input_shape[1]
        self.channels = input_shape[0]
        self.batch_normalize = batch_normalize
        self.hidden_dims = hidden_dims.copy()
        if pooling is not None:
            pool_fn = get_pooling2d(pooling)
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features, h_dim))
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
                    'noninteger number of features - perhaps there is too much pooling?'
                )
            in_features = int(in_features)
        self.activation = get_activation(output_activation)
        self.prediction = nn.Linear(in_features, 4)
        if batch_normalize:
            self.output = nn.Sequential(
                nn.BatchNorm1d(4),
                self.activation,
            )
        else:
            self.output = self.activation

    def forward(self, x: Tensor, lod: int = 0) -> Tensor:
        x = self.layers(x)
        x = self.prediction(x)
        x = self.output(x)
        return x
