import torch
from torch import Size, Tensor, nn

from math import ceil
from typing import List

from .classifier import Classifier
from .resnet2d import BasicBlock2d
from .util import get_pooling2d


class ResNetClassifier2d(Classifier):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 num_classes: int,
                 load_weights: str = None,
                 dropout: float = 0.4,
                 pooling: str = None) -> None:
        super().__init__(name=name, num_classes=num_classes)
        self.width = input_shape[2]
        self.height = input_shape[1]
        self.channels = input_shape[0]
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
        self.output = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Sigmoid(),
        )
        if load_weights is not None:
            new = self.state_dict()
            old = torch.load(load_weights)['state_dict']
            for k, v in new.items():
                ok = f'classifier.{k}'
                if ok in old:
                    new[k] = old[ok].cpu()
                    print(f'Loaded weights for layer {k}')
            self.load_state_dict(new)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = self.layers(input)
        y = self.output(y)
        return y
