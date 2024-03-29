import torch
from math import ceil
from torch import Size, Tensor, nn
from typing import List

from .classifier import Classifier
from .resnet1d import BasicBlock1d
from .util import get_pooling1d


class ResNetClassifier1d(Classifier):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 num_classes: int,
                 load_weights: str = None,
                 dropout: float = 0.0,
                 pooling: str = None,
                 kernel_size: int = 3,
                 padding: int = 1,
                 logits_only: bool = False) -> None:
        super().__init__(name=name, num_classes=num_classes)
        self.num_samples = input_shape[1]
        self.channels = input_shape[0]
        self.dropout = nn.Dropout(dropout)
        self.hidden_dims = hidden_dims.copy()
        self.logits_only = logits_only
        if pooling is not None:
            pool_fn = get_pooling1d(pooling)
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(
                BasicBlock1d(in_features,
                             h_dim,
                             kernel_size=kernel_size,
                             padding=padding))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.layers = nn.Sequential(*modules)
        in_features = hidden_dims[-1] * self.num_samples
        if pooling is not None:
            in_features /= 2**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?'
                )
            in_features = int(in_features)
        self.output = nn.Linear(in_features, num_classes)
        if load_weights is not None:
            new = self.state_dict()
            old = torch.load(load_weights)['state_dict']
            for k, v in new.items():
                ok = f'model.{k}'
                if ok in old:
                    new[k] = old[ok].cpu()
                    print(f'Loaded weights for layer {k}')
            self.load_state_dict(new)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        if self.logits_only:
            return x
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
