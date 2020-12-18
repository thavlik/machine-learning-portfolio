import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet1d import BasicBlock1d, TransposeBasicBlock1d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3
from .util import get_pooling1d, get_activation

class ResNetVAE1d(BaseVAE):
    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 num_samples: int,
                 channels: int,
                 dropout: float = 0.4,
                 pooling: str = None,
                 output_activation: str = 'tanh') -> None:
        super(ResNetVAE1d, self).__init__(name=name,
                                          latent_dim=latent_dim)
        self.num_samples = num_samples
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        if pooling is not None:
            pool_fn = get_pooling1d(pooling)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock1d(in_features, h_dim))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * num_samples
        if pooling is not None:
            in_features /= 2**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError('noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.mu = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.var = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock1d(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv1d(hidden_dims[-1],
                      num_samples * channels,
                      kernel_size=3,
                      padding=1),
            get_activation(output_activation),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-2:] != (self.channels, self.num_samples):
            raise ValueError('wrong input shape')
        x = self.encoder(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[-1], -1)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels, self.num_samples)
        return x
