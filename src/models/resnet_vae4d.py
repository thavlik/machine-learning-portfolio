import torch
from torch import nn
from torch.nn import functional as F
from .conv4d import Conv4d
from .base import BaseVAE
from .util import get_pooling4d, get_activation
from .resnet4d import BasicBlock4d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil


class ResNetVAE4d(BaseVAE):
    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.4,
                 width: int = 100,
                 height: int = 100,
                 depth: int = 100,
                 frames: int = 64,
                 channels: int = 1,
                 pooling: str = None,
                 output_activation: str = 'sigmoid') -> None:
        super(ResNetVAE4d, self).__init__(name=name,
                                          latent_dim=latent_dim)
        self.width = width
        self.height = height
        self.depth = depth
        self.frames = frames
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        if pooling != None:
            pool_fn = get_pooling4d(pooling)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock4d(in_features, h_dim))
            if pooling != None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * width * height * depth * frames
        if pooling != None:
            in_features /= 16**len(hidden_dims)
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

        # Decoder
        act_options = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
        }
        if output_activation not in act_options:
            raise ValueError(
                f'Unknown activation function "{output_activation}"')

        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 16)
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(BasicBlock4d(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            Conv4d(hidden_dims[-1],
                   width * height * depth * frames * channels // 16,
                   kernel_size=3,
                   padding=1),
            get_activation(output_activation),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-5:] != (self.channels, self.frames, self.depth, self.height, self.width):
            raise ValueError('wrong input shape')
        x = self.encoder(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[0], 2, 2, 2, 2)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels,
                   self.frames, self.depth,
                   self.height, self.width)
        return x
