import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet3d import BasicBlock3d, TransposeBasicBlock3d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3
from .util import get_pooling3d, get_activation


class ResNetVAE3d(BaseVAE):
    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 depth: int,
                 channels: int,
                 dropout: float = 0.4,
                 enable_fid: bool = False,  # per-frame FID, for video
                 pooling: str = None,
                 output_activation: str = 'sigmoid') -> None:
        super(ResNetVAE3d, self).__init__(name=name,
                                          latent_dim=latent_dim)
        self.width = width
        self.height = height
        self.depth = depth
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        if pooling != None:
            pool_fn = get_pooling3d(pooling)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock3d(in_features, h_dim))
            if pooling != None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * width * height * depth
        if pooling != None:
            in_features /= 8**len(hidden_dims)
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
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 8)
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock3d(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv3d(hidden_dims[-1],
                      width * height * depth * channels // 8,
                      kernel_size=3,
                      padding=1),
            get_activation(output_activation),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-4:] != (self.channels, self.depth, self.height, self.width):
            raise ValueError('wrong input shape')
        x = self.encoder(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[0], 2, 2, 2)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels,
                   self.depth, self.height, self.width)
        return x
