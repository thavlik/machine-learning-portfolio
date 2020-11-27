import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet import BasicBlock, TransposeBasicBlock
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil


class ResNetVAE(BaseVAE):

    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.4,
                 width: int = 320,
                 height: int = 200,
                 channels: int = 3,
                 enable_fid: bool = False,
                 output_activation: str = 'sigmoid') -> None:
        super(ResNetVAE, self).__init__(name=name,
                                        latent_dim=latent_dim,
                                        enable_fid=enable_fid)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock(in_features, h_dim))
            modules.append(nn.MaxPool2d((2, 1)))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * width * height // 2**len(hidden_dims)
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
        self.decoder_input = nn.Linear(latent_dim, height * 4)
        modules = []
        in_features = height
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv2d(hidden_dims[-1],
                      width * height * channels // 4,
                      kernel_size=3,
                      padding=1)
        )
        self.decoder_final = nn.Sequential(
            act_options[output_activation](),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-3:] != (self.channels, self.height, self.width):
            raise ValueError('wrong input shape')
        x = self.encoder(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.height, 2, 2)
        x = self.decoder(x)
        x = self.decoder_final(x)
        x = x.view(x.shape[0], self.channels, self.height, self.width)
        return x
