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


class ResNetVAE1d(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.4,
                 length: int = 128,
                 channels: int = 1,
                 output_activation: str = 'sigmoid') -> None:
        super(ResNetVAE1d, self).__init__(latent_dim=latent_dim)
        self.length = length
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock1d(in_features, h_dim))
            modules.append(nn.MaxPool1d((2, 1)))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[-1] * length // 2**len(hidden_dims)
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
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock1d(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv1d(hidden_dims[-1],
                      length * channels,
                      kernel_size=3,
                      padding=1),
            act_options[output_activation](),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-2:] != (self.channels, self.length):
            raise ValueError('wrong input shape')
        x = self.encoder(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[-1], -1)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels, self.length)
        return x
