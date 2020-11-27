import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet import BasicBlock, TransposeBasicBlock
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


class BasicVAE(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.4,
                 width: int = 320,
                 height: int = 200,
                 channels: int = 3,
                 enable_fid: bool = False,
                 output_activation: str = 'sigmoid') -> None:
        super(BasicVAE, self).__init__(latent_dim=latent_dim,
                                       enable_fid=enable_fid)
        self.width = width
        self.height = height
        self.channels = channels

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock(in_features, h_dim))
            in_features = h_dim
        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.var = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        # Decoder
        hidden_dims.reverse()
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            *modules,
        )
        act_options = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
        }
        if output_activation not in act_options:
            raise ValueError(
                f'Unknown activation function "{output_activation}"')
        out_features = width * height * channels
        self.decoder_final = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[-1], out_features),
            nn.BatchNorm2d(out_features),
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
        x = self.decoder(z)
        x = self.decoder_final(x)
        x = x.view(x.shape[0], self.channels, self.height, self.width)
        return x

    

    
