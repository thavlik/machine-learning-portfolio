import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import sqrt, ceil
from .inception import InceptionV3


class ResNetVAE2d(BaseVAE):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.4,
                 width: int = 320,
                 height: int = 200,
                 channels: int = 3,
                 enable_fid: bool = False,
                 output_activation: str = 'sigmoid',
                 fid_blocks: List[int] = [2048]) -> None:
        super(ResNetVAE2d, self).__init__(latent_dim=latent_dim)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        self.enable_fid = enable_fid
        if enable_fid:
            for block in fid_blocks:
                if block not in InceptionV3.BLOCK_INDEX_BY_DIM:
                    raise ValueError(f'Invalid fid_block {block}, '
                                     f'valid options are {InceptionV3.BLOCK_INDEX_BY_DIM}')
            block_idx = [InceptionV3.BLOCK_INDEX_BY_DIM[i]
                         for i in fid_blocks]
            self.inception = InceptionV3(block_idx)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features, h_dim))
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
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(TransposeBasicBlock2d(in_features, h_dim))
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv2d(hidden_dims[-1],
                      width * height * channels // 4,
                      kernel_size=3,
                      padding=1),
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
        x = x.view(x.shape[0], self.hidden_dims[0], 2, 2)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels, self.height, self.width)
        return x

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        result = super(ResNetVAE2d, self).loss_function(*args, **kwargs)

        recons = args[0]
        orig = args[1]

        fid_weight = kwargs['fid_weight']
        if fid_weight != 0.0:
            fid_loss = self.fid(orig, recons)
            result['FID_Loss'] = fid_loss
            result['loss'] += fid_loss * fid_weight

        return result

    def fid(self, a: Tensor, b: Tensor) -> Tensor:
        a = self.inception(a)
        b = self.inception(b)
        d = torch.sum([torch.mean((x - y) ** 2)
                       for x, y in zip(a, b)])
        return d
