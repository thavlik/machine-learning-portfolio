import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from math import ceil
from .inception import InceptionV3
from .util import get_pooling2d, get_activation
from .encoder_wrapper import EncoderWrapper
from .upscale2d import Upscale2d


class ResNetRenderer2d(nn.Module):
    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 dropout: float = 0.4,
                 output_activation: str = 'sigmoid',
                 pooling: str = None,
                 enable_fid: bool = False,
                 fid_blocks: List[int] = [2048]) -> None:
        super().__init__()
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

        if pooling != None:
            pool_fn = get_pooling2d(pooling)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features,
                                        h_dim))
            if pooling != None:
                modules.append(pool_fn(2))
            in_features = h_dim
        layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )

        in_features = hidden_dims[-1] * width * height
        if pooling != None:
            in_features /= 4**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.encoder = EncoderWrapper(
            latent_dim=latent_dim,
            layers=layers,
            mu=nn.Sequential(
                nn.Linear(in_features, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            ),
            var=nn.Sequential(
                nn.Linear(in_features, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            ),
        )

        # Decoder
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            layer = TransposeBasicBlock2d(in_features, h_dim)
            modules.append(layer)
            in_features = h_dim
        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Sequential(
            nn.Conv2d(in_features,
                        width * height * channels // 4,
                        kernel_size=3,
                        padding=1))
        self.decoder_activation = get_activation(output_activation)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def get_encoder(self) -> List[nn.Module]:
        return self.encoder

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-3:] != (self.channels, self.height, self.width):
            raise ValueError('wrong input shape')
        return self.encoder(input)

    def decode(self,
               z: Tensor,
               lod: int = 0,
               alpha: float = 1.0,
               **kwargs) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[-1], 2, 2)
        x = self.decoder(x)
        x = self.decoder_output(x)
        x = x.view(x.shape[0], self.channels, self.height, self.width)
        x = self.decoder_activation(x)
        return x

    def loss_function(self,
                      recons: Tensor,
                      orig: Tensor,
                      *args,
                      **kwargs) -> dict:
        result = super().loss_function(
            recons, orig, *args, **kwargs)

        fid_weight = kwargs.get('fid_weight', 0.0)
        if fid_weight != 0.0:
            fid_loss = self.fid(orig, recons)
            result['FID_Loss'] = fid_loss
            result['loss'] += fid_loss * fid_weight

        return result

    def fid(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[1] == 1:
            # Convert monochrome to RGB
            a = a.repeat(1, 3, 1, 1)
            b = b.repeat(1, 3, 1, 1)
        a = self.inception(a)
        b = self.inception(b)
        fid_loss = sum(torch.mean((x - y) ** 2)
                       for x, y in zip(a, b))
        return fid_loss