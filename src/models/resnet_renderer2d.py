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
from .renderer import BaseRenderer


class ResNetRenderer2d(BaseRenderer):
    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 output_activation: str = 'sigmoid') -> None:
        super().__init__(name=name)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims

        # Decoder
        self.decoder_input = nn.Linear(16, hidden_dims[0] * 4)
        modules = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            layer = TransposeBasicBlock2d(in_features, h_dim)
            modules.append(layer)
            in_features = h_dim
        self.decoder = nn.Sequential(
            *modules,
            nn.Conv2d(in_features,
                      width * height * channels // 4,
                      kernel_size=3,
                      padding=1),
        )
        self.activation = get_activation(output_activation)

    def decode(self,
               world_matrix: Tensor,
               **kwargs) -> Tensor:
        x = self.decoder_input(world_matrix)
        x = x.view(x.shape[0], self.hidden_dims[-1], 2, 2)
        x = self.decoder(x)
        x = x.view(x.shape[0], self.channels, self.height, self.width)
        x = self.activation(x)
        return x
