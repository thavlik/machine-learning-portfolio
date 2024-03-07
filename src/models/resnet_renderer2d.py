from torch import Tensor, nn
from typing import List

import numpy as np

from .renderer import BaseRenderer
from .resnet2d import TransposeBasicBlock2d
from .util import get_activation


class ResNetRenderer2d(BaseRenderer):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 enable_fid: bool = True,
                 output_activation: str = 'sigmoid') -> None:
        super().__init__(name=name, enable_fid=enable_fid)
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
        self.decoder = nn.Sequential(*modules)
        num_lods = np.min([np.log(width), np.log(height)]) / np.log(2) - 2
        activation = get_activation(output_activation)
        self.initial_output = nn.Sequential(
            TransposeBasicBlock2d(in_features, 4 * 4 * 3 // 4),
            activation,
        )
        output_layers = []
        for _ in range(num_lods):
            output_layers.append(
                nn.Sequential(
                    TransposeBasicBlock2d(3, 128),
                    TransposeBasicBlock2d(128, 3),
                    activation,
                ))
        self.output_layers = output_layers

    def decode(self,
               world_matrix: Tensor,
               lod: int = 0,
               alpha: float = 0.0,
               **kwargs) -> Tensor:
        x = self.decoder_input(world_matrix)
        x = x.view(x.shape[0], self.hidden_dims[-1], 2, 2)
        x = self.decoder(x)
        x = self.initial_output(x)
        x = x.view(x.shape[0], 3, 4, 4)
        for i, layer in enumerate(self.output_layers[:lod]):
            a = nn.Upsample(x.shape[2:] * 2)(x)
            b = layer(a)
            x = b if i < lod - 1 else a.lerp(b, alpha)
        return x
