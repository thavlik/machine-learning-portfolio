import torch
from torch import Size, Tensor, nn

from typing import List

from .augmenter import Augmenter
from .resnet1d import BasicBlock1d


class ResNetAugmenter1d(Augmenter):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 load_weights: str = None,
                 kernel_size: int = 3,
                 padding: int = 1) -> None:
        super().__init__(name=name)
        self.num_samples = input_shape[1]
        self.channels = input_shape[0]
        self.hidden_dims = hidden_dims.copy()
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(
                BasicBlock1d(in_features,
                             h_dim,
                             kernel_size=kernel_size,
                             padding=padding))
            in_features = h_dim
        modules.append(
            BasicBlock1d(in_features,
                         self.channels,
                         kernel_size=kernel_size,
                         padding=padding))
        self.layers = nn.Sequential(*modules)
        if load_weights is not None:
            new = self.state_dict()
            old = torch.load(load_weights)['state_dict']
            for k, v in new.items():
                ok = f'model.{k}'
                if ok in old:
                    new[k] = old[ok].cpu()
                    print(f'Loaded weights for layer {k}')
            self.load_state_dict(new)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
