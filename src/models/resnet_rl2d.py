import torch
from torch import nn
from math import ceil
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from .resnet2d import BasicBlock2d
from .util import get_pooling2d
from typing import List


class ResNetRL2d(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs: int,
                 model_config: dict,
                 name: str,
                 width: int,
                 height: int,
                 channels: int,
                 hidden_dims: List[int],
                 pooling: str = None,
                 dropout: float = 0.4):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)

        if pooling is not None:
            pool_fn = get_pooling2d(pooling)

        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features, h_dim))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        self.layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),   
        )
        if pooling is not None:
            in_features /= 4**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?')
            in_features = int(in_features)
        self.output = nn.Linear(
            nn.Linear(in_features, action_space.num_actions),
            nn.Sigmoid(),
        )
        self.value_out = nn.Linear(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].to(self.device)
        x = self.layers(obs)
        model_out = self.output(x)
        self._value_out = self.value_out(x)
        return model_out, state

    def value_function(self):
        return self._value_out.flatten()
