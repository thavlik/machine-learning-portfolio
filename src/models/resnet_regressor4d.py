from torch import Size, Tensor, nn
from typing import List

from .resnet4d import BasicBlock4d
from .util import get_activation


class ResNetRegressor4d(nn.Module):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 input_shape: Size,
                 output_features: int,
                 dropout: float = 0.3,
                 batch_normalize: bool = False,
                 output_activation: str = 'sigmoid') -> None:
        super().__init__(name=name)
        self.width = input_shape[4]
        self.height = input_shape[3]
        self.depth = input_shape[2]
        self.frames = input_shape[1]
        self.channels = input_shape[0]
        self.batch_normalize = batch_normalize
        self.hidden_dims = hidden_dims.copy()
        modules = []
        in_features = self.channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock4d(in_features, h_dim))
            in_features = h_dim
        self.layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )
        in_features = hidden_dims[
            -1] * self.width * self.height * self.depth * self.frames
        self.activation = get_activation(output_activation)
        self.prediction = nn.Linear(in_features, output_features)
        if batch_normalize:
            self.output = nn.Sequential(
                nn.BatchNorm1d(output_features),
                self.activation,
            )
        else:
            self.output = self.activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.prediction(x)
        x = self.output(x)
        return x
