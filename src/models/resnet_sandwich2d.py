from torch import Tensor, nn

from typing import List

from .base import reparameterize
from .classifier import Classifier
from .encoder_wrapper import EncoderWrapper
from .resnet2d import BasicBlock2d


class ResNetSandwich2d(Classifier):

    def __init__(self,
                 name: str,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 num_classes: int,
                 encoder: EncoderWrapper,
                 sandwich_layers: List[nn.Module],
                 dropout: float = 0.4,
                 pooling: str = None) -> None:
        super(ResNetSandwich2d, self).__init__(name=name)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()
        self.sandwich_layers = sandwich_layers
        self.encoder = encoder

        self.decoder = nn.Linear(encoder.latent_dim, hidden_dims[0] * 4)
        modules = []
        for h_dim, (layer, features) in zip(hidden_dims, sandwich_layers):
            modules.append(layer)
            in_features = features
            modules.append(BasicBlock2d(in_features, h_dim))
            in_features = h_dim
        self.hidden_layers = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features * 4, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encoder(input)
        z = reparameterize(mu, log_var)
        y = self.decoder(z)
        y = y.view(y.shape[0], self.hidden_dims[-1], 2, 2)
        y = self.hidden_layers(y)
        y = y.reshape(y.shape[0], -1)
        y = self.output_layer(y)
        return y

    def set_sandwich_frozen(self, frozen: bool):
        self.encoder.requires_grad = frozen
        for layer, _ in self.sandwich_layers:
            layer.requires_grad = frozen
