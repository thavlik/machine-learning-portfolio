from torch import Tensor, nn

from typing import List


class EncoderWrapper(nn.Module):

    def __init__(self, latent_dim: int, layers: nn.Module, mu: nn.Module,
                 var: nn.Module):
        super(EncoderWrapper, self).__init__()
        self.latent_dim = latent_dim
        self.layers = layers
        self.mu = mu
        self.var = var

    def forward(self, input: Tensor) -> List[Tensor]:
        x = self.layers(input)
        mu = self.mu(x)
        var = self.var(x)
        return [mu, var]
