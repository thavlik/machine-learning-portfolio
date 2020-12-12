import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class BaseVAE(nn.Module):
    def __init__(self,
                 name: str,
                 latent_dim: int) -> None:
        super(BaseVAE, self).__init__()
        self.name = name
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: Tensor, **kwargs) -> Any:
        raise NotImplementedError

    def get_sandwich_layers(self) -> List[nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def get_encoder(self) -> List[nn.Module]:
        raise NotImplementedError

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = reparameterize(mu, log_var)
        return [self.decode(z, **kwargs), input, mu, log_var, z]

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def loss_function(self,
                      recons: Tensor,
                      input: Tensor,
                      mu: Tensor,
                      log_var: Tensor,
                      z: Tensor,
                      objective: str = 'default',
                      beta: float = 1.0,
                      gamma: float = 1.0,
                      target_capacity: float = 25.0) -> dict:
        recons_loss = F.mse_loss(recons, input)

        result = {'loss': recons_loss,
                  'Reconstruction_Loss': recons_loss}

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        result['KLD_Loss'] = kld_loss

        if objective == 'default':
            # O.G. beta loss term applied directly to KLD
            result['loss'] += beta * kld_loss
        elif objective == 'controlled_capacity':
            # Use controlled capacity increase from
            # https://arxiv.org/pdf/1804.03599.pdf
            capacity_loss = torch.abs(kld_loss - target_capacity)
            result['Capacity_Loss'] = capacity_loss
            result['loss'] += gamma * capacity_loss
        else:
            raise ValueError(f'unknown objective "{objective}"')

        return result
