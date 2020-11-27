import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .inception import InceptionV3

class BaseVAE(nn.Module):
    def __init__(self,
                 name: str,
                 latent_dim: int,
                 enable_fid: bool) -> None:
        super(BaseVAE, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.enable_fid = enable_fid
        if enable_fid:
            input_dim = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[input_dim]
            self.inception = InceptionV3([block_idx])

    @abstractmethod
    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

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
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['kld_weight']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        result = {'Reconstruction_Loss': recons_loss,
                  'KLD_Loss': -kld_loss}

        loss = recons_loss + kld_weight * kld_loss

        fid_weight = kwargs['fid_weight']
        if fid_weight != 0.0:
            fid_loss = 0.0
            result['FID_Loss'] = fid_loss
            loss += fid_weight * fid_loss
            raise NotImplementedError

        result['loss'] = loss
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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
