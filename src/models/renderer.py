import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


class BaseRenderer(nn.Module):
    def __init__(self,
                 name: str) -> None:
        super(BaseRenderer, self).__init__()
        self.name = name

    @abstractmethod
    def decode(self,
               world_matrix: Tensor,
               **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, world_matrix: Tensor, **kwargs) -> List[Tensor]:
        return self.decode(torch.flatten(world_matrix, start_dim=1))

    def loss_function(self,
                      recons: Tensor,
                      orig: Tensor,
                      beta: float = 1.0,
                      **kwargs) -> dict:
        recons_loss = F.mse_loss(recons, orig)

        loss = recons_loss

        result = {'loss': loss,
                  'Reconstruction_Loss': recons_loss}

        return result
