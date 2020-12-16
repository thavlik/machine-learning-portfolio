import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


class Regression(nn.Module):
    """ Base class for a model that carries out nonlinear regression.
    """
    def __init__(self,
                 name: str,
                 num_output_features: int) -> None:
        super().__init__()
        self.name = name
        self.num_output_features = num_output_features

    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def loss_function(self,
                      prediction: Tensor,
                      target: Tensor) -> dict:
        loss = F.mse_loss(prediction, target)
        return {'loss': loss}
