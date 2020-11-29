import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


class Classifier(nn.Module):
    def __init__(self,
                 name: str) -> None:
        super(Classifier, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        prediction = args[0]
        target = args[1]
        loss = F.nll_loss(prediction, target)
        return dict(loss=loss)

