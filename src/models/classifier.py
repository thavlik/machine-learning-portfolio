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
                      prediction: Tensor,
                      target: Tensor,
                      loss_fn: str = 'mse') -> dict:
        result = {}
        if loss_fn == 'nll':
            result['loss'] = F.nll_loss(prediction, target)
            result['accuracy'] = torch.sum(
                target == prediction.argmax(1)).float() / target.shape[0]
        elif loss_fn == 'mse':
            result['loss'] = F.mse_loss(prediction, target)

            # Calculate overall average accuracy. Every class
            # prediction for every example in the batch is an
            # opportunity for the network to, on average,
            # guess correctly.
            # TODO: figure out a better way to measure accuracy
            # for multiclass loss
            correct = torch.round(prediction).int() == target
            possible_correct = torch.prod(torch.Tensor(list(target.shape)))
            accuracy = correct.int().sum() / possible_correct
            result['accuracy'] = accuracy
        else:
            raise NotImplementedError
        return result
