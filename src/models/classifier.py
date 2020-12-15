import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple


class Classifier(nn.Module):
    def __init__(self,
                 name: str,
                 num_classes: int) -> None:
        super(Classifier, self).__init__()
        self.name = name
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def loss_function(self,
                      prediction: Tensor,
                      target: Tensor,
                      objective: str = 'mse',
                      baseline_accuracy: float = None) -> dict:
        result = {}
        if objective == 'nll':
            result['loss'] = F.nll_loss(prediction, target)
            result['accuracy'] = torch.sum(
                target == prediction.argmax(1)).float() / target.shape[0]
        elif objective == 'mse':
            result['loss'] = F.mse_loss(prediction, target)
            # Calculate overall average accuracy. Every class
            # prediction for every example in the batch is an
            # opportunity for the network to, on average,
            # guess correctly.
            # TODO: figure out a better way to measure accuracy
            # for multiclass loss
            correct = torch.round(prediction).int() == target
            possible_correct = torch.prod(torch.Tensor(list(target.shape)))
            accuracy = correct.float().sum() / possible_correct
            result['accuracy'] = accuracy
        else:
            raise ValueError(f'Objective "{objective}" not implemented')

        any_acc = torch.sum(torch.round(prediction), dim=1).clamp(0, 1).int() == torch.sum(target, dim=1).clamp(0, 1).int()
        any_acc = any_acc.float().mean()
        result['accuracy/any'] = any_acc

        avg_acc = torch.round(prediction).int() == target
        avg_acc = avg_acc.float().mean()
        result['accuracy/avg'] = avg_acc

        if baseline_accuracy is not None:
            result['rel_acc/avg'] = (avg_acc - baseline_accuracy) / (1.0 - baseline_accuracy)
            result['rel_acc/any'] = (any_acc - baseline_accuracy) / (1.0 - baseline_accuracy)

        num_classes = target.shape[1]

        for i in range(num_classes):
            acc = torch.round(prediction[:, i]).int() == target[:, i]
            acc = acc.float().mean()
            result[f'accuracy/class_{i}'] = acc
            if baseline_accuracy is not None:
                result[f'rel_acc/class_{i}'] = (acc - baseline_accuracy) / (1.0 - baseline_accuracy)
        
        return result
