import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .util import reparameterize


class Localizer(nn.Module):
    """ Base class for a model that carries out nonlinear localization.
    """

    def __init__(self,
                 name: str,
                 num_output_features: int) -> None:
        super().__init__()
        self.name = name
        self.num_output_features = num_output_features

    @abstractmethod
    def predict(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        label, mu, log_var = self.predict(input)
        pred = reparameterize(mu, log_var)
        return label, pred

    def loss_function(self,
                      predictions: Tensor,
                      targets: Tensor,
                      objective: str = 'mse',
                      localization_weight: float = 1.0) -> dict:
        label_loss = torch.Tensor([0.0])
        localization_loss = torch.Tensor([0.0])
        for pred_label, pred_params, targ_label, targ_params in zip(predictions[0], predictions[1], targets[0], targets[1]):
            label_loss += (pred_label - targ_label) ** 2
            if torch.is_nonzero(targ_label):
                localization_loss += F.mse_loss(pred_params, targ_params)
        loss = label_loss + localization_loss * localization_weight
        return {'loss': loss,
                'Label_Loss': label_loss,
                'Localization_Loss': localization_loss}
