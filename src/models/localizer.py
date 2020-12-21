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
        mu, log_var = self.predict(input)
        pred = reparameterize(mu, log_var)
        return pred

    def loss_function(self,
                      pred_params: Tensor,
                      targ_params: Tensor,
                      objective: str = 'iou') -> dict:
        eps = 1e-7
        localization_loss = bb_intersection_over_union(pred_params, targ_params).mean() + eps
        #localization_loss = -torch.log(localization_loss)
        localization_loss = 1.0 / localization_loss
        loss = localization_loss
        return {'loss': loss,
                'Localization_Loss': localization_loss}


def bb_intersection_over_union(boxA, boxB):
    # Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])
    # compute the area of intersection rectangle
    zeros = torch.zeros(xA.shape)
    interArea = torch.max(zeros, xB - xA + 1) * torch.max(zeros, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea).float()
    # return the intersection over union value
    return iou
