import torch
from abc import abstractmethod
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional

from torchvision.ops import (
    complete_box_iou_loss,
    distance_box_iou_loss,
    generalized_box_iou_loss,
)


class Localizer(nn.Module):
    """ Base class for a model that carries out localization.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def loss_function(self,
                      pred_params: Tensor,
                      targ_params: Tensor,
                      objective: Optional[str] = 'cbiou+dbiou') -> dict:
        # Sanity check to ensure that the parameters are valid BBs.
<<<<<<< HEAD
        assert (pred_params[:, 0] <= pred_params[:, 2]).all(), \
            "Predicted BBs are invalid."
        assert (pred_params[:, 1] <= pred_params[:, 3]).all(), \
            "Predicted BBs are invalid."
        assert (targ_params[:, 0] <= targ_params[:, 2]).all(), \
            "Target BBs are invalid."
        assert (targ_params[:, 1] <= targ_params[:, 3]).all(), \
            "Target BBs are invalid."
        # Calculate various loss metrics. Some may go unused.
        losses = dict(mse=F.mse_loss(pred_params, targ_params),
                      dbiou=distance_box_iou_loss(pred_params,
                                                  targ_params,
                                                  reduction='sum'),
                      cbiou=complete_box_iou_loss(pred_params,
                                                  targ_params,
                                                  reduction='sum'),
                      gbiou=generalized_box_iou_loss(pred_params,
                                                     targ_params,
                                                     reduction='sum'))
        # Calculate the total loss based on the objectives.
        loss = torch.zeros(1)
        for obj in objective.split('+'):
            assert obj in losses, f"Objective '{obj}' not recognized."
            loss += losses[obj]
        return {'loss': loss, **{f'{k}_Loss': v for k, v in losses.items()}}
=======
        assert (pred_params[:, 0] <= pred_params[:, 2]).all()
        assert (pred_params[:, 1] <= pred_params[:, 3]).all()
        assert (targ_params[:, 0] <= targ_params[:, 2]).all()
        assert (targ_params[:, 1] <= targ_params[:, 3]).all()

        if objective != 'cbiou+dbiou':
            raise NotImplementedError

        mse_loss = F.mse_loss(pred_params, targ_params)
        dbiou_loss = distance_box_iou_loss(pred_params,
                                           targ_params,
                                           reduction='sum')
        cbiou_loss = complete_box_iou_loss(pred_params,
                                           targ_params,
                                           reduction='sum')
        loss = dbiou_loss + cbiou_loss + mse_loss

        return {
            'loss': loss,
            'cbiou_Loss': cbiou_loss,
            'dbiou_Loss': dbiou_loss,
            'mse_Loss': mse_loss,
        }
>>>>>>> 3cc61b3ba71673d49c1cf82fdc11c882461b94f9


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
    boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea).float()
    # return the intersection over union value
    return iou
