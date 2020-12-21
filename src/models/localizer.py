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
        #pred = reparameterize(mu, log_var)
        return label, mu #pred

    def loss_function(self,
                      predictions: Tensor,
                      targets: Tensor,
                      objective: str = 'iou',
                      localization_weight: float = 1.0,
                      iou_weight: float = 1.0) -> dict:
        label_loss = torch.Tensor([0.0])
        localization_loss = torch.Tensor([0.0])
        iou_loss = torch.Tensor([0.0])
        for pred_label, pred_params, targ_label, targ_params in zip(predictions[0], predictions[1], targets[0], targets[1]):
            label_loss += (pred_label - targ_label) ** 2
            if torch.is_nonzero(targ_label):
                localization_loss += F.mse_loss(pred_params, targ_params)
                iou_loss += 1.0 - bb_intersection_over_union(pred_params, targ_params)
        loss = label_loss + localization_loss * localization_weight + iou_loss * iou_weight
        return {'loss': loss,
                'Label_Loss': label_loss,
                'IOU_Loss': iou_loss,
                'Localization_Loss': localization_loss}

def bb_intersection_over_union(boxA, boxB):
    # Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou