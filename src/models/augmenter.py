import torch
from abc import abstractmethod
from torch import Tensor, nn
from typing import List


class Augmenter(nn.Module):

    def __init__(self, name: str) -> None:
        super(Augmenter, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def loss_function(self,
                      x: Tensor,
                      constraint: nn.Module,
                      alpha: float = 1.0) -> dict:
        t = self.forward(x)
        co = constraint(x)
        ct = constraint(t)
        td = torch.pow(ct - co, 2).mean()  # lower is better
        ud = torch.pow(x - t, 2).mean()  # higher is better
        loss = td - ud * alpha
        return {
            'loss': loss,
            'TransformedDelta': td.detach().cpu(),
            'UntransformedDelta': ud.detach().cpu()
        }
