import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List


class Augmenter(nn.Module):
    def __init__(self,
                 name: str) -> None:
        super(Augmenter, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        raise NotImplementedError

    def loss_function(self,
                      x: Tensor,
                      constraint: nn.Module) -> dict:
        co = constraint(x)
        t = self.forward(x)
        ct = constraint(t)
        ud = torch.pow(x - t, 2) # higher is better
        td = torch.pow(ct - co, 2) # lower is better
        return td - ud
