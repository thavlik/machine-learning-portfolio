import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .inception import InceptionV3

class BaseRenderer(nn.Module):
    def __init__(self,
                 name: str,
                 enable_fid: bool,
                 fid_blocks: List[int] = [2048]) -> None:
        super(BaseRenderer, self).__init__()
        self.name = name
        self.enable_fid = enable_fid
        if enable_fid:
            for block in fid_blocks:
                if block not in InceptionV3.BLOCK_INDEX_BY_DIM:
                    raise ValueError(f'Invalid fid_block {block}, '
                                     f'valid options are {InceptionV3.BLOCK_INDEX_BY_DIM}')
            block_idx = [InceptionV3.BLOCK_INDEX_BY_DIM[i]
                         for i in fid_blocks]
            print('Loading InceptionV3')
            self.inception = InceptionV3(block_idx, use_fid_inception=True)
            print('InceptionV3 loaded')

    @abstractmethod
    def decode(self,
               world_matrix: Tensor,
               **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, world_matrix: Tensor, **kwargs) -> List[Tensor]:
        return self.decode(torch.flatten(world_matrix, start_dim=1))

    def loss_function(self,
                      recons: Tensor,
                      orig: Tensor,
                      fid_weight: float = 1.0) -> dict:
        recons_loss = F.mse_loss(recons, orig)

        loss = recons_loss

        result = {'loss': loss,
                  'Reconstruction_Loss': recons_loss}

        if self.enable_fid:
            fid_loss = self.fid(orig, recons).sum()
            result['FID_Loss'] = fid_loss
            result['loss'] += fid_loss * fid_weight

        return result

    def fid(self, a: Tensor, b: Tensor) -> Tensor:
        a = self.inception(a)
        b = self.inception(b)
        fid = [torch.mean((x - y) ** 2)
               for x, y in zip(a, b)]
        fid = torch.Tensor(fid)
        return fid
