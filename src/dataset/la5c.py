import os
import numpy as np
import nilearn as nl
import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
from typing import Optional

class LA5cDataset(data.Dataset):
    """ UCLA Consortium for Neuropsychiatric Phenomics LA5c Study

    https://openneuro.org/datasets/ds000030/

    Args:
        root: Path to download directory, e.g. /data/ds000030-download

    Reference:
        Gorgolewski KJ, Durnez J and Poldrack RA. Preprocessed Consortium for
        Neuropsychiatric Phenomics dataset [version 2; peer review: 2 approved].
        F1000Research 2017, 6:1262 (https://doi.org/10.12688/f1000research.11964.2)
    """

    def __init__(self,
                 root: str):
        super(LA5cDataset, self).__init__()
        
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    ds = LA5cDataset(
        root='/data/openneuro/ds000030-download')
    