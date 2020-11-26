import os
import numpy as np
import torch
import torch.utils.data as data


class TReNDSDataset(data.Dataset):
    def __init__(self):
        super(TReNDSDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
