import os
import numpy as np
import torch
import torch.utils.data as data


class DeepLesionDataset(data.Dataset):
    def __init__(self, dir: str):
        super(DeepLesionDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    ds = DeepLesionDataset('../../data/deeplesion/')
    print(ds[0].shape)
    print(ds[1].shape)
