import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom


class RSNAIntracranialDataset(data.Dataset):
    def __init__(self, dir: str):
        super(RSNAIntracranialDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        data = ds.PixelData
        return data

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = RSNAIntracranialDataset('../../data/rsna-intracranial/stage_2_train')
    print(ds[0].shape)
    print(ds[1].shape)
