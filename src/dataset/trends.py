import os
import numpy as np
import torch
import torch.utils.data as data
import h5py

class TReNDSDataset(data.Dataset):
    def __init__(self, dir: str):
        super(TReNDSDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        f = h5py.File(path, 'r')
        data = f['SM_feature']
        array = data[()]
        raise array

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = TReNDSDataset('../../data/trends-fmri/fMRI_train')
    print(ds[0].shape)
    print(ds[1].shape)
