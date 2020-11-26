import os
import numpy as np
import torch
import torch.utils.data as data
import h5py


class TReNDSfMRIDataset(data.Dataset):
    def __init__(self, dir: str):
        super(TReNDSfMRIDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        # reference: https://www.kaggle.com/mks2192/reading-matlab-mat-files-and-eda
        f = h5py.File(path, 'r')
        data = f['SM_feature']
        array = data[()]
        return array

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = TReNDSfMRIDataset('../../data/trends-fmri/fMRI_train')
    print(ds[0].shape)
    print(ds[1].shape)
