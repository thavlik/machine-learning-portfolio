import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from dicom_util import raw_dicom_pixels


class CQ500Dataset(data.Dataset):
    def __init__(self, dir: str):
        super(CQ500Dataset, self).__init__()
        self.files = [os.path.join(dp, f)
                      for dp, dn, fn in os.walk(os.path.expanduser(dir))
                      for f in fn
                      if f.endswith('.dcm')]

    def __getitem__(self, index):
        ds = pydicom.dcmread(self.files[index], stop_before_pixels=False)
        data = raw_dicom_pixels(ds)
        return data

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = CQ500Dataset('../../data/cq500')
    print(ds[0].shape)
    print(ds[1].shape)
