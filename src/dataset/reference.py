import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from dicom_util import raw_dicom_pixels, normalized_dicom_pixels
from torchvision import datasets

class ReferenceDataset(data.Dataset):
    def __init__(self,
                 name: str,
                 params: dict):
        super(ReferenceDataset, self).__init__()
        self.ds = datasets[name](**params)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)
