import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from dicom_util import normalized_dicom_pixels


class RSNAIntracranialDataset(data.Dataset):
    def __init__(self, dir: str):
        super(RSNAIntracranialDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        data = normalized_dicom_pixels(ds)
        return data

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import matplotlib.pylab as plt
    ds = RSNAIntracranialDataset('E:/rsna-intracranial/stage_2_test')
    fig = plt.figure(figsize=(15, 10))
    columns = 5
    rows = 4
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(ds[i], cmap=plt.cm.bone)
    plt.show()
