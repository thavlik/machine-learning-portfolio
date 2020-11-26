import os
import numpy as np
import torch
import torch.utils.data as data
from skimage.io import imread

def read_hu(x):
    return imread(x).astype(np.float32) - 32768.0

class DeepLesionDataset(data.Dataset):
    def __init__(self, dir: str):
        super(DeepLesionDataset, self).__init__()
        self.files = [os.path.join(dp, f)
                      for dp, dn, fn in os.walk(os.path.expanduser(dir))
                      for f in fn
                      if f.endswith('.png')]

    def __getitem__(self, index):
        img = read_hu(self.files[index])
        return torch.Tensor(img)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = DeepLesionDataset('E:/deeplesion/')
    print(ds[0].shape)
    print(ds[1].shape)
