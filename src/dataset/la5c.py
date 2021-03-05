import os
import numpy as np
import nilearn as nl
import nilearn.plotting
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
        self.root = root
        self.files = [f for f in os.listdir(root)
                      if f.startswith('sub-')]

    def __getitem__(self, index):
        sub = self.files[index]
        path = os.path.join(self.root, sub, 'anat', f'{sub}_T1w.nii.gz')
        img = nl.image.load_img(path)
        img = Tensor(np.asanyarray(img.dataobj))
        img = img.unsqueeze(0)
        return (img, [])

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = LA5cDataset(
        root='/data/openneuro/ds000030-download')
    print(ds[0][0].shape)
    print(ds[len(ds)-1][0].shape)