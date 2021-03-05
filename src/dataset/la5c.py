import os
import numpy as np
import nilearn as nl
import nilearn.plotting
import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
from typing import List


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
                 root: str,
                 phenotypes: List[str] = ['language/bilingual'],
                 exclude_na: bool = True):
        super(LA5cDataset, self).__init__()
        self.root = root
        self.subjects = [f for f in os.listdir(root)
                         if f.startswith('sub-')]
        labels = {}
        for phenotype in phenotypes:
            parts = phenotype.split('/')
            tsv_path = os.path.join(root, 'phenotype', parts[0] + '.tsv')
            with open(tsv_path, 'r') as f:
                columns = f.readline().strip().split('\t')
                col_no = None
                for i, column in enumerate(columns):
                    if column == parts[1]:
                        col_no = i
                        break
                if col_no is None:
                    raise ValueError(
                        f'unable to find metric {parts[1]} in {tsv_path}')
                cols = [line.strip().split('\t') for line in f]
                for col in cols:
                    sub = col[0]
                    if sub not in self.subjects:
                        continue
                    label = col[col_no]
                    if label == 'n/a' and exclude_na:
                        self.subjects.remove(sub)
                        continue
                    elif label == 'N':
                        label = 0
                    elif label == 'Y':
                        label = 1
                    else:
                        try:
                            label = float(label)
                        except:
                            pass
                    if sub in labels:
                        labels[sub].append(label)
                    else:
                        labels[sub] = [label]
        self.labels = labels

    def __getitem__(self, index):
        sub = self.subjects[index]
        labels = Tensor(self.labels[sub])
        path = os.path.join(self.root, sub, 'anat', f'{sub}_T1w.nii.gz')
        img = nl.image.load_img(path)
        img = Tensor(np.asanyarray(img.dataobj))
        img = img.unsqueeze(0)
        return (img, labels)

    def __len__(self):
        return len(self.subjects)


if __name__ == '__main__':
    ds = LA5cDataset(
        root='/data/openneuro/ds000030-download')
    print(ds[0][1])
    print(ds[len(ds)-1][0].shape)
