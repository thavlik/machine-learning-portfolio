import json
import os
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
from math import ceil, floor
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
import h5py
from torch import Tensor
from typing import Optional


def load_scenes(path: str) -> list:
    with open(path, "r") as f:
        result = []
        for line in f:
            parts = line.strip().split(',')
            t = float(parts[0])
            name = parts[1][1:-1]
            is_day = 1 if parts[2] == "\"DAY\"" else 0
            is_exterior = 1 if parts[3] == "\"INT\"" else 0
            result.append((t, name, is_day, is_exterior))
        return result


def soft_label(scenes: list, t0: float, t1: float) -> float:
    """ Calculate soft labels, defined as the weighted average
    of labels across all involved scenes.
    """
    if t1 < scenes[0][0]:
        return (0, 0)
    values = []
    weights = []
    for i, scene in enumerate(scenes):
        if scene[0] > t1:
            break
        s0 = scene[0]
        if i == len(scenes)-1:
            s1 = 7198.0
        else:
            s1 = scenes[i+1][0]
        # t0 within s, t1 within s, s completely within t, t completely within s
        if (s0 <= t0 and t0 <= s1) or (s0 <= t1 and t1 <= s1) or (t0 <= s0 and s1 <= t1) or (s0 <= t0 and t1 <= s1):
            i0 = max(s0, t0)
            i1 = min(s1, t1)
            dur = i1 - i0
            weights.append(dur)
            labels = scene[2:]
            values.append(labels)
    labels = np.average(values, axis=0, weights=weights)
    return tuple(labels)


def convert_labels(scenes: list,
                   offset: float,
                   frame_dur: float) -> list:
    labels = []
    for i in range(3599):
        t0 = i * frame_dur - offset
        t1 = t0 + frame_dur
        label = soft_label(scenes, t0, t1)
        labels.append(label)
    return Tensor(labels)


def load_metadata(path: str):
    with open(path, "r") as f:
        return json.loads(f.read())


class ForrestGumpDataset(data.Dataset):
    """ Forrest Gump fMRI dataset from OpenNeuro. BOLD imagery acquired
    at 0.5 Hz for the entire duration of the film are associated with
    class labels assigned to each scene.

    https://openneuro.org/datasets/ds000113/

    Args:
        root: Path to download directory, e.g. /data/ds000113-download

        offset: Number of seconds to delay between stimulation and label
            assignment. Activity of interest may only be visible after a
            short delay. Adjust this value so the apparent activity
            correlates optimally with the stimulation. Note: fMRI by itself
            has a delay on the order of seconds, so further offset may not
            be necessary.

        alignment: Optional alignment transformation geometry. Valid values
            are "raw", "linear", and "nonlinear".
        
        squeeze: Option to squeeze the data tensor, so as to make a single
            example 3D instead of 4D.

    Labels:
        0: The scene takes place indoors
        1: The scene takes place outdoors

    Reference:
        Hanke, M., Baumgartner, F., Ibe, P. et al. A high-resolution 7-Tesla
        fMRI dataset from complex natural stimulation with an audio movie.
        Sci Data 1, 140003 (2014). https://doi.org/10.1038/sdata.2014.3

        C.H. Liao, K.J. Worsley, J.-B. Poline, J.A.D. Aston, G.H. Duncan,
        A.C. Evans. Estimating the Delay of the fMRI Response. NeuroImage,
        Volume 16, Issue 3, Part A. 2002. Pages 593-606. ISSN 1053-8119.
        https://doi.org/10.1006/nimg.2002.1096.
        https://www.math.mcgill.ca/keith/delay/delay.pdf.
    """

    def __init__(self,
                 root: str,
                 offset: float = 0.0,
                 alignment: Optional[str] = 'raw',
                 squeeze: Optional[bool] = False):
        super(ForrestGumpDataset, self).__init__()
        self.root = root
        self.squeeze = squeeze
        self.scenes = load_scenes(os.path.join(
            root, "stimuli", "annotations", "scenes.csv"))
        self.labels = convert_labels(self.scenes, offset=offset, frame_dur=2.0)
        if alignment == 'raw':
            self.data_dir = os.path.join(root, 'converted', 'raw')
        elif alignment == 'linear':
            self.data_dir = os.path.join(root, 'converted', 'linear')
        elif alignment == 'nonlinear':
            self.data_dir = os.path.join(root, 'converted', 'nonlinear')
        else:
            raise ValueError(f"unknown alignment value '{alignment}'")
        metadata = load_metadata(os.path.join(self.data_dir, 'metadata.json'))
        self.subjects = metadata['subjects']
        self.subject_keys = sorted(self.subjects.keys())
        num_examples = 0
        for key in self.subject_keys:
            subject = self.subjects[key]
            num_examples += subject['num_frames']
        self.num_examples = num_examples

    def __getitem__(self, index):
        sub_no = 0
        offset = 0
        for key in self.subject_keys:
            subject = self.subjects[key]
            num_frames = subject['num_frames']
            if offset + num_frames > index:
                break
            offset += num_frames
            sub_no += 1
        index -= offset
        labels = self.labels[index]
        chunk_no = 0
        offset = 0
        for chunk in subject['chunks']:
            if offset + chunk > index:
                break
            offset += chunk
            chunk_no += 1
        index -= offset
        chunk = np.load(os.path.join(self.data_dir, key, f'{key}_{chunk_no}.npy'))
        img = chunk[index:index+1, ...]
        img = Tensor(img)
        if self.squeeze:
            img = img.squeeze()
        return (img, labels)

    def __len__(self):
        return self.num_examples


if __name__ == '__main__':
    ds = ForrestGumpDataset(root='/data/openneuro/ds000113-download',
                            alignment='nonlinear',
                            squeeze=True)
    print(ds[0])
    print(ds[1])
