import os
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
import h5py
from torch import Tensor
from typing import Optional


def compile_forrest_gump_h5py(root: str,
                              alignment: Optional[str] = None):
    if alignment is None:
        data_dir = root
        identifier = 'acq-dico'
    elif alignment == 'linear':
        data_dir = os.path.join(root, 'linear_anatomical_alignment')
        identifier = 'rec-dico7Tad2grpbold7Tad'
    elif alignment == 'nonlinear':
        data_dir = os.path.join(
            root, 'non-linear_anatomical_alignment')
        identifier = 'rec-dico7Tad2grpbold7TadNL'
    else:
        raise ValueError(f"unknown alignment value '{alignment}'")
    subjects = [f for f in os.listdir(data_dir)
                if f.startswith('sub-') and len(f) == 6]
    num_frames = 3599
    try:
        os.mkdir(os.path.join(root, 'compiled'))
    except:
        pass
    out_path = os.path.join(root, 'compiled', f'compiled_{identifier}.hdf5')
    with h5py.File(out_path, 'w') as f:
        ds = f.create_dataset(
            'default', (len(subjects), num_frames, 160, 160, 36), dtype='int16')
        for subject in subjects:
            print(f'Compiling {subject}')
            subj_no = int(subject[4:])-1
            frame_no = 0
            frames = None
            for run in range(8):
                filename = f'{subject}_ses-forrestgump_task-forrestgump_{identifier}_run-0{run+1}_bold.nii.gz'
                filename = os.path.join(
                    data_dir, subject, 'ses-forrestgump', 'func', filename)
                img = nl.image.load_img(filename)
                img = img.get_data()
                img = np.transpose(img, (3, 0, 1, 2))
                frames = img if frames is None else np.concatenate(
                    (frames, img), axis=0)
            ds[subj_no] = frames
            if frames.shape[0] != num_frames:
                raise ValueError(
                    f'{subject} has {len(frames)} frames, expected {num_frames}')


class ForrestGumpDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 alignment: Optional[str] = None):
        super(ForrestGumpDataset, self).__init__()
        self.root = root

        # sub-01_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii.gz
        # sub-01_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7Tad_run-01_bold.nii.gz
        # sub-01_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7TadNL_run-01_bold.nii.gz

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    compile_forrest_gump_h5py('/data/openneuro/ds000113-download')
    ds = ForrestGumpDataset(root='/data/openneuro/ds000113-download')
    print(ds[0])
