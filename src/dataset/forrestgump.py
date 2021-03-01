import os
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
from math import floor
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


def load_scenes(path: str) -> list:
    with open(path, "r") as f:
        result = []
        for line in f:
            parts = line.strip().split(',')
            t = float(parts[0])
            name = parts[1][1:-1]
            is_day = parts[2] == "\"DAY\""
            is_exterior = parts[3] == "\"INT\""
            result.append((t, name, is_day, is_exterior))
        return result


def calc_scene_examples(scenes: list,
                        num_frames: int,
                        stride: int = 0):
    if stride != 0:
        raise NotImplementedError
    frame_dur_sec = 2.0
    scene_examples = []
    for i in range(len(scenes)):
        t0 = scenes[i+0][0]
        if i == len(scenes) - 1:
            t1 = 3599.0 * frame_dur_sec
        else:
            t1 = scenes[i+1][0]
        dur_sec = t1 - t0
        num_examples = dur_sec / frame_dur_sec - num_frames + 1
        num_examples = floor(num_examples)
        num_examples = max(0, num_examples)
        scene_examples.append(num_examples)
    return scene_examples


class ForrestGumpDataset(data.Dataset):
    FRAME_COUNTS = [451, 441, 438, 488, 462, 439, 542, 338]

    def __init__(self,
                 root: str,
                 num_frames: int = 32,
                 alignment: Optional[str] = None,
                 stride: int = 0):
        super(ForrestGumpDataset, self).__init__()
        self.root = root
        self.num_frames = num_frames
        self.scenes = load_scenes(os.path.join(
            root, "stimuli", "annotations", "scenes.csv"))
        if alignment is None:
            self.data_dir = root
            self.identifier = 'acq-dico'
        elif alignment == 'linear':
            self.data_dir = os.path.join(root, 'linear_anatomical_alignment')
            self.identifier = 'rec-dico7Tad2grpbold7Tad'
        elif alignment == 'nonlinear':
            self.data_dir = os.path.join(
                root, 'non-linear_anatomical_alignment')
            self.identifier = 'rec-dico7Tad2grpbold7TadNL'
        else:
            raise ValueError(f"unknown alignment value '{alignment}'")
        subjects = [f for f in os.listdir(self.data_dir)
                    if f.startswith('sub-') and len(f) == 6 and int(f[len('sub-'):]) <= 20]
        self.subjects = subjects
        self.scene_examples = calc_scene_examples(self.scenes,
                                                  num_frames=num_frames,
                                                  stride=stride)
        self.examples_per_subject = sum(self.scene_examples)

    def __getitem__(self, index):
        frame_dur_sec = 2.0
        example_dur = frame_dur_sec * self.num_frames
        subj_no = int(floor(index / self.examples_per_subject)) + 1
        example_no = index % self.examples_per_subject
        scene_no = 0
        offset = 0
        for scene, num_examples in zip(self.scenes, self.scene_examples):
            if offset + num_examples > example_no:
                break
            offset += num_examples
            scene_no += 1
        scene_example = example_no - offset
        scene = self.scenes[scene_no]
        start_time = scene[0] + frame_dur_sec * scene_example
        end_time = start_time + example_dur
        file_times = [c * frame_dur_sec for c in self.FRAME_COUNTS]
        start_file = None
        end_file = None
        file_start_time = 0
        for i, file_time in enumerate(file_times):
            file_end_time = file_start_time + file_time
            if file_start_time <= start_time and start_time < file_end_time:
                start_file = i+1
            if file_start_time < end_time and end_time <= file_end_time:
                end_file = i+1
            if start_file is not None and end_file is not None:
                break
            file_start_time = file_end_time
        if start_file is None:
            raise ValueError("bad start file")
        if end_file is None:
            raise ValueError("bad end file")
        if start_file != end_file:
            raise ValueError("loading scenes on boundary currently not supported")

        subj = str(subj_no)
        if len(subj) == 1:
            subj = '0' + subj
        subj = f'sub-{subj}'
        filename = f'{subj}_ses-forrestgump_task-forrestgump_{self.identifier}_run-0{start_file}_bold.nii.gz'
        filename = os.path.join(self.data_dir, subj,
                                'ses-forrestgump', 'func', filename)
        img = nl.image.load_img(filename)
        img = img.get_data()
        img = img[:, :, :, scene_example:scene_example+self.num_frames]
        img = np.transpose(img, (3, 2, 0, 1))
        return (img, 1 if scene[3] else 0)

    def __len__(self):
        return self.examples_per_subject * len(self.subjects)


if __name__ == '__main__':
    # compile_forrest_gump_h5py('/data/openneuro/ds000113-download')
    ds = ForrestGumpDataset(root='/data/openneuro/ds000113-download')
    print(ds[0][1])
    print(ds[len(ds)-1][1])
    #i = len(ds) - 1
    #while i >= 0:
    #    print(f'{i}. {ds[i][1]}')
    #    i -= 1
