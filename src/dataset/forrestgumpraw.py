import os
import torch
import torch.utils.data as data
from math import floor
from torch import Tensor
from typing import Optional

import nilearn as nl
import nilearn.plotting
import numpy as np


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


def calc_scene_examples(scenes: list, num_frames: int):
    frame_dur_sec = 2.0
    scene_examples = []
    for i in range(len(scenes)):
        t0 = scenes[i + 0][0]
        if i == len(scenes) - 1:
            t1 = 3599.0 * frame_dur_sec
        else:
            t1 = scenes[i + 1][0]
        dur_sec = t1 - t0
        num_examples = dur_sec / frame_dur_sec - num_frames + 1
        num_examples = floor(num_examples)
        num_examples = max(0, num_examples)
        scene_examples.append(num_examples)
    return scene_examples


class ForrestGumpRawDataset(data.Dataset):
    """ Forrest Gump fMRI dataset from OpenNeuro. BOLD imagery acquired
    at 0.5 Hz for the entire duration of the film are associated with
    class labels assigned to each scene.

    https://openneuro.org/datasets/ds000113/

    Args:
        root: Path to download directory, e.g. /data/ds000113-download

        num_frames: Number of BOLD frames in an example. Note: each frame
            is 2.0 seconds in duration.

        offset_frames: Number of BOLD frames to delay between stimulation
            and label assignment. Activity of interest may only be visible
            after a short delay. Adjust this value so the apparent activity
            correlates optimally with the stimulation. Note: fMRI by itself
            has a delay on the order of seconds, so further offset may not
            be necessary.

        alignment: Optional alignment transformation geometry. Valid values
            are "linear" and "nonlinear".

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

    FILE_DURATIONS = [902, 882, 876, 976, 924, 878, 1084, 676]

    def __init__(self,
                 root: str,
                 num_frames: int = 8,
                 offset_frames: int = 0,
                 alignment: Optional[str] = None):
        super(ForrestGumpRawDataset, self).__init__()
        if offset_frames != 0:
            raise NotImplementedError
        self.root = root
        self.num_frames = num_frames
        self.scenes = load_scenes(
            os.path.join(root, "stimuli", "annotations", "scenes.csv"))
        if alignment is None:
            self.data_dir = root
            self.identifier = 'acq-raw'
        elif alignment == 'linear':
            self.data_dir = os.path.join(root, 'derivatives',
                                         'linear_anatomical_alignment')
            self.identifier = 'rec-dico7Tad2grpbold7Tad'
        elif alignment == 'nonlinear':
            self.data_dir = os.path.join(root, 'derivatives',
                                         'non-linear_anatomical_alignment')
            self.identifier = 'rec-dico7Tad2grpbold7TadNL'
        else:
            raise ValueError(f"unknown alignment value '{alignment}'")
        subjects = [
            f for f in os.listdir(self.data_dir) if f.startswith('sub-')
            and len(f) == 6 and int(f[len('sub-'):]) <= 20
        ]
        self.subjects = subjects
        self.scene_examples = calc_scene_examples(self.scenes,
                                                  num_frames=num_frames)
        self.examples_per_subject = sum(self.scene_examples)

    def __getitem__(self, index):
        frame_dur_sec = 2.0
        example_dur = frame_dur_sec * self.num_frames
        subj_no = int(floor(index / self.examples_per_subject))
        subj = self.subjects[subj_no]
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
        label = 1 if scene[3] else 0
        start_time = scene[0] + frame_dur_sec * scene_example
        end_time = start_time + example_dur
        start_file = None
        end_file = None
        file_start_time = 0
        for i, file_time in enumerate(self.FILE_DURATIONS):
            file_end_time = file_start_time + file_time
            if file_start_time <= start_time and start_time < file_end_time:
                start_file = i + 1
            if file_start_time < end_time and end_time <= file_end_time:
                end_file = i + 1
            if start_file is not None and end_file is not None:
                break
            file_start_time = file_end_time
        if start_file is None:
            raise ValueError("unable to seek start file")
        if end_file is None:
            raise ValueError("unable to seek end file")
        if start_file != end_file:
            start_img_time = sum(self.FILE_DURATIONS[:start_file])
            start_dur = start_img_time - start_time
            start_frames = int(start_dur / frame_dur_sec)
            remainder = int(self.num_frames - start_frames)

            start_path = f'{subj}_ses-forrestgump_task-forrestgump_{self.identifier}_run-0{start_file}_bold.nii.gz'
            start_path = os.path.join(self.data_dir, subj, 'ses-forrestgump',
                                      'func', start_path)
            start_img = nl.image.load_img(start_path)
            start_img = start_img.get_data()
            start_img = start_img[:, :, :, start_img.shape[-1] - start_frames:]
            start_img = np.transpose(start_img, (3, 2, 0, 1))

            end_path = f'{subj}_ses-forrestgump_task-forrestgump_{self.identifier}_run-0{end_file}_bold.nii.gz'
            end_path = os.path.join(self.data_dir, subj, 'ses-forrestgump',
                                    'func', end_path)
            end_img = nl.image.load_img(end_path)
            end_img = end_img.get_data()
            end_img = end_img[:, :, :, :remainder]
            end_img = np.transpose(end_img, (3, 2, 0, 1))

            img = np.concatenate([start_img, end_img], axis=0)
        else:
            filename = f'{subj}_ses-forrestgump_task-forrestgump_{self.identifier}_run-0{start_file}_bold.nii.gz'
            filename = os.path.join(self.data_dir, subj, 'ses-forrestgump',
                                    'func', filename)
            img = nl.image.load_img(filename)
            img = img.get_data()
            img = img[:, :, :, scene_example:scene_example + self.num_frames]
            img = np.transpose(img, (3, 2, 0, 1))
        return (img, label)

    def __len__(self):
        return self.examples_per_subject * len(self.subjects)


if __name__ == '__main__':
    ds = ForrestGumpRawDataset(root='/data/openneuro/ds000113-download',
                               alignment='linear')
    #print(f'last: {ds[len(ds)-1][1]}')
    for i in range(len(ds.subjects)):
        print(
            f'subject {i+1}: {ds[ds.examples_per_subject * i][1]}, {ds[ds.examples_per_subject * (i+1) - 1][1]}'
        )
    # for i, x in enumerate(ds):
    #    print(f'{i}. {x[1]}')
    #i = len(ds) - 1
    # while i >= 0:
    #    print(f'{i}. {ds[i][1]}')
    #    i -= 1
