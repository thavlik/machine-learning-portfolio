import json
import os
import numpy as np
import nilearn as nl
import nilearn.plotting
from math import ceil
import numpy as np


def convert_forrest_gump(root: str,
                         alignment: str = 'raw',
                         max_chunk_samples: int = 16):
    if alignment == 'raw':
        data_dir = root
        identifier = 'acq-raw'
    elif alignment == 'linear':
        data_dir = os.path.join(root, 'derivatives',
                                'linear_anatomical_alignment')
        identifier = 'rec-dico7Tad2grpbold7Tad'
    elif alignment == 'nonlinear':
        data_dir = os.path.join(root, 'derivatives',
                                'non-linear_anatomical_alignment')
        identifier = 'rec-dico7Tad2grpbold7TadNL'
    else:
        raise ValueError(f"unknown alignment value '{alignment}'")
    subjects = [
        f for f in os.listdir(data_dir)
        if f.startswith('sub-') and len(f) == 6 and int(f[len('sub-'):]) <= 20
    ]
    num_frames = 3599
    out_dir = os.path.join(root, 'converted', alignment)
    try:
        os.makedirs(out_dir)
    except:
        pass
    metadata = {
        'alignment': alignment,
        'max_chunk_samples': max_chunk_samples,
        'subjects': {},
    }
    for subject in subjects:
        print(f'Converting {alignment}/{subject}')
        subj_no = int(subject[4:]) - 1
        frame_no = 0
        frames = None
        for run in range(8):
            filename = f'{subject}_ses-forrestgump_task-forrestgump_{identifier}_run-0{run+1}_bold.nii.gz'
            filename = os.path.join(data_dir, subject, 'ses-forrestgump',
                                    'func', filename)
            img = nl.image.load_img(filename)
            img = img.get_data()
            img = np.transpose(img, (3, 2, 0, 1))
            frames = img if frames is None else np.concatenate(
                (frames, img), axis=0)
        if frames.shape[0] != num_frames:
            print(
                f'WARNING: {subject} has {len(frames)} frames, expected {num_frames}'
            )
        try:
            os.makedirs(os.path.join(out_dir, subject))
        except:
            pass
        num_chunks = ceil(frames.shape[0] / max_chunk_samples)
        chunks = []
        for i in range(num_chunks):
            a = max_chunk_samples * i
            b = min(a + max_chunk_samples, frames.shape[0])
            out_path = os.path.join(out_dir, subject, subject + f'_{i}')
            chunk = frames[a:b, ...]
            chunks.append(chunk.shape[0])
            np.save(out_path, chunk)
        metadata['subjects'][subject] = {
            'num_frames': frames.shape[0],
            'chunks': chunks,
        }
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(metadata))
    print('Finished converting')


if __name__ == '__main__':
    convert_forrest_gump('/data/openneuro/ds000113-download',
                         alignment='linear')
