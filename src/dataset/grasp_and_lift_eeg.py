import os
import numpy as np
import torch
import torch.utils.data as data

GRASPLIFT_EEG_HEADER = 'id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10\n'


class GraspAndLiftEEGDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 num_samples: int):
        super(GraspAndLiftEEGDataset, self).__init__()
        self.num_samples = num_samples
        files = [os.path.join(dp, f)
                 for dp, dn, fn in os.walk(os.path.expanduser(dir))
                 for f in fn
                 if f.endswith('_data.csv')]
        X = []
        total_examples = 0
        for file in files:
            example = []
            with open(file, 'r') as f:
                hdr = f.readline()
                if hdr != GRASPLIFT_EEG_HEADER:
                    raise ValueError('bad header')
                for line in f:
                    channels = line.strip().split(',')[1:]
                    channels = [float(x) for x in channels]
                    channels = torch.Tensor(channels).unsqueeze(1)
                    example.append(channels)
            example = torch.cat(example, dim=1)
            X.append(example)
            total_examples += example.shape[1] - num_samples + 1
        self.X = X
        self.total_examples = total_examples

    def __getitem__(self, index):
        ofs = 0
        for example in self.X:
            sample_count = example.shape[1]
            num_examples = sample_count - self.num_samples + 1
            if index >= ofs + num_examples:
                ofs += num_examples
                continue
            i = index - ofs
            x = example[:, i:i+self.num_samples]
            return (x, [])
        raise ValueError(f'unable to seek {index}')

    def __len__(self):
        return self.total_examples


if __name__ == '__main__':
    ds = GraspAndLiftEEGDataset('E:/grasp-and-lift-eeg-detection/test',
                                num_samples=4096)
    print(ds[0][0].shape)
