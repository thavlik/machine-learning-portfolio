import os
import numpy as np
import torch
import torch.utils.data as data
import glob

GRASPLIFT_EEG_HEADER = 'id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10\n'

NUM_CHANNELS = 32


class GraspAndLiftEEGDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 num_samples: int):
        super(GraspAndLiftEEGDataset, self).__init__()
        self.num_samples = num_samples
        csv_suffix = '_data.csv'
        bin_suffix = '_data.csv.bin'
        csv_files = [os.path.join(dp, f)
                     for dp, dn, fn in os.walk(os.path.expanduser(dir))
                     for f in fn
                     if f.endswith(csv_suffix)]
        bin_files = [os.path.join(dp, f)
                     for dp, dn, fn in os.walk(os.path.expanduser(dir))
                     for f in fn
                     if f.endswith(bin_suffix)]

        should_compile = False

        if len(bin_files) != len(csv_files):
            print(f'Number of .csv.bin files ({len(bin_files)}) '
                  f'does not match number of .csv ({len(csv_files)}).'
                  ' Compiling binary representation...')
            should_compile = True

        if should_compile:
            self.X, self.total_examples = self.compile_bin(csv_files)
        else:
            X = []
            total_examples = 0
            for file in csv_files:
                samples = torch.load(file + '.bin')
                X.append(samples)
                total_examples += samples.shape[1] - num_samples + 1
            self.X = X
            self.total_examples = total_examples

    def compile_bin(self,
                    csv_files: list,
                    normalize: bool = False):
        examples = []
        total_examples = 0
        high = None
        for i, file in enumerate(csv_files):
            samples = []
            with open(file, 'r') as f:
                hdr = f.readline()
                if hdr != GRASPLIFT_EEG_HEADER:
                    raise ValueError('bad header')
                for line in f:
                    channels = line.strip().split(',')[1:]
                    channels = [float(x) for x in channels]
                    channels = torch.Tensor(channels).unsqueeze(1)
                    samples.append(channels)
            samples = torch.cat(samples, dim=1)
            total_examples += samples.shape[1] - self.num_samples + 1
            examples.append(samples)
            if normalize:
                h = samples.max()
                if high == None or h > high:
                    high = h
            else:
                torch.save(samples, file + '.bin')
            print(f'Processed {i+1}/{len(csv_files)} {file}')
        if normalize:
            for i, file in enumerate(csv_files):
                examples[i] /= 0.5 * high
                examples[i] -= 1.0
                out_path = file + '.bin'
                torch.save(samples, out_path)
        return examples, total_examples

    def __getitem__(self, index):
        ofs = 0
        for samples in self.X:
            num_examples = samples.shape[1] - self.num_samples + 1
            if index >= ofs + num_examples:
                ofs += num_examples
                continue
            i = index - ofs
            return (samples[:, i:i+self.num_samples], [])
        raise ValueError(f'unable to seek {index}')

    def __len__(self):
        return self.total_examples


if __name__ == '__main__':
    num_samples = 4096
    ds = GraspAndLiftEEGDataset('E:/grasp-and-lift-eeg-detection/test',
                                num_samples=num_samples)
    for i, (x, _) in enumerate(ds):
        assert x.shape == (NUM_CHANNELS, num_samples)
    print(ds[0][0].shape)
