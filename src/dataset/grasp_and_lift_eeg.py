import os
import numpy as np
import torch
import torch.utils.data as data
import glob

GRASPLIFT_DATA_HEADER = 'id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10\n'

GRASPLIFT_EVENTS_HEADER = 'id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n'

NUM_CHANNELS = 32


class GraspAndLiftEEGDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 num_samples: int = None):
        super(GraspAndLiftEEGDataset, self).__init__()
        self.num_samples = num_samples
        csv_suffix = '.csv'
        bin_suffix = '.csv.bin'
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
            self.X, self.Y = self.compile_bin(csv_files)
            if num_samples != None:
                # Divide each example up into windows
                self.total_examples = 0
                for x in self.X:
                    self.total_examples += x.shape[1] - num_samples + 1
        else:
            examples = {}
            self.total_examples = 0
            for file in csv_files:
                is_data = file.endswith('_data.csv')
                series = file[:-len('_data.csv') if is_data else -len('_events.csv')]
                samples = torch.load(file + '.bin')
                item = examples.get(series, [None, None])
                item[0 if is_data else 1] = samples
                examples[series] = item
                if is_data and num_samples != None:
                    self.total_examples += samples.shape[1] - num_samples + 1
            self.X = []
            self.Y = []
            for series in sorted(examples):
                x, y = examples[series]
                self.X.append(x)
                if y != None:
                    self.Y.append(y)
            if len(self.Y) == 0:
                self.Y = None

    def compile_bin(self,
                    csv_files: list,
                    normalize: bool = False):
        examples = {}
        high = None
        for i, file in enumerate(csv_files):
            is_data = file.endswith('_data.csv')
            samples = []
            with open(file, 'r') as f:
                hdr = f.readline()
                expected_hdr = GRASPLIFT_DATA_HEADER if is_data else GRASPLIFT_EVENTS_HEADER 
                if hdr != expected_hdr:
                    raise ValueError('bad header')
                for line in f:
                    channels = line.strip().split(',')[1:]
                    if is_data:
                        # Data is converted to float eventually anyway
                        channels = [float(x) for x in channels]
                    else:
                        # Labels are integer format
                        channels = [int(x) for x in channels]
                    channels = torch.Tensor(channels).unsqueeze(1)
                    samples.append(channels)
            samples = torch.cat(samples, dim=1)
            series = file[:-len('_data.csv') if is_data else -len('_events.csv')]
            item = examples.get(series, [None, None])
            item[0 if is_data else 1] = samples
            examples[series] = item
            if normalize:
                h = samples.max()
                if high == None or h > high:
                    high = h
            else:
                # Go ahead and save
                torch.save(samples, file + '.bin')
            print(f'Processed {i+1}/{len(csv_files)} {file}')
        X = []
        Y = []
        for series in sorted(examples):
            x, y = examples[series]
            x /= 0.5 * high
            x -= 1.0
            torch.save(samples, series + '_data.csv.bin')
            X.append(x)
            if y != None:
                torch.save(samples, series + '_events.csv.bin')
                Y.append(y)
        return X, Y if len(Y) > 0 else None

    def __getitem__(self, index):
        if self.num_samples == None:
            # Return the entire example (e.g. reinforcement learning)
            return (self.X[index], self.Y[index] if self.Y != None else [])
        # Find the example and offset for the index
        ofs = 0
        for i, x in enumerate(self.X):
            num_examples = x.shape[1] - self.num_samples + 1
            if index >= ofs + num_examples:
                ofs += num_examples
                continue
            j = index - ofs
            x = x[:, j:j+self.num_samples]
            y = self.Y[i][:, j:j+self.num_samples] if self.Y != None else []
            return x, y
        raise ValueError(f'unable to seek {index}')

    def __len__(self):
        if self.num_samples == None:
            # No windowing - each example is full length
            return len(self.X)
        # Use precalculated dataset length
        return self.total_examples


if __name__ == '__main__':
    num_samples = 4096
    ds = GraspAndLiftEEGDataset('E:/grasp-and-lift-eeg-detection/test',
                                num_samples=num_samples)
    for i, (x, _) in enumerate(ds):
        assert x.shape == (NUM_CHANNELS, num_samples)
    print(ds[0][0].shape)
