import os
import numpy as np
import torch
import torch.utils.data as data
import glob

GRASPLIFT_DATA_HEADER = 'id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10\n'

GRASPLIFT_EVENTS_HEADER = 'id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n'

NUM_CHANNELS = 32

ZIP_URL = 'https://grasplifteeg.nyc3.digitaloceanspaces.com/grasp-and-lift-eeg-detection.zip'

ZIP_SIZE_BYTES = 980887394

class GraspAndLiftEEGDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 num_samples: int = None):
        super(GraspAndLiftEEGDataset, self).__init__()
        self.num_samples = num_samples
        dir = os.path.join(root, 'train' if train else 'test')
        if not os.path.exists(dir):
            if not download:
                raise ValueError(f'{dir} does not exist')
            if not os.path.exists(root):
                os.makedirs(root)
            self.download(root)
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

        if len(bin_files) < len(csv_files):
            print(f'Number of .csv.bin files ({len(bin_files)}) '
                  f'is less than the number of .csv ({len(csv_files)}).'
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
            for file in bin_files:
                is_data = file.endswith('_data.csv.bin')
                series = file[:-len('_data.csv.bin')
                              if is_data else -len('_events.csv.bin')]
                samples = torch.load(file)
                item = examples.get(series, [None, None])
                item[0 if is_data else 1] = samples
                examples[series] = item
                if is_data and num_samples != None:
                    self.total_examples += samples.shape[1] - num_samples + 1
            self.X = []
            Y = []
            for series in sorted(examples):
                x, y = examples[series]
                self.X.append(x)
                if y != None:
                    Y.append(y)
            self.Y = Y if len(Y) > 0 else None

    def download(self, root: str):
        import requests
        import time
        import zipfile
        zip_path = os.path.join(root, 'grasp-and-lift-eeg-detection.zip')
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) != ZIP_SIZE_BYTES:
            print(f'Downloading from {ZIP_URL}')
            start = time.time()
            r = requests.get(ZIP_URL)
            if r.status_code != 200:
                raise ValueError(f'Expected status code 200, got {r.status_code}')
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            delta = time.time() - start
            print(f'Downloaded in {delta} seconds')
        print(f'Extracting {zip_path} to {root}')
        start = time.time()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        delta = time.time() - start
        print(f'Unzipped in {delta} seconds')
        os.remove(zip_path)

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
            series = file[:-len('_data.csv')
                          if is_data else -len('_events.csv')]
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
    num_samples = None
    ds = GraspAndLiftEEGDataset('E:/grasp-and-lift-eeg-detection/test',
                                num_samples=num_samples)
    mins = []
    maxs = []
    for i, (x, _) in enumerate(ds):
        mins.append(x.min())
        maxs.append(x.max())
    mins = np.min(mins)
    maxs = np.max(maxs)
    print(ds[0][0].shape)
