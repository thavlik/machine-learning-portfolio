import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from .dicom_util import normalized_dicom_pixels
import boto3
import tempfile
from botocore import UNSIGNED
from botocore.config import Config


def get_inventory(bucket, root, prefix):
    filename = 'inventory.txt'
    path = os.path.join(root, prefix, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            return [line.strip() for line in f]
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, 'wb') as f:
        obj = bucket.Object(prefix + filename)
        obj.download_fileobj(f)
    with open(path, 'r') as f:
        return [line.strip() for line in f]


def load_labels_csv(path: str) -> list:
    if not os.path.exists(path):
        raise ValueError(
            f'Labels file {path} does not exist')
    labels = {}
    label_idx = [
        'epidural',
        'intraparenchymal',
        'intraventricular',
        'subarachnoid',
        'subdural',
        'any',
    ]
    with open(path, 'r') as f:
        hdr = f.readline()
        if hdr != 'ID,Label\n':
            raise ValueError(f'bad header (got "{hdr}")')
        cur_id = None
        cur_labels = None
        for i, line in enumerate(f):
            item, label = line.strip().split(',')
            label = int(label)
            if not label in [0, 1]:
                raise ValueError(f'invalid class label on line {i}')
            _, id, classname = item.split('_')
            if cur_id == None:
                cur_id = id
                cur_labels = [0] * len(label_idx)
            elif id != cur_id:
                labels[cur_id] = cur_labels
                cur_id = id
                cur_labels = [0] * len(label_idx)
            cur_labels[label_idx.index(classname)] = label
        # Don't exclude the last item
        labels[cur_id] = cur_labels
    return labels


def process_labels(files: list, path: str) -> torch.Tensor:
    labels_dict = load_labels_csv(path)
    labels = []
    for f in files:
        id = os.path.basename(f)[3:-4]
        if id not in labels_dict:
            raise ValueError(f'missing class labels for {f}')
        labels.append(labels_dict[id])
    return torch.Tensor(labels)

def notexist(path):
    return not os.path.exists(path) or os.path.getsize(path) == 0

class RSNAIntracranialDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 s3_bucket: str = 'rsna-intracranial',
                 s3_endpoint_url: str = 'https://nyc3.digitaloceanspaces.com',
                 delete_after_use: bool = False):
        super(RSNAIntracranialDataset, self).__init__()
        self.root = root
        self.train = train
        self.download = download
        self.delete_after_use = delete_after_use
        self.prefix = 'stage_2_train/' if train else 'stage_2_test/'
        dcm_path = os.path.join(root, self.prefix)
        self.dcm_path = dcm_path
        if self.download:
            s3 = boto3.resource('s3',
                                endpoint_url=s3_endpoint_url,
                                config=Config(signature_version=UNSIGNED))
            self.bucket = s3.Bucket(s3_bucket)
            self.files = get_inventory(self.bucket, root, self.prefix)
            if train:
                labels_csv_path = os.path.join(root, 'stage_2_train.csv')
                if notexist(labels_csv_path):
                    with open(labels_csv_path, 'wb') as f:
                        obj = self.bucket.Object('stage_2_train.csv')
                        obj.download_fileobj(f)
                self.labels = process_labels(
                    self.files, labels_csv_path) if train else None
            else:
                self.labels = None
        else:
            if not os.path.exists(dcm_path):
                raise ValueError(f'Directory {dcm_path} does not exist')
            self.files = [f for f in os.listdir(dcm_path)
                          if f.endswith('.dcm')]
            self.labels = process_labels(
                self.files, os.path.join(root, 'stage_2_train.csv')) if train else None

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.dcm_path, file)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            x = pydicom.dcmread(path, stop_before_pixels=False)
            x = normalized_dicom_pixels(x)
            y = self.labels[index] if self.labels != None else []
            return (x, y)
        elif not self.download:
            raise ValueError(f'File {path} does not exist')
        print(f'Downloading {file}')
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'wb') as f:
            obj = self.bucket.Object(self.prefix + file)
            obj.download_fileobj(f)
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        data = normalized_dicom_pixels(ds)
        if self.delete_after_use:
            os.remove(path)
        return (data, [])

    def __len__(self):
        return len(self.files)

    def get_labels(self, index: int) -> torch.Tensor:
        return self.labels[index]


if __name__ == '__main__':
    import matplotlib.pylab as plt
    ds = RSNAIntracranialDataset(root='E:/rsna-intracranial',
                                 download=False)
    fig = plt.figure(figsize=(15, 10))
    columns = 5
    rows = 4
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(ds[i], cmap=plt.cm.bone)
    plt.show()
