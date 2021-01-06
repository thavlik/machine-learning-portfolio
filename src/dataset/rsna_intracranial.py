import os
import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
import pydicom
from .dicom_util import normalized_dicom_pixels
import boto3
import tempfile
from botocore import UNSIGNED
from botocore.config import Config
import subprocess
import gzip
from torch.nn import functional as F

def get_inventory(bucket,
                  root: str,
                  prefix: str,
                  download: bool,
                  use_gzip: bool):
    filename = 'inventory.txt'
    if use_gzip:
        filename += '.gz'
    path = os.path.join(root, prefix, filename)
    if not_exist(path):
        if not download:
            raise ValueError(f'with download == False, {path} not found')
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        with open(path, 'wb') as f:
            obj = bucket.Object(prefix + filename)
            obj.download_fileobj(f)
    if use_gzip:
        with gzip.open(path) as f:
            content = f.read().decode('utf-8')
    else:
        with open(path, 'r') as f:
            content = f.read()
    lines = content.splitlines()
    return lines


def load_labels_csv(path: str) -> list:
    if not os.path.exists(path):
        raise ValueError(
            f'Labels file {path} does not exist')
    labels = {}
    label_idx = ['epidural',
                 'intraparenchymal',
                 'intraventricular',
                 'subarachnoid',
                 'subdural',
                 'any']
    if path.endswith('.gz'):
        with gzip.open(path) as f:
            content = f.read().decode('utf-8')
    else:
        with open(path, 'r') as f:
            content = f.read()
    lines = content.splitlines()
    hdr = lines[0]
    if hdr != 'ID,Label':
        raise ValueError(f'bad header (got "{hdr}")')
    lines = lines[1:]
    cur_id = None
    cur_labels = None
    for i, line in enumerate(lines):
        item, label = line.split(',')
        label = int(label)
        if not label in [0, 1]:
            raise ValueError(f'invalid class label on line {i}')
        _, id, classname = item.split('_')
        if cur_id is None:
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
        id = os.path.basename(f)[3:f.index('.')]
        if id not in labels_dict:
            raise ValueError(f'missing class labels for {f}')
        labels.append(labels_dict[id])
    return torch.Tensor(labels)


def not_exist(path):
    return not os.path.exists(path) or os.path.getsize(path) == 0


class RSNAIntracranialDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 s3_bucket: str = 'rsna-ich',
                 s3_endpoint_url: str = 'https://nyc3.digitaloceanspaces.com',
                 delete_after_use: bool = False,
                 use_gzip: bool = True,
                 lod: int = 0):
        super(RSNAIntracranialDataset, self).__init__()
        self.root = root
        self.train = train
        self.s3_bucket = s3_bucket
        self.s3_endpoint_url = s3_endpoint_url
        self.download = download
        self.delete_after_use = delete_after_use
        self.prefix = 'stage_2_train/' if train else 'stage_2_test/'
        self.use_gzip = use_gzip
        self.lod = lod
        dcm_path = os.path.join(root, self.prefix)
        self.dcm_path = dcm_path
        if self.download:
            s3 = boto3.resource('s3',
                                endpoint_url=s3_endpoint_url)
            bucket = s3.Bucket(s3_bucket)
            self.files = get_inventory(
                bucket, root, self.prefix, download=download, use_gzip=use_gzip)
            if train:
                labels_csv_key = 'stage_2_train.csv'
                if use_gzip:
                    labels_csv_key += '.gz'
                labels_csv_path = os.path.join(root, labels_csv_key)
                if not_exist(labels_csv_path):
                    with open(labels_csv_path, 'wb') as f:
                        obj = bucket.Object(labels_csv_key)
                        obj.download_fileobj(f)
                self.labels = process_labels(
                    self.files, labels_csv_path) if train else None
            else:
                self.labels = None
        else:
            if not os.path.exists(dcm_path):
                raise ValueError(f'Directory {dcm_path} does not exist')
            if use_gzip:
                ext = '.dcm.gz'
                labels_file = 'stage_2_train.csv.gz'
            else:
                ext = '.dcm'
                labels_file = 'stage_2_train.csv'
            self.files = [f for f in os.listdir(dcm_path)
                          if f.endswith(ext)]
            self.labels = process_labels(
                self.files, os.path.join(root, labels_file)) if train else None

    def load_dcm(self, path: str) -> Tensor:
        if self.use_gzip:
            f = gzip.open(path)
            try:
                x = pydicom.dcmread(f, stop_before_pixels=False)
            finally:
                f.close()
        else:
            x = pydicom.dcmread(path, stop_before_pixels=False)
        x = normalized_dicom_pixels(x)
        return x

    def download_dcm(self, file: str, path: str):
        s3 = boto3.resource('s3',
                            endpoint_url=self.s3_endpoint_url)
        bucket = s3.Bucket(self.s3_bucket)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'wb') as f:
            key = self.prefix + file
            obj = bucket.Object(key)
            obj.download_fileobj(f)

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.dcm_path, file)
        y = self.labels[index] if self.labels is not None else []
        if not_exist(path):
            if not self.download:
                raise ValueError(f'File {path} does not exist')
            self.download_dcm(file, path)
        x = self.load_dcm(path)
        if self.delete_after_use:
            os.remove(path)
        if self.lod is not None:
            for _ in range(self.lod):
                x = F.avg_pool2d(x, 2, stride=2)
        return (x, y)

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
