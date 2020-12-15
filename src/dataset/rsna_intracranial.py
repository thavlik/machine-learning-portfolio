import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from .dicom_util import normalized_dicom_pixels
import boto3
import tempfile


def get_inventory(bucket, dcm_path, prefix):
    filename = 'inventory.txt'
    path = os.path.join(dcm_path, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f]
    with open(path, 'w') as f:
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
        if f.readline() != 'ID,Label\n':
            raise ValueError('bad header')
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


class RSNAIntracranialDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 s3_path: str = 's3://rsna-intracranial',
                 s3_endpoint_url: str = 'https://nyc3.digitaloceanspaces.com',
                 limit: int = None,
                 generate_inventory: bool = True,
                 delete_after_use: bool = False):
        super(RSNAIntracranialDataset, self).__init__()
        self.root = root
        self.train = train
        self.download = download
        self.delete_after_use = delete_after_use
        dcm_path = os.path.join(
            root, 'stage_2_train' if train else 'stage_2_test')
        self.dcm_path = dcm_path
        if not self.download:
            if not os.path.exists(dcm_path):
                raise ValueError(f'Directory {dcm_path} does not exist')
            self.files = [f for f in os.listdir(dcm_path)
                          if f.endswith('.dcm')]
            self.labels = process_labels(
                self.files, os.path.join(root, 'stage_2_train.csv')) if train else None
        else:
            if s3_path == None:
                raise ValueError(
                    "You must provide s3_path when download == True")
            bucket = s3_path[len('s3://'):]
            try:
                bucket = bucket[:bucket.index('/')]
            except:
                pass
            prefix = s3_path[s3_path.index(bucket)+len(bucket):]
            if prefix.startswith('/'):
                prefix = prefix[1:]
            if not prefix.endswith('/'):
                prefix += '/'
            self.prefix = prefix
            s3 = boto3.resource('s3', endpoint_url=s3_endpoint_url)
            self.bucket = s3.Bucket(bucket)
            try:
                # Check if inventory file exists
                self.files = get_inventory(self.bucket, dcm_path, prefix)
                if limit != None:
                    self.files = self.files[:limit]
            except:
                self.files = []
                prefix_len = len(prefix)
                for o in self.bucket.objects.filter(Prefix=prefix):
                    if limit != None and len(self.files) >= limit:
                        break
                    if o.key.endswith('.dcm'):
                        self.files.append(o.key[prefix_len:])
                if generate_inventory:
                    path = os.path.join(dcm_path, 'inventory.txt')
                    with open(path, 'w') as f:
                        for line in self.files:
                            f.write(f'{line}\n')
            if train:
                labels_csv_path = os.path.join(root, 'stage_2_train.csv')
                if not os.path.exists(labels_csv_path):
                    with open(labels_csv_path, 'w') as f:
                        obj = self.bucket.Object('stage_2_train.csv')
                        obj.download_fileobj(f)
                self.labels = process_labels(
                    self.files, os.path.join(root, 'stage_2_train.csv')) if train else None
            else:
                self.labels = None

    def __getitem__(self, index):
        file = self.files[index]
        path = os.path.join(self.dcm_path, file)
        if os.path.exists(path):
            x = pydicom.dcmread(path, stop_before_pixels=False)
            x = normalized_dicom_pixels(x)
            y = self.labels[index] if self.labels != None else []
            return (x, y)
        elif not self.download:
            raise ValueError(f'File {path} does not exist')
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


if __name__ == '__main__':
    import matplotlib.pylab as plt
    #ds = RSNAIntracranialDataset('E:/rsna-intracranial/stage_2_test')
    # s3_path='s3://rsna-intracranial/stage_2_train',
    ds = RSNAIntracranialDataset(dcm_path='E:/rsna-intracranial/stage_2_train',
                                 labels_csv_path='E:/rsna-intracranial/stage_2_train.csv',
                                 download=False)
    fig = plt.figure(figsize=(15, 10))
    columns = 5
    rows = 4
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(ds[i], cmap=plt.cm.bone)
    plt.show()
