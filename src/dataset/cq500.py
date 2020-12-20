import os
import numpy as np
import torch
import torch.utils.data as data
import boto3
import pydicom

CQ500_HEADER = 'name,Category,R1:ICH,R1:IPH,R1:IVH,R1:SDH,R1:EDH,R1:SAH,R1:BleedLocation-Left,R1:BleedLocation-Right,R1:ChronicBleed,R1:Fracture,R1:CalvarialFracture,R1:OtherFracture,R1:MassEffect,R1:MidlineShift,R2:ICH,R2:IPH,R2:IVH,R2:SDH,R2:EDH,R2:SAH,R2:BleedLocation-Left,R2:BleedLocation-Right,R2:ChronicBleed,R2:Fracture,R2:CalvarialFracture,R2:OtherFracture,R2:MassEffect,R2:MidlineShift,R3:ICH,R3:IPH,R3:IVH,R3:SDH,R3:EDH,R3:SAH,R3:BleedLocation-Left,R3:BleedLocation-Right,R3:ChronicBleed,R3:Fracture,R3:CalvarialFracture,R3:OtherFracture,R3:MassEffect,R3:MidlineShift\n'

def load_labels(path: str) -> dict:
    labels = {}
    with open(path, 'r') as f:
        hdr = f.readline()
        if hdr != CQ500_HEADER:
            raise ValueError('bad header')
        for line in f:
            parts = line.strip().split(',')
            idx = parts[0][9:]
            labels[idx] = torch.Tensor([int(b) for b in parts[2:]])
    return labels

class CQ500Dataset(data.Dataset):
    def __init__(self,
                 root: str,
                 download: bool = True,
                 use_gzip: bool = True,
                 s3_bucket: str = 'cq500',
                 s3_endpoint: str = 'https://nyc3.digitaloceanspaces.com',
                 delete_after_use: bool = False):
        super().__init__()
        self.root = root
        self.download = download
        self.use_gzip = use_gzip
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint
        labels_csv_path = os.path.join(root, 'reads.csv')
        if download:
            if not os.path.exists(root):
                os.makedirs(root)
            if not os.path.exists(labels_csv_path) or os.path.getsize(labels_csv_path) == 0:
                s3 = boto3.resource('s3', endpoint_url=s3_endpoint)
                bucket = s3.Bucket(s3_bucket)
                with open(labels_csv_path, 'w') as f:
                    obj = bucket.Object('reads.csv')
                    obj.download_fileobj(f)
        elif not os.path.exists(labels_csv_path):
            raise ValueError(f'with download == False, {labels_csv_path} does not exist')
        self.labels = load_labels(labels_csv_path)

    def __getitem__(self, index):
        path = ''
        if self.use_gzip:
            path += '.gz'
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            key = ''
            s3 = boto3.resource('s3', endpoint_url=self.s3_endpoint)
            bucket = s3.Bucket(self.s3_bucket)
            with open(path, 'wb') as f:
                obj = bucket.Object(key)
                obj.download_fileobj(f)

        ds = pydicom.dcmread(self.files[index], stop_before_pixels=False)
        data = raw_dicom_pixels(ds)
        return (data, [])

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    ds = CQ500Dataset('E:/cq500')
    print(ds[0].shape)
    print(ds[1].shape)
