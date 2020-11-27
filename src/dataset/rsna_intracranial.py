import os
import numpy as np
import torch
import torch.utils.data as data
import pydicom
from dicom_util import normalized_dicom_pixels
import boto3
import tempfile

def get_inventory(bucket, cache_dir, prefix):
    filename = 'inventory.txt'
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.readlines()
    with open(path, 'w') as f:
        obj = bucket.Object(prefix + filename)
        obj.download_fileobj(f)
    with open(path, 'r') as f:
        return f.readlines()

class RSNAIntracranialDataset(data.Dataset):
    def __init__(self,
                 dcm_path: str,
                 s3_endpoint_url: str = 'https://nyc3.digitaloceanspaces.com',
                 cache_dir: str = None,
                 limit: int = None,
                 generate_inventory: bool = True):
        super(RSNAIntracranialDataset, self).__init__()
        self.dcm_path = dcm_path
        self.local = not dcm_path.startswith('s3://')
        if self.local:
            if not os.path.exists(dcm_path):
                raise ValueError(f'{dcm_path} does not exist')
            self.files = [f for f in os.listdir(dcm_path)
                          if f.endswith('.dcm')]
        else:
            if cache_dir == None:
                raise ValueError("You must provide a cache_dir when using s3")
            bucket = dcm_path[len('s3://'):]
            try:
                bucket = bucket[:bucket.index('/')]
            except:
                pass
            prefix = dcm_path[dcm_path.index(bucket)+len(bucket):]
            if prefix.startswith('/'):
                prefix = prefix[1:]
            if not prefix.endswith('/'):
                prefix += '/'
            s3 = boto3.resource('s3', endpoint_url=s3_endpoint_url)
            self.bucket = s3.Bucket(bucket)
            try:
                # Check if inventory file exists
                self.files = get_inventory(self.bucket, cache_dir, prefix)
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
                    path = os.path.join(cache_dir, 'inventory.txt')
                    with open(path, 'w') as f:
                        for line in self.files:
                            f.write(f'{line}\n')
            self.cache_dir = cache_dir

    def __getitem__(self, index):
        file = self.files[index]
        if self.local:
            path = os.path.join(self.dcm_path, file)
            ds = pydicom.dcmread(path, stop_before_pixels=False)
            return normalized_dicom_pixels(ds)
        path = os.path.join(self.cache_dir, file)
        if os.path.exists(path):
            ds = pydicom.dcmread(path, stop_before_pixels=False)
            return normalized_dicom_pixels(ds)
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, 'wb') as f:
            obj = self.bucket.Object(file)
            obj.download_fileobj(f)
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        data = normalized_dicom_pixels(ds)
        return data

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import matplotlib.pylab as plt
    #ds = RSNAIntracranialDataset('E:/rsna-intracranial/stage_2_test')
    ds = RSNAIntracranialDataset('s3://rsna-intracranial/stage_2_train',
                                 cache_dir='E:/rsna-intracranial/stage_2_train')
    print(ds[3].shape)
    fig = plt.figure(figsize=(15, 10))
    columns = 5
    rows = 4
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(ds[i], cmap=plt.cm.bone)
    plt.show()
