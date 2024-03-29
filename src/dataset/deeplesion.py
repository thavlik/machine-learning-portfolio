import os
import torch
import torch.utils.data as data
from torch.nn import functional as F
from typing import List

import boto3
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def read_hu(x):
    return resize(imread(x).astype(np.float32) - 32768, (512, 512))


HEADER = "File_name,Patient_index,Study_index,Series_ID,Key_slice_index,Measurement_coordinates,Bounding_boxes,Lesion_diameters_Pixel_,Normalized_lesion_location,Coarse_lesion_type,Possibly_noisy,Slice_range,Spacing_mm_px_,Image_size,DICOM_windows,Patient_gender,Patient_age,Train_Val_Test\n"

COMPONENT_LENGTHS = {
    'measurement_coordinates': 8,
    'bounding_boxes': 4,
    'lesion_diameters_pixel': 2,
    'normalized_lesion_location': 3,
    'coarse_lesion_type': 1,
    'possibly_noisy': 1,
    'gender': 1,
    'slice_range': 2,
    'spacing_mm_px': 3,
    'age': 1,
    'size': 2,
    'dicom_windows': 2,
}


def get_output_features(components: List[str]) -> int:
    return sum([COMPONENT_LENGTHS[k] for k in components])


def flatten(test_list):
    # define base case to exit recursive method
    if len(test_list) == 0:
        return []
    elif isinstance(test_list, list) and type(test_list[0]) in [int, str]:
        return [test_list[0]] + flatten(test_list[1:])
    elif isinstance(test_list, list) and isinstance(test_list[0], list):
        return test_list[0] + flatten(test_list[1:])
    else:
        return flatten(test_list[1:])


def load_labels_csv(path: str, components: List[str],
                    flatten_components: bool) -> dict:
    labels = {}
    with open(path, 'r') as f:
        hdr = f.readline()
        if hdr != HEADER:
            raise ValueError('bad header')
        for line in f:
            parts = line.strip().split('"')
            parts = [part for part in parts if part != ',']
            # filename, patient_index, study_index, series_id, key_slice_index = [p
            #                                                                    for p in parts[0].split(',')
            #                                                                    if len(p) > 0]
            filename = parts[0].split(',')[0]
            parts = parts[1:]
            measurement_coordinates, bounding_boxes, lesion_diameters_pixel, normalized_lesion_location = parts[:
                                                                                                                4]
            parts = parts[4:]
            measurement_coordinates = [
                float(s.strip()) for s in measurement_coordinates.split(',')
            ]
            bounding_boxes = [
                float(s.strip()) for s in bounding_boxes.split(',')
            ]
            lesion_diameters_pixel = [
                float(s.strip()) for s in lesion_diameters_pixel.split(',')
            ]
            normalized_lesion_location = [
                float(s.strip()) for s in normalized_lesion_location.split(',')
            ]

            coarse_lesion_type, possibly_noisy = [
                int(s.strip()) for s in parts[0][1:-1].split(',')
            ]
            parts = parts[1:]
            slice_range = [int(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            spacing_mm_px = [float(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            width, height = [int(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            dicom_windows = [float(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            gender, age, _ = parts[0][1:].split(',')
            age = int(age) if age != 'NaN' else 0

            # Normalize bounding box coordinates
            bounding_boxes[0] /= width
            bounding_boxes[1] /= height
            bounding_boxes[2] /= width
            bounding_boxes[3] /= height

            values = {
                'measurement_coordinates': measurement_coordinates,
                'bounding_boxes': bounding_boxes,
                'lesion_diameters_pixel': lesion_diameters_pixel,
                'normalized_lesion_location': normalized_lesion_location,
                'coarse_lesion_type': coarse_lesion_type,
                'possibly_noisy': possibly_noisy,
                'gender': 1 if gender == 'F' else 0,
                'slice_range': slice_range,
                'spacing_mm_px': spacing_mm_px,
                'age': age,
                'size': [width, height],
                'dicom_windows': dicom_windows,
            }
            comps = [values[k] for k in components]
            if flatten_components:
                comps = torch.Tensor(flatten(comps))
            labels[filename] = comps
    return labels


def ensure_downloaded(key, path, bucket):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'wb') as f:
            obj = bucket.Object(key)
            obj.download_fileobj(f)


class DeepLesionDataset(data.Dataset):

    def __init__(self,
                 root: str,
                 download: bool = True,
                 s3_bucket: str = 'deeplesion',
                 s3_endpoint: str = 'https://nyc3.digitaloceanspaces.com',
                 delete_after_use: bool = False,
                 only_positives: bool = False,
                 flatten_labels: bool = True,
                 lod: int = 0,
                 limit: int = None,
                 include_label: bool = True,
                 components: List[str] = [
                     'measurement_coordinates',
                     'bounding_boxes',
                     'lesion_diameters_pixel',
                     'normalized_lesion_location',
                     'coarse_lesion_type',
                     'possibly_noisy',
                     'gender',
                     'slice_range',
                     'spacing_mm_px',
                     'age',
                     'size',
                     'dicom_windows',
                 ]):
        super(DeepLesionDataset, self).__init__()
        self.root = root
        self.download = download
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint
        self.delete_after_use = delete_after_use
        self.lod = lod
        self.include_label = include_label
        labels_csv_path = os.path.join(root, 'DL_info.csv')
        if self.download:
            if not os.path.exists(root):
                os.makedirs(root)
            s3 = boto3.resource('s3', endpoint_url=s3_endpoint)
            bucket = s3.Bucket(s3_bucket)
            inventory_path = os.path.join(root, 'inventory.txt')
            ensure_downloaded('inventory.txt', inventory_path, bucket)
            with open(inventory_path, 'r') as f:
                self.files = [tuple(line.strip().split(',')) for line in f]
            ensure_downloaded('DL_info.csv', labels_csv_path, bucket)
        else:
            images_dir = os.path.join(root, 'Images_png')
            files = []
            for d in os.listdir(images_dir):
                df = os.path.join(images_dir, d)
                for f in os.listdir(df):
                    files.append((d, f))
            self.files = files
        self.labels = load_labels_csv(labels_csv_path, components,
                                      flatten_labels)
        self.zeros = torch.zeros(self.labels[list(
            self.labels.keys())[0]].shape[0])
        if only_positives:
            self.files = [(d, f) for d, f in self.files
                          if f'{d}_{f}' in self.labels]
        if limit is not None:
            self.files = self.files[:limit]

    def get_label(self, index):
        d, f = self.files[index]
        key = f'{d}_{f}'
        y = 1 if key in self.labels else 0
        return torch.Tensor([y])

    def get_positive_example(self, end_idx: int = None):
        n = len(self)
        start_idx = np.random.randint(0, n) if end_idx is None else 0
        for i in range(n - start_idx):
            index = i + start_idx
            d, f = self.files[index]
            key = f'{d}_{f}'
            if key in self.labels:
                return self.__getitem__(index)
        if end_idx is not None:
            raise ValueError('unable to seek')
        return self.get_positive_example(end_idx=start_idx)

    def __getitem__(self, index):
        d, f = self.files[index]
        path = os.path.join(self.root, 'Images_png', d, f)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            if not self.download:
                raise ValueError(
                    f'with download == False, {path} was not found')
            s3 = boto3.resource('s3', endpoint_url=self.s3_endpoint)
            bucket = s3.Bucket(self.s3_bucket)
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(path, 'wb') as file:
                key = f'Images_png/{d}/{f}'
                obj = bucket.Object(key)
                obj.download_fileobj(file)
        x = read_hu(path)
        x = torch.Tensor(x)
        x = x.unsqueeze(0)
        key = f'{d}_{f}'
        label = key in self.labels
        y = self.labels[key] if label else self.zeros
        label = torch.Tensor([float(label)])
        if self.delete_after_use:
            os.remove(path)
        if x.shape != torch.Size([1, 512, 512]):
            raise ValueError(f'Invalid shape {x.shape}')
        if self.lod is not None:
            for _ in range(self.lod):
                x = F.avg_pool2d(x, 2, stride=2)
        if self.include_label:
            return (x, label, y)
        else:
            return (x, y)

    def __len__(self):
        return len(self.files)
