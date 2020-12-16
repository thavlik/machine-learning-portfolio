import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np
import torch
import torch.utils.data as data
from skimage.io import imread
import tempfile
from typing import List


def read_hu(x):
    return imread(x).astype(np.float32) - 32768.0


HEADER = "File_name,Patient_index,Study_index,Series_ID,Key_slice_index,Measurement_coordinates,Bounding_boxes,Lesion_diameters_Pixel_,Normalized_lesion_location,Coarse_lesion_type,Possibly_noisy,Slice_range,Spacing_mm_px_,Image_size,DICOM_windows,Patient_gender,Patient_age,Train_Val_Test\n"


def load_labels_csv(path: str, components: List[str]) -> dict:
    labels = {}
    with open(path, 'r') as f:
        hdr = f.readline()
        if hdr != HEADER:
            raise ValueError('bad header')
        for line in f:
            parts = line.strip().split('"')
            parts = [part for part in parts
                     if part != ',']
            # filename, patient_index, study_index, series_id, key_slice_index = [p
            #                                                                    for p in parts[0].split(',')
            #                                                                    if len(p) > 0]
            filename = parts[0].split(',')[0]
            parts = parts[1:]
            measurement_coordinates, bounding_boxes, lesion_diameters_pixel, normalized_lesion_location = parts[
                :4]
            parts = parts[4:]
            measurement_coordinates = [
                float(s.strip()) for s in measurement_coordinates.split(',')]
            bounding_boxes = [float(s.strip())
                              for s in bounding_boxes.split(',')]
            lesion_diameters_pixel = [
                float(s.strip()) for s in lesion_diameters_pixel.split(',')]
            normalized_lesion_location = [
                float(s.strip()) for s in normalized_lesion_location.split(',')]

            coarse_lesion_type, possibly_noisy = [
                int(s.strip()) for s in parts[0][1:-1].split(',')]
            parts = parts[1:]
            slice_range = [int(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            spacing_mm_px_ = [float(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            width, height = [int(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            dicom_windows = [float(s.strip()) for s in parts[0].split(',')]
            parts = parts[1:]
            gender, age, _ = parts[0][1:].split(',')
            age = int(age) if age != 'NaN' else 0
            values = {
                'measurement_coordinates': measurement_coordinates,
                'bounding_boxes': bounding_boxes,
                'lesion_diameters_pixel': lesion_diameters_pixel,
                'normalized_lesion_location': normalized_lesion_location,
                'coarse_lesion_type': coarse_lesion_type,
                'possibly_noisy': possibly_noisy,
                'gender': 1 if gender == 'F' else 0,
                'slice_range': slice_range,
                'spacing_mm_px_': spacing_mm_px_,
                'age': age,
                'size': [width, height],
                'dicom_windows': dicom_windows,
            }
            labels[filename] = [values[k] for k in components]
    return labels


class DeepLesionDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 download: bool = True,
                 s3_bucket: str = 'deeplesion',
                 s3_endpoint_url: str = 'https://s3.us-central-1.wasabisys.com',
                 delete_after_use: bool = False,
                 components: List[str] = [
                     'measurement_coordinates',
                     'bounding_boxes',
                     'lesion_diameters_pixel',
                     'normalized_lesion_location',
                     'coarse_lesion_type',
                     'possibly_noisy',
                     'gender',
                     'slice_range',
                     'spacing_mm_px_',
                     'age',
                     'size',
                     'dicom_windows',
                 ]):
        super(DeepLesionDataset, self).__init__()
        self.root = root
        self.download = download
        self.delete_after_use = delete_after_use
        labels_csv_path = os.path.join(root, 'DL_info.csv')
        if self.download:
            s3 = boto3.resource('s3', endpoint_url=s3_endpoint_url)
            self.bucket = s3.Bucket(s3_bucket)
            inventory_path = os.path.join(root, 'inventory.txt')
            if not os.path.exists(inventory_path):
                with open(inventory_path, 'wb') as f:
                    obj = self.bucket.Object('inventory.txt')
                    obj.download_fileobj(f)
            with open(inventory_path, 'r') as f:
                self.files = [tuple(line.strip().split(','))
                              for line in f]
            if not os.path.exists(labels_csv_path):
                with open(inventory_path, 'wb') as f:
                    obj = self.bucket.Object('DL_info.csv')
                    obj.download_fileobj(f)
        else:
            images_dir = os.path.join(root, 'Images_png')
            files = []
            for d in os.listdir(images_dir):
                df = os.path.join(images_dir, d)
                for f in os.listdir(df):
                    files.append((d, f))
            self.files = files
        self.labels = load_labels_csv(labels_csv_path, components)

    def __getitem__(self, index):
        d, f = self.files[index]
        path = os.path.join(self.root, 'Images_png', d, f)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            if not self.download:
                raise ValueError(
                    f'with download == False, {path} was not found')
            with open(path, 'wb') as file:
                obj = self.bucket.Object(f'Images_png/{d}/{f}')
                obj.download_fileobj(file)
        x = read_hu(path)
        x = torch.Tensor(x)
        x = x.unsqueeze(0)
        y = self.labels.get(f'{d}_{f}', None)
        if self.delete_after_use:
            os.remove(path)
        return (x, y)

    def __len__(self):
        return len(self.files)


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


if __name__ == '__main__':
    os.environ['AWS_PROFILE'] = 'wasabi'
    ds = DeepLesionDataset('E:/deeplesion/')
    print(ds[0][0].shape)
    print(ds[1][0].shape)
