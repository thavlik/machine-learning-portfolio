import os
import numpy as np
import torch
import torch.utils.data as data
from skimage.io import imread


def read_hu(x):
    return imread(x).astype(np.float32) - 32768.0


HEADER = "File_name,Patient_index,Study_index,Series_ID,Key_slice_index,Measurement_coordinates,Bounding_boxes,Lesion_diameters_Pixel_,Normalized_lesion_location,Coarse_lesion_type,Possibly_noisy,Slice_range,Spacing_mm_px_,Image_size,DICOM_windows,Patient_gender,Patient_age,Train_Val_Test\n"


def load_labels_csv(path: str) -> dict:
    labels = {}
    with open(path, 'r') as f:
        hdr = f.readline()
        if hdr != HEADER:
            raise ValueError('bad header')
        for line in f:
            parts = line.strip().split('"')
            parts = [part for part in parts
                     if part != ',']
            #filename, patient_index, study_index, series_id, key_slice_index = [p
            #                                                                    for p in parts[0].split(',')
            #                                                                    if len(p) > 0]
            filename = parts[0].split(',')[0]
            parts = parts[1:]
            measurement_coordinates, bounding_boxes, lesion_diameters_pixel_, normalized_lesion_location = parts[:4]
            parts = parts[4:]
            measurement_coordinates = [float(s.strip()) for s in measurement_coordinates.split(',')]
            bounding_boxes = [float(s.strip()) for s in bounding_boxes.split(',')]
            lesion_diameters_pixel_ = [float(s.strip()) for s in lesion_diameters_pixel_.split(',')]
            normalized_lesion_location = [float(s.strip()) for s in normalized_lesion_location.split(',')]

            coarse_lesion_type, possibly_noisy = [int(s.strip()) for s in parts[0][1:-1].split(',')]
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
            labels[filename] = [
                measurement_coordinates,
                bounding_boxes,
                lesion_diameters_pixel_,
                normalized_lesion_location,
                coarse_lesion_type,
                possibly_noisy,
                slice_range,
                [width, height],
                spacing_mm_px_,
                dicom_windows,
                gender,
                age,
            ]
    return labels


class DeepLesionDataset(data.Dataset):
    def __init__(self, root: str):
        super(DeepLesionDataset, self).__init__()
        self.root = root
        self.labels = load_labels_csv(os.path.join(root, 'DL_info.csv'))
        images_dir = os.path.join(root, 'Images_png')
        files = []
        for d in os.listdir(images_dir):
            df = os.path.join(images_dir, d)
            for f in os.listdir(df):
                files.append((d, f))
        self.files = files

    def __getitem__(self, index):
        d, f = self.files[index]
        path = os.path.join(self.root, 'Images_png', d, f)
        x = read_hu(path)
        x = torch.Tensor(x)
        x = x.unsqueeze(0)
        y = self.labels.get(f'{d}_{f}', None)
        return (x, y)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    ds = DeepLesionDataset('E:/deeplesion/')
    print(ds[0][0].shape)
    print(ds[1][0].shape)
