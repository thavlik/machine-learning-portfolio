import torch
import numpy as np


def raw_dicom_pixels(ds):
    signed = ds.PixelRepresentation == 1
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    x = ds.pixel_array
    x = np.frombuffer(x, dtype='int16' if signed else 'uint16')
    x = np.array(x, dtype='float32')
    x = x * slope + intercept
    x = torch.Tensor(x)
    #x = x.clamp(0.0, 1.0)
    # TODO: fix normalization
    x = x.view(1, 512, 512)
    return x


def normalized_dicom_pixels(ds):
    signed = ds.PixelRepresentation == 1
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    x = ds.pixel_array
    if ds.BitsStored == 12 and not signed and int(intercept) > -100:
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        x += 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        intercept -= 1000
    x = np.frombuffer(x, dtype='int16' if signed else 'uint16')
    x = np.array(x, dtype='float32')
    x = x * slope + intercept
    x = torch.Tensor(x)
    x = x.view(1, 512, 512)
    assert x.shape == torch.Size([1, 512, 512]), f'got shape {ds.shape}'
    return x
