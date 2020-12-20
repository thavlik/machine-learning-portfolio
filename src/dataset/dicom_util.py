import torch
import numpy as np
from torchvision.transforms import Resize, ToPILImage, ToTensor


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
    if x.numel() != 512 * 512:
        dim = torch.sqrt(torch.Tensor([x.numel()]))
        if dim.floor() != dim.ceil():
            raise ValueError('Non-square number of input elements '
                             f'got {x.numel()}')
        x = ToPILImage()(x)
        x = Resize((512, 512))(x)
        x = ToTensor()(x)
        print(f'Successfully resized from {int(dim)}x{int(dim)}')
    x = x.view(1, 512, 512)
    return x
