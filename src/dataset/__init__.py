from .cq500 import *
from .deeplesion import *
from .reference import *
from .rsna_intracranial import *
from .trends_fmri import *
from .video import *

def get_dataset(name: str, params: dict):
    if name == 'cq500':
        return CQ500Dataset(**params)
    elif name == 'deeplesion':
        return DeepLesionDataset(**params)
    elif name == 'reference':
        return ReferenceDataset(**params)
    elif name == 'rsna-intracranial':
        return RSNAIntracranialDataset(**params)
    elif name == 'trends-fmri':
        return TReNDSfMRIDataset(**params)
    elif name == 'video':
        return VideoDataset(**params)
    else:
        raise ValueError(f"unknown dataset loader '{name}'")


dataset_dims = {
    'eeg': (1, 8192),  # channels, length
    'cq500': (1, 512, 512),  # channels, height, width
    'deeplesion': (1, 512, 512),
    'rsna-intracranial': (1, 512, 512),
    'trends-fmri': (53, 63, 52, 53),
}


def get_example_shape(data: dict):
    name = data['name']
    if name == 'reference':
        ds = ReferenceDataset(**data['training'])
        x, _ = ds[0]
        return x.shape
    if name == 'video':
        params = data['training']
        return torch.Size((3, params['height'], params['width']))
    if name not in dataset_dims:
        raise ValueError(f'unknown dataset "{name}"')
    return torch.Size(dataset_dims[name])
