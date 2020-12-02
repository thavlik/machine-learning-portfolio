from .cq500 import *
from .deeplesion import *
from .dicom_util import *
from .grasp_and_lift_eeg import *
from .reference import *
from .rsna_intracranial import *
from .trends_fmri import *
from .video import *
from .batch_video import *

datasets = {
    'cq500': CQ500Dataset,
    'deeplesion': DeepLesionDataset,
    'reference': ReferenceDataset,
    'rsna-intracranial': RSNAIntracranialDataset,
    'trends-fmri': TReNDSfMRIDataset,
    'grasp-and-lift-eeg': GraspAndLiftEEGDataset,
    'video': VideoDataset,
}


def get_dataset(name: str, params: dict):
    if name not in datasets:
        raise ValueError(f"unknown dataset '{name}'")
    return datasets[name](**params)

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
        train = data['training']
        return torch.Size((3, train['height'], train['width']))
    if name == 'batch-video':
        l = data['loader']
        return torch.Size((l['num_frames'], 3, l['height'], l['width']))
    if name == 'grasp-and-lift-eeg':
        return (32, data['training']['num_samples'])
    if name not in dataset_dims:
        raise ValueError(f'unknown dataset "{name}"')
    return torch.Size(dataset_dims[name])

