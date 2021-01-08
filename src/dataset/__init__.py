from .cq500 import *
from .deeplesion import DeepLesionDataset, get_output_features as deeplesion_get_output_features
from .dicom_util import *
from .grasp_and_lift_eeg import *
from .reference import *
from .rsna_intracranial import *
from .trends_fmri import *
#from .video import *
from .batch_video import *
from .toy_neural_graphics import *
import nonechucks as nc


def split_dataset(dataset, split):
    n_train_imgs = np.floor(len(dataset) * split).astype('int')
    n_val_imgs = len(dataset) - n_train_imgs
    cur_state = torch.get_rng_state()
    torch.manual_seed(torch.initial_seed())
    parts = torch.utils.data.random_split(dataset,
                                          [n_train_imgs, n_val_imgs])
    torch.set_rng_state(cur_state)
    return parts


datasets = {
    'cq500': CQ500Dataset,
    'deeplesion': DeepLesionDataset,
    'reference': ReferenceDataset,
    'rsna-intracranial': RSNAIntracranialDataset,
    'trends-fmri': TReNDSfMRIDataset,
    'grasp-and-lift-eeg': GraspAndLiftEEGDataset,
    'toy-neural-graphics': ToyNeuralGraphicsDataset,
    #'video': VideoDataset,
}


def get_dataset(name: str,
                params: dict,
                split: float = None,
                train: bool = True,
                safe: bool = True):
    if name not in datasets:
        raise ValueError(f"unknown dataset '{name}'")
    ds = datasets[name](**params)
    if split is not None:
        ds = split_dataset(ds, split)[1 if train else 0]
    if safe:
        ds = nc.SafeDataset(ds)
    return ds


dataset_dims = {
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
        size = data['training']['num_samples']
        lod = data['training'].get('lod', 0)
        divisor = 2 ** lod
        size = size // divisor
        return torch.Size((32, size))
    if name in ['rsna-intracranial', 'deeplesion', 'cq500']:
        lod = data['training'].get('lod', 0)
        divisor = 2 ** lod
        size = 512 // divisor
        size = (1, size, size)
        return torch.Size(size)
    if name not in dataset_dims:
        raise ValueError(f'unknown dataset "{name}"')
    return torch.Size(dataset_dims[name])


def get_output_features(data: dict) -> int:
    if data['name'] == 'deeplesion':
        # First component is certainty of class label
        return deeplesion_get_output_features(data['training']['components'])
    else:
        raise NotImplementedError
