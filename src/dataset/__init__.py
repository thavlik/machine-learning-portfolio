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