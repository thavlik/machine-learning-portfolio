from .base import BaseVAE
from .resnet1d import BasicBlock1d, TransposeBasicBlock1d
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from .resnet3d import BasicBlock3d, TransposeBasicBlock3d
from .resnet4d import BasicBlock4d, TransposeBasicBlock4d
from .resnet_vae1d import ResNetVAE1d
from .resnet_vae2d import ResNetVAE2d
from .resnet_vae3d import ResNetVAE3d
from .resnet_vae4d import ResNetVAE4d

models = {
    'ResNetVAE1d': ResNetVAE1d,
    'ResNetVAE2d': ResNetVAE2d,
    'ResNetVAE3d': ResNetVAE3d,
    'ResNetVAE4d': ResNetVAE4d,
}

def create_model(arch: str, **kwargs):
    if arch not in models:
        raise ValueError(f'unknown model architecture "{arch}" '
                         f'valid options are {models}')
    return models[arch](**kwargs)
