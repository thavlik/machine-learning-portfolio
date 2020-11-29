from .base import BaseVAE
from .classifier import Classifier
from .resnet_classifier2d import ResNetClassifier2d
from .resnet_vae1d import ResNetVAE1d
from .resnet_vae2d import ResNetVAE2d
from .resnet_vae3d import ResNetVAE3d
from .resnet_vae4d import ResNetVAE4d

models = {
    'ResNetClassifier2d': ResNetClassifier2d,
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
