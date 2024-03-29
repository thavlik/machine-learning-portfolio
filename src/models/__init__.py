from ray.rllib.models import ModelCatalog

from .base import BaseVAE
from .classifier import Classifier
from .localizer import Localizer
from .renderer import BaseRenderer
from .resnet_augmenter1d import ResNetAugmenter1d
from .resnet_classifier1d import ResNetClassifier1d
from .resnet_classifier2d import ResNetClassifier2d
from .resnet_classifier3d import ResNetClassifier3d
from .resnet_embed2d import ResNetEmbed2d
from .resnet_gaussian_localizer2d import ResNetGaussianLocalizer2d
from .resnet_localizer2d import ResNetLocalizer2d
from .resnet_renderer2d import ResNetRenderer2d
from .resnet_rl1d import ResNetRL1d
from .resnet_rl2d import ResNetRL2d
from .resnet_sandwich2d import ResNetSandwich2d
from .resnet_vae1d import ResNetVAE1d
from .resnet_vae2d import ResNetVAE2d
from .resnet_vae3d import ResNetVAE3d
from .resnet_vae4d import ResNetVAE4d

# ModelCatalog.register_custom_model("ResNetRL1d", ResNetRL1d)
# ModelCatalog.register_custom_model("ResNetRL2d", ResNetRL2d)

models = {
    'ResNetAugmenter1d': ResNetAugmenter1d,
    'ResNetClassifier1d': ResNetClassifier1d,
    'ResNetClassifier2d': ResNetClassifier2d,
    'ResNetClassifier3d': ResNetClassifier3d,
    'ResNetEmbed2d': ResNetEmbed2d,
    'ResNetSandwich2d': ResNetSandwich2d,
    'ResNetGaussianLocalizer2d': ResNetGaussianLocalizer2d,
    'ResNetLocalizer2d': ResNetLocalizer2d,
    'ResNetRenderer2d': ResNetRenderer2d,
    'ResNetRL1d': ResNetRL1d,
    'ResNetRL2d': ResNetRL2d,
    'ResNetVAE1d': ResNetVAE1d,
    'ResNetVAE2d': ResNetVAE2d,
    'ResNetVAE3d': ResNetVAE3d,
    'ResNetVAE4d': ResNetVAE4d,
}


def create_model(arch: str, **kwargs):
    if arch not in models:
        raise ValueError(f'unknown model architecture "{arch}" '
                         f'valid options are {models}')
    try:
        model = models[arch](**kwargs)
    except:
        print(f'failed to create model "{arch}"')
        raise
    return model
