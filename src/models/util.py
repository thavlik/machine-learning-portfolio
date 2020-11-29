from torch import nn
from .maxpool4d import MaxPool4d

pool1d = {
    'max': nn.MaxPool1d,
    'min': nn.MaxPool1d,
    'avg': nn.AvgPool1d,
}

pool2d = {
    'max': nn.MaxPool2d,
    'min': nn.MaxPool2d,
    'avg': nn.AvgPool2d,
}

pool3d = {
    'max': nn.MaxPool3d,
    'min': nn.MaxPool3d,
    'avg': nn.AvgPool3d,
}

pool4d = {
    'max': MaxPool4d,
}


def get_pooling1d(name: str) -> nn.Module:
    if name not in pool1d:
        raise ValueError(f'Unknown 1D pool function "{name}", '
                         f'valid options are {pool1d}')
    return pool1d[name]


def get_pooling2d(name: str) -> nn.Module:
    if name not in pool2d:
        raise ValueError(f'Unknown 2D pooling function "{name}", '
                         f'valid options are {pool2d}')
    return pool2d[name]


def get_pooling3d(name: str) -> nn.Module:
    if name not in pool3d:
        raise ValueError(f'Unknown 3D pooling function "{name}", '
                         f'valid options are {pool3d}')
    return pool3d[name]


def get_pooling4d(name: str) -> nn.Module:
    if name not in pool4d:
        raise ValueError(f'Unknown 4D pooling function "{name}", '
                         f'valid options are {pool4d}')
    return pool4d[name]


act_options = {
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
}


def get_activation(name: str) -> nn.Module:
    if name not in act_options:
        raise ValueError(
            f'Unknown activation function "{name}"')
    return act_options[name]()
