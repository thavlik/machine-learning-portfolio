from torch import nn

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


def get_pooling1d(name: str) -> nn.Module:
    if name not in pool1d:
        raise ValueError(f'Unknown 1d pool function "{name}", '
                         f'valid options are {pool1d}')
    return pool1d[name]


def get_pooling2d(name: str) -> nn.Module:
    if name not in pool2d:
        raise ValueError(f'Unknown pool function "{name}", '
                         f'valid options are {pool2d}')
    return pool2d[name]
