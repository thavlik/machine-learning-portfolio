import torch
from models import create_model
from vae import VAEExperiment
from dataset import ReferenceDataset, get_example_shape


def vae1d(config: dict):
    exp_params = config['exp_params']
    c, l = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         num_samples=l,
                         channels=c)
    return VAEExperiment(model,
                         params=exp_params)


def vae2d(config: dict):
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         enable_fid='fid_weight' in exp_params)
    return VAEExperiment(model,
                         params=exp_params)


def vae3d(config: dict):
    exp_params = config['exp_params']
    c, d, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         depth=d,
                         channels=c)
    return VAEExperiment(model,
                         params=exp_params)


entrypoints = {
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
}


def create_experiment(config: dict):
    if 'entrypoint' not in config:
        raise ValueError('config has no entrypoint')
    entrypoint = config['entrypoint']
    if entrypoint not in entrypoints:
        raise ValueError(f'unknown entrypoint "{entrypoint}" '
                         f'valid options are {entrypoints}')
    return entrypoints[entrypoint](config)
