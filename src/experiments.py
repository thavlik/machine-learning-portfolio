import torch
from models import create_model
from vae import VAEExperiment
from dataset import ReferenceDataset, get_example_shape


def vae1d(config: dict,
          dataset: dict):
    c, l = get_example_shape(dataset)
    model = create_model(**config['model_params'],
                         length=l,
                         channels=c)
    exp_params = config['exp_params']
    return VAEExperiment(model,
                         params=exp_params,
                         dataset=dataset)


def vae2d(config: dict,
          dataset: dict):
    c, h, w = get_example_shape(dataset)
    exp_params = config['exp_params']
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         enable_fid='fid_weight' in exp_params)
    return VAEExperiment(model,
                         params=exp_params,
                         dataset=dataset)


def vae3d(config: dict,
          dataset: dict):
    c, d, h, w = get_example_shape(dataset)
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         depth=d,
                         channels=c)
    exp_params = config['exp_params']
    return VAEExperiment(model,
                         params=exp_params,
                         dataset=dataset)


experiments = {
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
}


def create_experiment(entrypoint: str, config: dict, dataset: dict):
    if entrypoint not in experiments:
        raise ValueError(f'unknown entrypoint "{entrypoint}" '
                         f'valid options are {experiments}')
    return experiments[entrypoint](config, dataset)
