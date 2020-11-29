import torch
from models import create_model
from vae import VAEExperiment
from classification import ClassificationExperiment
from dataset import ReferenceDataset, get_example_shape
from pytorch_lightning.loggers import TestTubeLogger
import numpy as np
from pytorch_lightning import Trainer


def classification2d(config: dict, run_args: dict):
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c)
    return ClassificationExperiment(model,
                                    params=exp_params)


def classification_sandwich2d(config: dict, run_args: dict):
    base_experiment = experiment_main(config['base_experiment'], **run_args)
    sandwich_layers = base_experiment.model.get_sandwich_layers()
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         sandwich_layers=sandwich_layers)
    return ClassificationExperiment(model,
                                    params=exp_params)


def vae1d(config: dict, run_args: dict):
    exp_params = config['exp_params']
    c, l = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         num_samples=l,
                         channels=c)
    return VAEExperiment(model,
                         params=exp_params)


def vae2d(config: dict, run_args: dict):
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         enable_fid='fid_weight' in exp_params)
    return VAEExperiment(model,
                         params=exp_params)


def vae3d(config: dict, run_args: dict):
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
    'classification2d': classification2d,
    'classification_sandwich2d': classification_sandwich2d,
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
}


def create_experiment(config: dict, run_args: dict):
    if 'entrypoint' not in config:
        raise ValueError('config has no entrypoint')
    entrypoint = config['entrypoint']
    if entrypoint not in entrypoints:
        raise ValueError(f'unknown entrypoint "{entrypoint}" '
                         f'valid options are {entrypoints}')
    return entrypoints[entrypoint](config, run_args)


def experiment_main(config: dict,
                    save_dir: str,
                    exp_no: int,
                    total_experiments: int,
                    dry_run: bool):
    torch.manual_seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    experiment = create_experiment(config, run_args=dict(
        save_dir=save_dir,
        exp_no=exp_no,
        total_experiments=total_experiments,
        dry_run=dry_run,
    )).cuda()
    tt_logger = TestTubeLogger(save_dir=save_dir,
                               name=config['logging_params']['name'],
                               debug=False,
                               create_git_tag=False)
    if dry_run:
        config['trainer_params']['max_steps'] = 5
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     num_sanity_val_steps=5,
                     logger=tt_logger,
                     **config['trainer_params'])
    print(
        f"======= Training {config['model_params']['name']}/{config['logging_params']['name']} (Experiment {exp_no+1}/{total_experiments}) =======")
    print(config)
    runner.fit(experiment)
    return experiment
