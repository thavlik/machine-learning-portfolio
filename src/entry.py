import os
import torch
from models import create_model
from vae import VAEExperiment
from classification import ClassificationExperiment
from dataset import ReferenceDataset, get_example_shape
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
import numpy as np
from pytorch_lightning import Trainer
from load_config import load_config
from ray import tune
from ray.tune.logger import TBXLogger
from ray.rllib.models import ModelCatalog
from env import get_env
from plot import plot_comparison


def classification2d(config: dict, run_args: dict) -> ClassificationExperiment:
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c)
    return ClassificationExperiment(model,
                                    params=exp_params)


def classification_embed2d(config: dict, run_args: dict) -> ClassificationExperiment:
    base_experiment = experiment_main(
        load_config(config['base_experiment']), **run_args)
    encoder = base_experiment.model.get_encoder()
    encoder.requires_grad = False
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         encoder=encoder)
    return ClassificationExperiment(model,
                                    params=exp_params)


def classification_sandwich2d(config: dict, run_args: dict) -> ClassificationExperiment:
    base_experiment = experiment_main(
        load_config(config['base_experiment']), **run_args)
    encoder = base_experiment.model.get_encoder()
    encoder.requires_grad = False
    sandwich_layers = base_experiment.model.get_sandwich_layers()
    for layer, _ in sandwich_layers:
        layer.requires_grad = False
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         encoder=encoder,
                         sandwich_layers=sandwich_layers)
    return ClassificationExperiment(model,
                                    params=exp_params)


def vae1d(config: dict, run_args: dict) -> VAEExperiment:
    exp_params = config['exp_params']
    c, l = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         num_samples=l,
                         channels=c)
    return VAEExperiment(model,
                         params=exp_params)


def vae2d(config: dict, run_args: dict) -> VAEExperiment:
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         enable_fid='fid_weight' in exp_params,
                         progressive_growing=len(exp_params['progressive_growing']) if 'progressive_growing' in exp_params else 0)
    return VAEExperiment(model,
                         params=exp_params)


def vae3d(config: dict, run_args: dict) -> VAEExperiment:
    exp_params = config['exp_params']
    c, d, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         depth=d,
                         channels=c)
    return VAEExperiment(model,
                         params=exp_params)


def vae4d(config: dict, run_args: dict) -> VAEExperiment:
    exp_params = config['exp_params']
    c, f, d, h, w = get_example_shape(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         depth=d,
                         channels=c,
                         frames=f,)
    return VAEExperiment(model,
                         params=exp_params)


def rl2d(config: dict, run_args: dict) -> VAEExperiment:
    run_config = config['run_params']['config']
    env = get_env(run_config['env_config']['name'])
    run_config = {**run_config,
                  'framework': 'torch',
                  'env': env}
    config['run_params']['config'] = run_config
    tune.run(config['algorithm'],
             loggers=[TBXLogger],
             **config['run_params'])


def comparison(config: dict, run_args: dict) -> None:
    metrics = config['plot']['metrics']
    results = {}
    for path in config['series']:
        experiment = experiment_main(load_config(path), **run_args)
        path = os.path.join(experiment.logger.save_dir,
                            experiment.logger.name,
                            'version_' + str(experiment.logger.version),
                            'metrics.csv')
        with open(path, 'r') as f:
            f_hdr = f.readline().strip().split(',')
            inds = []
            for metric in metrics:
                try:
                    inds.append(f_hdr.index(metric))
                except:
                    inds.append(None)
            cols = []
            for line in f:
                line = line.strip().split(',')
                for ind in inds:
                    cols.append(line[ind] if ind != None else None)
            results[experiment.logger.name] = cols
    for i, metric in enumerate(metrics):
        items = []
        for name, cols in results.items():
            if cols[i] == None:
                # Metric not available for this experiment
                continue
            items.append((name, cols))
        plot_comparison(items,
                        metric_name=metric,
                        save_dir=run_args['save_dir'])


entrypoints = {
    'classification2d': classification2d,
    'classification_embed2d': classification_embed2d,
    'classification_sandwich2d': classification_sandwich2d,
    'comparison': comparison,
    'rl2d': rl2d,
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
    'vae4d': vae4d,
}


def create_experiment(config: dict, run_args: dict) -> pl.LightningModule:
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
                    smoke_test: bool) -> pl.LightningModule:
    manual_seed = config.get('manual_seed', 100)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    experiment = create_experiment(config, run_args=dict(
        save_dir=save_dir,
        exp_no=exp_no,
        total_experiments=total_experiments,
        smoke_test=smoke_test,
    )).cuda()
    tt_logger = TestTubeLogger(save_dir=save_dir,
                               name=config['logging_params']['name'],
                               debug=False,
                               create_git_tag=False)
    if smoke_test:
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


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
