import os
import torch
from models import create_model
from vae import VAEExperiment
from classification import ClassificationExperiment
from localization import LocalizationExperiment
from dataset import ReferenceDataset, get_example_shape, get_output_features
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import numpy as np
from pytorch_lightning import Trainer
from load_config import load_config
from ray import tune
from ray.tune.logger import TBXLogger
from ray.rllib.models import ModelCatalog
from env import get_env
from plot import plot_comparison
import pandas as pd
from merge_strategy import strategy
from neural_gbuffer import neural_gbuffer


def localization2d(config: dict, run_args: dict) -> LocalizationExperiment:
    exp_params = config['exp_params']
    c, h, w = get_example_shape(exp_params['data'])
    num_output_features = get_output_features(exp_params['data'])
    model = create_model(**config['model_params'],
                         width=w,
                         height=h,
                         channels=c,
                         num_output_features=num_output_features)
    return LocalizationExperiment(model,
                                  params=exp_params)


def classification(config: dict, run_args: dict) -> ClassificationExperiment:
    exp_params = config['exp_params']
    model = create_model(**config['model_params'],
                         input_shape=get_example_shape(exp_params['data']))
    return ClassificationExperiment(model,
                                    params=exp_params)


def classification_embed2d(config: dict, run_args: dict) -> ClassificationExperiment:
    base_experiment = experiment_main(
        load_config(config['base_experiment']), run_args)
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
        load_config(config['base_experiment']), run_args)
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
                         params=exp_params,
                         enable_tune=run_args.get('enable_tune', False))


def recurse(config: dict, template: dict):
    if 'uniform' in config:
        return tune.uniform(lower=config['uniform']['lower'],
                            upper=config['uniform']['upper'])
    if 'choice' in config:
        return tune.choice(config['choice'])
    if 'grid_search' in config:
        return tune.grid_search(config['grid_search'])
    for key in template:
        if not key in config:
            continue
        if type(config[key]) == dict:
            config[key] = recurse(config[key], template[key])
    return config


def generate_run_config(config: dict):
    template = {
        'exp_params': {
            'warmup_steps': int,
            'batch_size': int,
            'optimizer': {
                'lr': float,
                'weight_decay': float,
            },
        },
    }
    return recurse(config.copy(), template)


def get_best_config(analysis,
                    metric: str = 'loss',
                    scope: str = 'last-5-avg') -> dict:
    """ Retrieves the best config from a tune hyperparameter
    search by averaging the metrics of all matching samples.
    This is more sophisticated than analysis.get_best_config,
    which only considers individual trials and does not do
    any averaging.
    """
    options = []
    for trial in analysis.trials:
        loss = trial.metric_analysis[metric][scope]
        found = False
        for config, losses in options:
            if config == trial.config:
                losses.append(loss)
                found = True
                break
        if not found:
            options.append((trial.config, [loss]))
    i = np.argmin([np.mean(losses)
                   for _, losses in options])
    best_config = options[i][0]
    return best_config


def hparam_search(config: dict, run_args: dict) -> VAEExperiment:
    import ray
    from ray import tune
    ray.init(num_cpus=6, num_gpus=1)
    run_config = load_config(config['experiment'])
    run_config = generate_run_config(run_config)
    run_config['trainer_params'] = strategy.merge(run_config['trainer_params'].copy(), {
        'max_steps': config.get('max_steps', 50),
        'log_every_n_steps': 1,
    })
    if config.get('randomize_seed', False):
        print('Warning: randomizing seed for each trial')
        run_config['manual_seed'] = tune.sample_from(lambda spec: np.random.randint(0, 64_000))
    analysis = tune.run(
        tune.with_parameters(experiment_main,
                             run_args=dict(**run_args,
                                           enable_tune=True)),
        name=run_config['entrypoint'],
        config=run_config,
        local_dir=run_args['save_dir'],
        num_samples=config['num_samples'],
        resources_per_trial={
            'cpu': 2,
            'gpu': 1,
        },
    )
    best_config = get_best_config(analysis)
    # Restore original trainer_params, which were overridden
    # so the hparam search is shorter than a full experiment.
    best_config['trainer_params'] = config['trainer_params']
    experiment_main(best_config, run_args)
    #exp_params = best_config['exp_params']
    #c, l = get_example_shape(exp_params['data'])
    # model = create_model(**best_config['model_params'],
    #                     num_samples=l,
    #                     channels=c)
    # return VAEExperiment(model,
    #                     params=exp_params)


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
                         frames=f)
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
    plot = config['plot']
    metrics = plot['metrics']
    results = {}
    num_samples = config.get('num_samples', 1)
    for _ in range(num_samples):
        for path in config['series']:
            experiment = experiment_main(load_config(path), run_args)
            path = os.path.join(experiment.logger.save_dir,
                                experiment.logger.name,
                                'version_' + str(experiment.logger.version),
                                'metrics.csv')
            with open(path, 'r') as f:
                f_hdr = f.readline().strip().split(',')
                metric_col = []
                for metric in metrics:
                    try:
                        metric_col.append(f_hdr.index(metric))
                    except:
                        # This metric is not available in this experiment
                        metric_col.append(None)
                metric_data = []
                for step, line in enumerate(f):
                    line = line.strip().split(',')
                    for idx in metric_col:
                        if idx is None:
                            # Metric not available
                            continue
                        col = line[idx]
                        if col == '':
                            # Metric was not logged on this step
                            continue
                        metric_data.append((step, float(col)))
                if metric in results:
                    results[metric].append(
                        (experiment.logger.name, metric_data))
                else:
                    results[metric] = [(experiment.logger.name, metric_data)]
    dir = os.path.join(run_args['save_dir'], config['name'])
    if os.path.exists(dir):
        version_no = len([f for f in os.listdir(dir)
                          if f.startswith('version_')])
    else:
        os.makedirs(dir)
        version_no = 0
    out_dir = os.path.join(dir, f'version_{version_no}')
    os.mkdir(out_dir)
    torch.save(results, os.path.join(out_dir, 'metrics.pt'))
    for metric, data in results.items():
        exps = {}
        for name, data in data:
            if name in exps:
                exps[name].append(data)
            else:
                exps[name] = [data]
        means = {}
        for name, data in exps.items():
            means[name] = np.mean(data, axis=0)
        plot_comparison(means,
                        metric_name=metric,
                        num_samples=num_samples,
                        width=plot['width'],
                        height=plot['height'],
                        out_path=os.path.join(out_dir,
                                              f'comparison_{metric}.png'),
                        scatter_params=plot.get('scatter_params', {}),
                        layout_params=plot.get('layout_params', {}))


def comparison_tune(config: dict, run_args: dict) -> None:
    import ray
    from ray.util.sgd.integration.torch import DistributedTrainableCreator
    ray.init(**config.get('ray_options', {}))
    distributed_train_cifar = DistributedTrainableCreator(
        my_trainable,
        use_gpu=True,
        num_workers=4,
        num_workers_per_host=4,
        num_gpus_per_worker=1,
        num_cpus_per_worker=1,
    )
    tune.run(
        distributed_train_cifar,
        resources_per_trial=None,
        config=config,
        num_samples=num_samples,
    )


entrypoints = {
    'classification': classification,
    'classification_embed2d': classification_embed2d,
    'classification_sandwich2d': classification_sandwich2d,
    'comparison': comparison,
    'localization2d': localization2d,
    'rl2d': rl2d,
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
    'vae4d': vae4d,
    'hparam_search': hparam_search,
    'neural_gbuffer': neural_gbuffer,
}


def create_experiment(config: dict, run_args: dict) -> pl.LightningModule:
    if 'entrypoint' not in config:
        raise ValueError('config has no entrypoint')
    entrypoint = config['entrypoint']
    if entrypoint not in entrypoints:
        raise ValueError(f'unknown entrypoint "{entrypoint}" '
                         f'valid options are {entrypoints}')
    return entrypoints[entrypoint](config, run_args)


def experiment_main(config: dict, run_args: dict) -> pl.LightningModule:
    torch.set_num_threads(run_args['num_threads'])
    manual_seed = config.get('manual_seed', 100)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    experiment = create_experiment(config, run_args)
    if experiment is None:
        return
    experiment.hparams = config
    experiment = experiment.cuda()
    tt_logger = TestTubeLogger(save_dir=run_args['save_dir'],
                               name=config['logging_params']['name'],
                               debug=False,
                               create_git_tag=False)
    tt_logger.log_hyperparams(config)
    if run_args['smoke_test']:
        config['trainer_params']['max_steps'] = 5
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='loss',
        mode='min'
    )
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     num_sanity_val_steps=5,
                     logger=tt_logger,
                     checkpoint_callback=checkpoint_callback,
                     **config['trainer_params'])
    print(
        f"======= Training {config['model_params']['name']}/{config['logging_params']['name']} (Experiment {run_args['exp_no']+1}/{run_args['total_experiments']}) =======")
    print(config)
    runner.fit(experiment)
    return experiment


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
