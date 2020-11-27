import yaml
import argparse
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from models import models
from vae import VAEExperiment
from dataset import ReferenceDataset

dataset_dims = {
    'eeg': (1, 8192),
    'cq500': (1, 512, 512),
    'deeplesion': (1, 512, 512),
    'rsna-intracranial': (1, 512, 512),
    'trends-fmri': (53, 63, 52, 53),
}


def get_example_shape(dataset: dict):
    loader = dataset['loader']
    if loader == 'reference':
        return ReferenceDataset(**dataset['params'])[0].shape
    if loader == 'video':
        params = dataset['training']
        return (3, params['height'], params['width'])
    if loader not in dataset_dims:
        raise ValueError(f'unknown dataset "{loader}"')
    return dataset_dims[loader]


def create_model(model_name, **kwargs):
    if model_name not in models:
        raise ValueError(f'unknown model "{model_name}"')
    return models[model_name](**kwargs)


def vae1d(config: dict,
          dataset: dict):
    c, l = get_example_shape(dataset)
    model = create_model(config['model_params']['name'],
                         **config['model_params'],
                         length=l,
                         channels=c,
                         enable_fid='fid_weight' in exp_params)
    exp_params = config['exp_params']
    return VAEExperiment(model,
                         params=exp_params,
                         dataset=dataset)


def vae2d(config: dict,
          dataset: dict):
    c, h, w = get_example_shape(dataset)
    exp_params = config['exp_params']
    model = create_model(config['model_params']['name'],
                         **config['model_params'],
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
    model = create_model(config['model_params']['name'],
                         **config['model_params'],
                         width=w,
                         height=h,
                         depth=d,
                         channels=c)
    exp_params = config['exp_params']
    return VAEExperiment(model,
                         params=exp_params,
                         dataset=dataset)


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


experiments = {
    'vae1d': vae1d,
    'vae2d': vae2d,
    'vae3d': vae3d,
}


def experiment_main(config: dict,
                    dataset: dict,
                    save_dir: str):
    torch.manual_seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    entrypoint = config['entrypoint']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if entrypoint not in experiments:
        raise ValueError(f"unknown entrypoint '{entrypoint}'")
    experiment = experiments[entrypoint](config, dataset).cuda()
    tt_logger = TestTubeLogger(
        save_dir=save_dir,
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     num_sanity_val_steps=5,
                     logger=tt_logger,
                     **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    return experiment


parser = argparse.ArgumentParser(
    description='Doom VAE training entrypoint')
parser.add_argument('--config',  '-c',
                    dest="config",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='../experiments/vae2d/basic_mse.yaml')
parser.add_argument('--dataset', '-d',
                    dest="dataset",
                    metavar='DATASET',
                    help='path to the dataset config file',
                    default='../data/rsna_intracranial.yaml')
parser.add_argument('--save-dir',
                    dest="save_dir",
                    metavar='SAVE_DIR',
                    help='Save directory for logs and screenshots',
                    default='../logs')

args = parser.parse_args()
config = load_config(args.config)
dataset = load_config(args.dataset)
cudnn.deterministic = True
cudnn.benchmark = False
experiment_main(config,
                dataset=dataset,
                save_dir=args.save_dir)
