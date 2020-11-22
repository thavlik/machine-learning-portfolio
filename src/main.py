import yaml
import argparse
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from models import BasicVAE
from basic_experiment import BasicExperiment


def basic_adversarial(config):
    raise NotImplementedError


def basic_mse(config):
    model = BasicVAE(**config['model_params'])
    return BasicExperiment(model,
                           params=config['exp_params'])


def basic_fid(config):
    raise NotImplementedError


def dataset_purifier(config):
    raise NotImplementedError


def temporal_discriminator(config):
    raise NotImplementedError


def temporal_distance(config):
    raise NotImplementedError


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


experiments = {
    'basic_adversarial': basic_adversarial,
    'basic_fid': basic_fid,
    'basic_mse': basic_mse,
    'dataset_purifier': dataset_purifier,
    'temporal_discriminator': temporal_discriminator,
    'temporal_distance': temporal_distance,
}


def experiment_main(config):
    torch.manual_seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    entrypoint = config['entrypoint']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    fn = experiments.get(entrypoint, None)
    assert fn != None, f"unknown entrypoint '{entrypoint}'"
    experiment = fn(config).to(device)
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
    )
    runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     log_save_interval=100,
                     train_percent_check=1.,
                     val_percent_check=1.,
                     num_sanity_val_steps=5,
                     early_stop_callback=False,
                     check_val_every_n_epoch=args.log_epoch,
                     **config['trainer_params'])
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    return experiment


parser = argparse.ArgumentParser(
    description='Doom VAE training entrypoint')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='../configs/basic_mse.yaml')
parser.add_argument('--log-epoch',
                    dest="log_epoch",
                    metavar='LOG_EPOCH',
                    help='number of epochs per validation pass',
                    default=200)
args = parser.parse_args()
config = load_config(args.filename)
cudnn.deterministic = True
cudnn.benchmark = False
experiment_main(config)
