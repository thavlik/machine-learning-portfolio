import yaml
import argparse
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from models import BasicVAE
from basic_experiment import BasicExperiment


def vae(config: dict,
        dataset_name: str,
        dataset_params: dict):
    model = BasicVAE(**config['model_params'])
    return BasicExperiment(model,
                           dataset_name=dataset_name,
                           dataset_params=dataset_params,
                           params=config['exp_params'])


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset(path):
    ds = load_config(path)
    return ds['name'], ds['params']


experiments = {
    'vae': vae,
}


def experiment_main(config: dict,
                    dataset_name: str,
                    dataset_params: dict,
                    save_dir: str):
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
        save_dir=save_dir,
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
                    default='../experiments/vae/basic_mse.yaml')
parser.add_argument('--dataset', '-d',
                    dest="dataset",
                    metavar='DATASET',
                    help='path to the dataset config file',
                    default='../data/doom/single_frame.yaml')
parser.add_argument('--save-dir',
                    dest="save_dir",
                    metavar='SAVE_DIR',
                    help='Save directory for logs and screenshots',
                    default='../logs')

args = parser.parse_args()
config = load_config(args.config)
dataset_name, dataset_params = load_dataset(args.dataset)
cudnn.deterministic = True
cudnn.benchmark = False
experiment_main(config,
                dataset_name,
                dataset_params,
                save_dir=args.save_dir)
