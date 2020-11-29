import yaml
import argparse
import gc
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models import create_model
from entrypoints import create_experiment
from merge_strategy import strategy

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        if 'series' in config:
            return [load_config(item)
                    for item in config['series']]
        if 'base' in config:
            bases = config['base']
            if type(bases) is not list:
                bases = [bases]
            merged = {}
            for base in bases:
                merged = strategy.merge(merged, load_config(base))
            config = strategy.merge(merged, config)
        return config


def experiment_main(config: dict,
                    save_dir: str,
                    exp_no: int,
                    total_experiments: int,
                    dry_run: bool):
    torch.manual_seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    experiment = create_experiment(config).cuda()
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
    print(f"======= Training {config['model_params']['name']}/{config['logging_params']['name']} (Experiment {exp_no+1}/{total_experiments}) =======")
    print(config)
    runner.fit(experiment)


def count_experiments(series: list) -> int:
    n = 0
    for item in series:
        if type(item) is list:
            n += count_experiments(item)
        else:
            n += 1
    return n


def run_series(series: list,
               save_dir: str,
               exp_no: int,
               total_experiments: int,
               dry_run: bool):
    if type(series) != list:
        series = [series]
    for item in series:
        if type(item) is list:
            exp_no = run_series(item, save_dir, exp_no, total_experiments, dry_run)
        else:
            experiment_main(item, save_dir, exp_no, total_experiments, dry_run)
            gc.collect()
            exp_no += 1
    return exp_no


parser = argparse.ArgumentParser(
    description='thavlik portfolio entrypoint')
parser.add_argument('--config',  '-c',
                    dest="config",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='experiments/reference/mnist/vae_fid.yaml')
parser.add_argument('--save-dir',
                    dest="save_dir",
                    metavar='SAVE_DIR',
                    help='save directory for logs and screenshots',
                    default='logs')
parser.add_argument('--dry-run',
                    dest="dry_run",
                    metavar='DRY_RUN',
                    help='dry run mode (stop after a couple steps)',
                    default=False)
args = parser.parse_args()

if args.dry_run:
    print('Executing dry run - training will stop after one step.')

config = load_config(args.config)
cudnn.deterministic = True
cudnn.benchmark = False
total_experiments = count_experiments(config)
run_series(config,
           save_dir=args.save_dir,
           exp_no=0,
           total_experiments=total_experiments,
           dry_run=args.dry_run)
