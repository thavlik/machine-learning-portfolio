import yaml
import argparse
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from models import create_model
from experiments import create_experiment
from deepmerge import Merger


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
            strategy = Merger([(list, "override"),
                               (dict, "merge")],
                              ["override"],
                              ["override"])
            merged = {}
            for base in bases:
                merged = strategy.merge(merged, load_config(base))
            config = strategy.merge(merged, config)
        return config


def experiment_main(config: dict,
                    save_dir: str,
                    exp_no: int,
                    total_experiments: int):
    torch.manual_seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])
    experiment = create_experiment(config).cuda()
    tt_logger = TestTubeLogger(save_dir=save_dir,
                               name=config['logging_params']['name'],
                               debug=False,
                               create_git_tag=False)
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
               total_experiments: int):
    if type(series) != list:
        series = [series]
    exp_no = 0
    for item in series:
        if type(item) is list:
            exp_no += run_series(item, save_dir, total_experiments)
        else:
            experiment_main(item, save_dir, exp_no, total_experiments)
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
                    help='Save directory for logs and screenshots',
                    default='logs')
args = parser.parse_args()
config = load_config(args.config)
cudnn.deterministic = True
cudnn.benchmark = False
total_experiments = count_experiments(config)
run_series(config,
           save_dir=args.save_dir,
           total_experiments=total_experiments)
