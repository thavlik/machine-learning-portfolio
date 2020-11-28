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
        if 'base' in config:
            bases = config['base']
            if type(bases) is not list:
                bases = [bases]
            strategy = Merger([(list, "override"),
                               (dict, "merge")],
                              ["use_existing"],
                              ["use_existing"])
            for base in bases:
                base = load_config(base)
                config = strategy.merge(base, config)
        return config


def experiment_main(config: dict,
                    save_dir: str):
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
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    return experiment


parser = argparse.ArgumentParser(
    description='thavlik portfolio entrypoint')
parser.add_argument('--config',  '-c',
                    dest="config",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='experiments/rsna-intracranial/vae_fid.yaml')
parser.add_argument('--save-dir',
                    dest="save_dir",
                    metavar='SAVE_DIR',
                    help='Save directory for logs and screenshots',
                    default='logs')

args = parser.parse_args()
config = load_config(args.config)
cudnn.deterministic = True
cudnn.benchmark = False
experiment_main(config,
                save_dir=args.save_dir)
