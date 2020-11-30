import argparse
import gc
import torch
import torch.backends.cudnn as cudnn
from models import create_model
from entrypoints import experiment_main
from load_config import load_config


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
               smoke_test: bool):
    if type(series) != list:
        series = [series]
    for item in series:
        if type(item) is list:
            exp_no = run_series(item,
                                save_dir=save_dir,
                                exp_no=exp_no,
                                total_experiments=total_experiments,
                                smoke_test=smoke_test)
        else:
            experiment_main(item,
                            save_dir=save_dir,
                            exp_no=exp_no,
                            total_experiments=total_experiments,
                            smoke_test=smoke_test)
            gc.collect()
            exp_no += 1
    return exp_no


parser = argparse.ArgumentParser(
    description='thavlik portfolio entrypoint')
parser.add_argument('--config',  '-c',
                    dest="config",
                    metavar='FILE',
                    help='path to the experiment config file',
                    default='experiments/all.yaml')
parser.add_argument('--save-dir',
                    dest="save_dir",
                    metavar='SAVE_DIR',
                    help='save directory for logs and screenshots',
                    default='logs')
parser.add_argument('--smoke-test',
                    dest="smoke_test",
                    metavar='DRY_RUN',
                    help='smoke test mode (stop after a couple steps)',
                    default=False)
args = parser.parse_args()

if args.smoke_test:
    print('Executing smoke test - training will stop after a couple steps.')

config = load_config(args.config)
cudnn.deterministic = True
cudnn.benchmark = False
total_experiments = count_experiments(config)
run_series(config,
           save_dir=args.save_dir,
           exp_no=0,
           total_experiments=total_experiments,
           smoke_test=args.smoke_test)
print(f"============== Completed ==============")
