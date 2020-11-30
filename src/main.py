import argparse
import gc
import time
import torch
import torch.backends.cudnn as cudnn
from models import create_model
from entry import experiment_main
from load_config import load_config
from typing import List, Union


def count_experiments(series: Union[dict, List[dict]]) -> int:
    if type(series) != list:
        series = [series]
    n = 0
    for item in series:
        if type(item) is list:
            n += count_experiments(item)
        else:
            n += 1
    return n


def run_series(series: Union[dict, List[dict]],
               exp_no: int,
               **kwargs) -> int:
    if type(series) != list:
        series = [series]
    for config in series:
        if type(config) is list:
            exp_no = run_series(config,
                                exp_no=exp_no,
                                **kwargs)
        else:
            experiment_main(config,
                            exp_no=exp_no,
                            **kwargs)
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
num_samples = config.get('num_samples', 1)
deltas = []
for i in range(num_samples):
    print(f'Running sample {i+1}/{num_samples}')
    start = time.time()
    run_series(config,
               save_dir=args.save_dir,
               exp_no=0,
               total_experiments=total_experiments,
               smoke_test=args.smoke_test)
    delta = time.time() - start
    deltas.append(deltas)
    print(f'Sample {i+1}/{num_samples} completed in {delta} seconds')
print(f'Each sample took {sum(deltas)/len(deltas)} seconds on average')
print(f"============== Completed ==============")
