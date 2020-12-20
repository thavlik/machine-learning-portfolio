import argparse
import sys

from dataset import get_dataset

parser = argparse.ArgumentParser(
    description='test dataset')
parser.add_argument('--dataset',
                    dest="dataset",
                    metavar='DATASET',
                    help='dataset name',
                    default='rsna-intracranial')
args = parser.parse_args()

for train in [False, True]:
    ds = get_dataset(args.dataset, {
        'root': 'E:/rsna-intracranial',
        'download': True,
        'use_gzip': True,
    }, train=train, safe=False)
    n = len(ds)
    bad_indices = []
    for i in range(n):
        try:
            ex = ds[i]
        except KeyboardInterrupt:
            raise
        except:
            bad_indices.append(i)
            print(f'Encountered bad index ({i}):')
            print(sys.exc_info()[0])
        if i % 1000 == 0:
            print(f'[{i}/{n}] checked')
    print('Bad indices:')
    print(bad_indices)
