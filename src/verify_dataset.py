import argparse
import sys
from tqdm import tqdm
from dataset import get_dataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test dataset')
    parser.add_argument('--dataset',
                        dest="dataset",
                        metavar='DATASET',
                        help='dataset name',
                        default='deeplesion')
    parser.add_argument('--root',
                        dest="root",
                        metavar='ROOT',
                        help='dataset root dir',
                        default=None)
    parser.add_argument('--num-workers',
                        dest="num_workers",
                        metavar='NUM_WORKERS',
                        help='number of worker processes for the data loader',
                        default=cpu_count())
    args = parser.parse_args()

    opts = {
        #'rsna-intracranial': {
        #'root': args.root or 'E:/rsna-intracranial',
        #'download': False,
        #'use_gzip': False,
        #},
        'rsna-intracranial': {
            'root': '/data/rsna-ich',
            'download': True,
            'use_gzip': True,
        },
        'deeplesion': {
            'root': args.root or 'E:/deeplesion',
            'download': False,
        },
    }

    print(f'Verifying {args.dataset} with {args.num_workers} workers')

    for train in [False, True]:
        ds = get_dataset(args.dataset,
                        opts[args.dataset],
                        train=train,
                        safe=False)
        loader = DataLoader(ds,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=False)
        n = len(loader)
        it = iter(loader)
        bad_indices = []
        for i in tqdm(range(n)):
            try:
                batch = next(it)
            except KeyboardInterrupt:
                raise
            except:
                bad_indices.append(i)
                print(f'Encountered bad index ({i}):')
                print(sys.exc_info())
        print('Bad indices:')
        print(bad_indices)
