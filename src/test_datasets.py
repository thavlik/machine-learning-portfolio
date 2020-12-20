import argparse
import sys
from tqdm import tqdm
from dataset import get_dataset

parser = argparse.ArgumentParser(
    description='test dataset')
parser.add_argument('--dataset',
                    dest="dataset",
                    metavar='DATASET',
                    help='dataset name',
                    default='deeplesion')
args = parser.parse_args()

opts = {
    #'rsna-intracranial': {
    #    'root': 'E:/rsna-intracranial',
    #    'download': False,
    #    'use_gzip': False,
    #},
    'rsna-intracranial': {
        'root': '/data/rsna-ich',
        'download': True,
        'use_gzip': True,
    },
    'deeplesion': {
        'root': 'E:/deeplesion',
        'download': False,
    },  
}

for train in [False, True]:
    ds = get_dataset(args.dataset,
                     opts[args.dataset],
                     train=train,
                     safe=False)
    n = len(ds)
    bad_indices = []
    for i in tqdm(range(n)):
        try:
            #path = os.path.join(ds.root, ds.files[i] + '.gz')
            #with gzip.open(path) as f:
            #    contents = f.read()
            ex = ds[i]
        except KeyboardInterrupt:
            raise
        except:
            bad_indices.append(i)
            print(f'Encountered bad index ({i}):')
            print(sys.exc_info())
        #if i % 50 == 0:000
        #    print(f'[{i}/{n}] checked')
    print('Bad indices:')
    print(bad_indices)
