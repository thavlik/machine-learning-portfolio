import os
import sys
import argparse
import time
from dataset import RSNAIntracranialDataset, normalized_dicom_pixels
import pydicom

parser = argparse.ArgumentParser(
    description='Remove corrupted DICOM')
parser.add_argument('--dir',  '-d',
                    dest="dir",
                    metavar='DIR',
                    help='path to directory containing dcm files',
                    default='E:/cq500')
parser.add_argument('--log_interval',
                    dest="log_interval",
                    metavar='LOG_INTERVAL',
                    help='how often to print updates to stdout',
                    default=1000)
args = parser.parse_args()

print(f'Removing corrupted DICOM files at {args.dir}')

start = time.time()

files = [os.path.join(dp, f)
         for dp, dn, fn in os.walk(os.path.expanduser(args.dir))
         for f in fn
         if f.endswith('.dcm')]

print(f'Processing {len(files)} files...')

num_removed = 0
for i, file in enumerate(files):
    try:
        x = pydicom.dcmread(file, stop_before_pixels=False)
        x = normalized_dicom_pixels(x)
        if x.shape != (1, 512, 512):
            raise ValueError('wrong shape')
    except:
        path = os.path.join(args.dir, file)
        os.remove(path)
        num_removed += 1
        print(f'Removed corrupted {path}: {sys.exc_info()}')
    if i > 0 and i % args.log_interval == 0:
        print(f'Processed {i}/{len(files)}')

elapsed = time.time() - start
print(f'Removed {num_removed} corrupt examples in {elapsed} seconds')
