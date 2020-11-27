import os
from dataset import RSNAIntracranialDataset

ds = RSNAIntracranialDataset(dcm_path='E:/rsna-intracranial/stage_2_train',
                             s3_path='s3://rsna-intracranial/stage_2_train',
                             download=False)
n = len(ds)
total = 0
for i in range(n):
    try:
        x, _ = ds[i]
    except:
        path = os.path.join(ds.dcm_path, ds.files[i])
        os.remove(path)
        total += 1
        print(f'Removed corrupted {path}')
print(f'Removed {total} corrupt examples')