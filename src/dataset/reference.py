import torch.utils.data as data
from torchvision import datasets

class ReferenceDataset(data.Dataset):
    """ Reference dataset loader that proxies a torchvision dataset
    """
    def __init__(self,
                 **params: dict):
        super(ReferenceDataset, self).__init__()
        self.ds = getattr(datasets, params['name'])(**params['params'])

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)
