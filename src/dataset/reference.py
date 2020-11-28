import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor

class ReferenceDataset(data.Dataset):
    """ Reference dataset that proxies a torchvision stock dataset
    """

    def __init__(self,
                 name: str,
                 params: dict):
        super(ReferenceDataset, self).__init__()
        self.ds = getattr(datasets, name)(**params)

    def __getitem__(self, index):
        x, y = self.ds[index]
        return ToTensor()(x), y

    def __len__(self):
        return len(self.ds)
