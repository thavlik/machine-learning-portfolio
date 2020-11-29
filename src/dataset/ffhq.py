import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor


class FFHQDataset(data.Dataset):
    def __init__(self,
                 dir: str):
        super(FFHQDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
