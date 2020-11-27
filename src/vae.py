import math
import torch
import numpy as np
from torch import optim, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import pytorch_lightning as pl
from models.base import BaseVAE
from utils import data_loader
from dataset import get_dataset
from abc import abstractmethod
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor


def resize2d(x,
             width: int,
             height: int):
    to_pil = ToPILImage()
    resize = Resize((height, width))
    to_tensor = ToTensor()
    return to_tensor(resize(to_pil(torch.Tensor(x))))


def add_fig1d(orig: Tensor,
              recons: Tensor,
              fig: Figure,
              row: int,
              col: int):
    pass


def plot1d(recons: Tensor,
           orig: Tensor,
           out_path: str,
           params: dict):
    rows = params['rows']
    cols = params['cols']
    fig = make_subplots(rows=rows, cols=cols)
    raise NotImplementedError
    return fig


def add_fig2d(orig: Tensor,
              recons: Tensor,
              fig: Figure,
              row: int,
              col: int):
    pass


def plot2d(recons: Tensor,
           orig: Tensor,
           out_path: str,
           params: dict):
    rows = params['rows']
    cols = params['cols']
    if 'thumbnail_size' in params:
        thumbnail_width = params['thumbnail_size']
        thumbnail_height = params['thumbnail_size']
    else:
        thumbnail_width = params.get('thumbnail_width', 512)
        thumbnail_height = params.get('thumbnail_height', 256)
    scaling = params.get('scaling', 2.0)
    fig = plt.figure(figsize=(cols * scaling, rows * scaling))
    grid = ImageGrid(fig,
                     111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1)  # pad between axes in inch.
    i = 0
    n = min(rows * cols, orig.shape[0])
    to_pil = ToPILImage()
    for _ in range(rows):
        done = False
        for _ in range(cols):
            if i >= n:
                done = True
                break
            img = torch.cat([orig[i], recons[i]], dim=-1)
            if img.shape[-2] != (thumbnail_height, thumbnail_width):
                img = resize2d(img, thumbnail_width, thumbnail_height)
            grid[i].imshow(to_pil(img))
            i += 1
        if done:
            break
    fig.savefig(out_path)


plot_fn = {
    'plot1d': plot2d,
    'plot2d': plot2d,
}


class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 dataset: dict) -> None:
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.dataset = dataset

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)

        results = self.forward(real_img, labels=labels)
        kld_weight = self.params.get('kld_weight', 0.0) * \
            self.params['batch_size']/self.num_train_imgs
        kwargs = dict(optimizer_idx=optimizer_idx,
                      batch_idx=batch_idx,
                      kld_weight=kld_weight)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        train_loss = self.model.loss_function(*results, **kwargs)

        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)

        results = self.forward(real_img, labels=labels)
        kld_weight = self.params.get('kld_weight', 0.0) * \
            self.params['batch_size']/self.num_val_imgs
        val_loss = self.model.loss_function(*results,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx,
                                            kld_weight=kld_weight,
                                            fid_weight=self.params.get('fid_weight', 0.0))

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.sample_dataloader))
        test_input = test_input[:8]
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, labels=[])
        out_path = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/recons_{self.logger.name}_{self.current_epoch}.png"
        orig = test_input.data.cpu()
        recons = recons.data.cpu()
        fn = plot_fn[self.params['plot']['fn']]
        fn(orig, recons, out_path, self.params['plot']['params'])
        del test_input, recons

    def configure_optimizers(self):
        optims = [optim.Adam(self.model.parameters(),
                             **self.params['optimizer'])]
        scheds = []
        return optims, scheds

    @data_loader
    def train_dataloader(self):
        dataset = get_dataset(self.dataset['loader'],
                              self.dataset.get('training', {}))
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.dataset.get('num_workers', 0))

    @data_loader
    def val_dataloader(self):
        dataset = get_dataset(self.dataset['loader'], {
            **self.dataset.get('training', {}),
            **self.dataset.get('validation', {}),
        })
        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=self.dataset.get('num_workers', 0))
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader


if __name__ == '__main__':
    plot2d(torch.rand(12, 1, 512, 512),
           torch.rand(12, 1, 512, 512),
           'plot.png',
           dict(rows=4,
                cols=4))
