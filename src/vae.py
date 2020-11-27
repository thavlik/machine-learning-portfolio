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
from plot import get_plot_fn


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
        kwargs = dict(optimizer_idx=optimizer_idx,
                      batch_idx=batch_idx,
                      kld_weight=kld_weight)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        val_loss = self.model.loss_function(*results, **kwargs)

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
        fn = get_plot_fn(self.params['plot']['fn'])
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

