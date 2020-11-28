import os
import math
import torch
import numpy as np
from torch import optim, Tensor
from torchvision import transforms
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor
import pytorch_lightning as pl
from dataset import get_dataset
from abc import abstractmethod
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import Callable, Optional
from plot import get_plot_fn
from models.base import BaseVAE


class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

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

    def validation_epoch_end(self, outputs: dict):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.sample_dataloader))
        test_input = test_input[:8]
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, labels=[])
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"recons_{self.logger.name}_{self.current_epoch}.png")
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable],
        on_tpu: bool,
        using_native_amp: bool,
        using_lbfgs: bool,
    ) -> None:
        # warm up lr, linear ramp
        warmup_steps = self.params.get('warmup_steps', 0)
        if warmup_steps > 0 and self.trainer.global_step < warmup_steps:
            lr_scale = min(1.0, float(
                self.trainer.global_step + 1) / float(warmup_steps))
            lr = lr_scale * self.params['optimizer']['lr']
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            self.log('lr', lr, prog_bar=True)

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        dataset = get_dataset(self.params['data']['name'],
                              self.params['data'].get('training', {}))
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.dataset.get('num_workers', 0))

    def val_dataloader(self):
        dataset = get_dataset(self.params['data']['name'], {
            **self.params['data'].get('training', {}),
            **self.params['data'].get('validation', {}),
        })
        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            drop_last=True,
                                            **self.params['data'].get('loader', {}))
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader
