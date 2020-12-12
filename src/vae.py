import os
import math
import gc
import time
import torch
import numpy as np
from torch import optim, Tensor
from torchvision import transforms
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor
import pytorch_lightning as pl
from dataset import get_dataset, BatchVideoDataLoader
from abc import abstractmethod
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from plot import get_plot_fn
from models.base import BaseVAE
from merge_strategy import strategy
from torch.nn.parallel import DistributedDataParallel
from ray import tune


class VAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 enable_tune: bool = False) -> None:
        super(VAEExperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.enable_tune = enable_tune

        plots = self.params['plot']
        if type(plots) is not list:
            plots = [plots]
        self.plots = plots

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def get_lod(self) -> Tuple[int, float]:
        schedule = self.params['progressive_growing']
        for i, step in enumerate(schedule):
            if self.trainer.global_step >= step:
                lod = len(schedule) - i - 1
                next_step = schedule[i+1]
                alpha = (self.trainer.global_step - step) / (next_step - step)
                return lod, alpha
        return (0, 0.0)

    def training_step(self, *args, **kwargs):
        train_loss, _ = self.training_step_raw(*args, **kwargs)
        return train_loss

    def training_step_raw(self,
                          batch,
                          batch_idx,
                          optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        results = self.forward(real_img, labels=labels)
        kwargs = dict(optimizer_idx=optimizer_idx,
                      batch_idx=batch_idx,
                      kld_weight=self.params.get('kld_weight', 0.0) *
                      self.params['batch_size']/self.num_train_imgs)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        train_loss = self.model.loss_function(*results, **kwargs)
        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})
        if self.enable_tune:
            tune.report(**{key: val.item()
                           for key, val in train_loss.items()})
        if self.global_step > 0:
            for plot, val_indices in zip(self.plots, self.val_indices):
                if self.global_step % plot['sample_every_n_steps'] == 0:
                    self.sample_images(plot, val_indices)
        return train_loss, results

    def validation_step(self, *args, **kwargs):
        val_loss, _ = self.validation_step_raw(*args, **kwargs)
        return val_loss
        
    def validation_step_raw(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        kwargs = dict()
        if 'progressive_growing' in self.params:
            kwargs['lod'], kwargs['alpha'] = self.get_lod()
        results = self.forward(real_img, labels=labels, **kwargs)
        kwargs = dict(optimizer_idx=optimizer_idx,
                      batch_idx=batch_idx,
                      kld_weight=self.params.get('kld_weight', 0.0) *
                      self.params['batch_size']/self.num_val_imgs,
                      **kwargs)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        val_loss = self.model.loss_function(*results, **kwargs)
        return val_loss, results

    def validation_epoch_end(self, outputs: dict):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        if self.enable_tune:
            tune.report(avg_val_loss=avg_loss)

    def sample_images(self, plot: dict, val_indices: Tensor):
        revert = self.training
        if revert:
            self.eval()
        test_input = []
        recons = []
        batch = torch.cat([self.sample_dataloader.dataset[int(i)][0].unsqueeze(0)
                           for i in val_indices], dim=0).to(self.curr_device)
        for x in batch:
            x = x.unsqueeze(0)
            test_input.append(x)
            x = self.model.generate(x, labels=[])
            recons.append(x)
        test_input = torch.cat(test_input, dim=0)
        recons = torch.cat(recons, dim=0)
        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        orig = test_input.data.cpu()
        recons = recons.data.cpu()
        fn = get_plot_fn(plot['fn'])
        fn(orig=orig,
           recons=recons,
           model_name=self.model.name,
           epoch=self.current_epoch,
           out_path=out_path,
           **plot['params'])
        gc.collect()
        if revert:
            self.train()

    def configure_optimizers(self):
        optims = [optim.Adam(self.model.parameters(),
                             **self.params['optimizer'])]
        scheds = []
        return optims, scheds

    def optimizer_step(self,
                       epoch: int,
                       batch_idx: int,
                       optimizer: Optimizer,
                       optimizer_idx: int,
                       optimizer_closure: Optional[Callable],
                       on_tpu: bool,
                       using_native_amp: bool,
                       using_lbfgs: bool) -> None:
        # warm up lr, linear ramp
        warmup_steps = self.params.get('warmup_steps', 0)
        if warmup_steps > 0 and self.trainer.global_step < warmup_steps:
            lr_scale = min(1.0, float(
                self.trainer.global_step + 1) / float(warmup_steps))
            lr = lr_scale * self.params['optimizer']['lr']
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        ds_params = self.params['data'].get('training', {})
        dl_params = self.params['data'].get('loader', {})
        if self.params['data']['name'] == 'batch-video':
            vl = BatchVideoDataLoader(**ds_params,
                                      batch_size=self.params['batch_size'],
                                      **dl_params)
            self.num_train_imgs = len(vl)
            return vl
        dataset = get_dataset(self.params['data']['name'], ds_params)
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          **dl_params)

    def val_dataloader(self):
        ds_params = strategy.merge(
            self.params['data'].get('training', {}).copy(),
            self.params['data'].get('validation', {}))
        dl_params = self.params['data'].get('loader', {})
        if self.params['data']['name'] == 'batch-video':
            vl = BatchVideoDataLoader(**ds_params,
                                      batch_size=self.params['batch_size'],
                                      **dl_params)
            self.num_val_imgs = len(vl)
            self.val_indices = None
            return vl
        dataset = get_dataset(self.params['data']['name'], ds_params)
        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            **dl_params)
        self.num_val_imgs = len(self.sample_dataloader)
        n = len(dataset)
        # Persist separate validation indices for each plot
        self.val_indices = [torch.randint(low=0,
                                          high=n,
                                          size=(plot['batch_size'], 1)).squeeze()
                            for plot in self.plots]
        return self.sample_dataloader
