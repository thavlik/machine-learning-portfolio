import os
import math
import torch
import numpy as np
from torch import optim, Tensor
from torchvision import transforms
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import nn
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
from models.classifier import Classifier
from merge_strategy import strategy


class ClassificationExperiment(pl.LightningModule):

    def __init__(self,
                 model: Classifier,
                 params: dict) -> None:
        super(ClassificationExperiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None

        if 'plot' in self.params:
            plots = self.params['plot']
            if type(plots) is not list:
                plots = [plots]
            self.plots = plots
        else:
            self.plots = []

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.forward(real_img).cpu()
        train_loss = self.model.loss_function(y, labels.cpu())
        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.forward(real_img).cpu()
        val_loss = self.model.loss_function(y, labels.cpu())
        return val_loss

    def validation_epoch_end(self, outputs: dict):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)

    def sample_images(self):
        for plot, val_indices in zip(self.plots, self.val_indices):
            test_input = []
            recons = []
            batch = torch.cat([self.sample_dataloader.dataset[i][0].unsqueeze(0)
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
                                    f"{self.logger.name}_{plot['fn']}_{self.current_epoch}")
            orig = test_input.data.cpu()
            recons = recons.data.cpu()
            fn = get_plot_fn(plot['fn'])
            fn(orig=orig,
               recons=recons,
               model_name=self.model.name,
               epoch=self.current_epoch,
               out_path=out_path,
               **plot['params'])

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
                          **self.params['data'].get('loader', {}))

    def val_dataloader(self):
        ds_params = strategy.merge(
            self.params['data'].get('training', {}).copy(),
            self.params['data'].get('validation', {}))
        dataset = get_dataset(self.params['data']['name'], ds_params)

        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            **self.params['data'].get('loader', {}))
        self.num_val_imgs = len(self.sample_dataloader)
        n = len(dataset)
        # Persist separate validation indices for each plot
        self.val_indices = [torch.randint(low=0,
                                          high=n,
                                          size=(plot['batch_size'], 1)).squeeze()
                            for plot in self.plots]
        return self.sample_dataloader
