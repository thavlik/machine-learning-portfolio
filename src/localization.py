import gc
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
from models import Localizer
from merge_strategy import strategy
from typing import List


class LocalizationExperiment(pl.LightningModule):
    def __init__(self,
                 localizer: Localizer,
                 params: dict) -> None:
        super().__init__()

        self.localizer = localizer
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
        return self.localizer(input, **kwargs)

    def sample_images(self, plot: dict, val_indices: Tensor):
        revert = self.training
        if revert:
            self.eval()

        test_input = []
        predictions = []

        for class_indices in val_indices:
            batch = [self.sample_dataloader.dataset[int(i)][0]
                     for i in class_indices]
            class_input = []
            for x in batch:
                x = x.unsqueeze(0)
                class_input.append(x)
                x = self.localizer(x)
                predictions.append(x)
            class_input = torch.cat(class_input, dim=0)
            test_input.append(class_input.unsqueeze(0))

        test_input = torch.cat(test_input, dim=0).cpu()
        targets = torch.cat(targets, dim=0).cpu()
        predictions = torch.cat(predictions, dim=0).cpu()

        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        fn(test_input=test_input,
           targets=targets,
           predictions=predictions,
           baselines=torch.Tensor([0.0 for _ in range(6)]),
           out_path=out_path,
           **plot['params'])

        gc.collect()
        if revert:
            self.train()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_labels, pred_params = [y.cpu() for y in self.forward(real_img)]

        train_loss = self.localizer.loss_function([pred_labels, pred_params],
                                                  [targ_labels, targ_params],
                                                  **self.params.get('loss_params', {}))
        self.logger.experiment.log({'train/' + key: val.item()
                                    for key, val in train_loss.items()})
        # if self.global_step > 0:
        #    for plot, val_indices in zip(self.plots, self.val_indices):
        #        if self.global_step % plot['sample_every_n_steps'] == 0:
        #            self.sample_images(plot, val_indices)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_labels, pred_params = [y.cpu() for y in self.forward(real_img)]
        val_loss = self.localizer.loss_function([pred_labels, pred_params],
                                                [targ_labels, targ_params],
                                                **self.params.get('loss_params', {}))
        return val_loss

    def validation_epoch_end(self, outputs: list):
        avg = {}
        for output in outputs:
            for k, v in output.items():
                items = avg.get(k, [])
                items.append(v)
                avg[k] = items
        for metric, values in avg.items():
            self.log('val/' + metric, torch.Tensor(values).mean())

    def configure_optimizers(self):
        optims = [optim.Adam(self.localizer.parameters(),
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

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        ds_params = self.params['data'].get('training', {})
        dataset = get_dataset(self.params['data']['name'],
                              ds_params,
                              split=self.params['data'].get('split', None),
                              train=True)
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          **self.params['data'].get('loader', {}))

    def val_dataloader(self):
        ds_params = strategy.merge(
            self.params['data'].get('training', {}).copy(),
            self.params['data'].get('validation', {}))
        dataset = get_dataset(self.params['data']['name'],
                              ds_params,
                              split=self.params['data'].get('split', None),
                              train=False)
        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            **self.params['data'].get('loader', {}))
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader