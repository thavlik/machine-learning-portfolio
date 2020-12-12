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
from models import create_model, BaseRenderer

class NeuralGBufferExperiment(pl.LightningModule):
    def __init__(self,
                 model: BaseRenderer,
                 params: dict) -> None:
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        pass

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        pass

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
        self.val_indices = [torch.randint(low=0,
                                          high=n,
                                          size=(plot['batch_size'], 1)).squeeze()
                            for plot in self.plots]
        return self.sample_dataloader


def neural_gbuffer(config: dict, run_args: dict) -> pl.LightningModule:
    exp_params = config['exp_params']
    model = create_model(**config['model_params'],
                         width=512,
                         height=512,
                         channels=3)
    return NeuralGBufferExperiment(model, exp_params)
