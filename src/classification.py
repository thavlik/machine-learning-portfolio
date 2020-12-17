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
from models import Classifier
from merge_strategy import strategy
from typing import List
from plot import get_random_example_with_label

class ClassificationExperiment(pl.LightningModule):

    def __init__(self,
                 classifier: Classifier,
                 params: dict) -> None:
        super(ClassificationExperiment, self).__init__()

        self.classifier = classifier
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
        return self.classifier(input, **kwargs)

    def sample_images(self, plot: dict, val_indices: Tensor):
        revert = self.training
        if revert:
            self.eval()

        test_input = []
        predictions = []
        targets = []
        for class_indices in val_indices:
            batch = [self.sample_dataloader.dataset[int(i)]
                     for i in class_indices]
            class_input = []
            for x, y in batch:
                x = x.unsqueeze(0)
                class_input.append(x)
                x = self.classifier(x.to(self.curr_device)).cpu()
                predictions.append(x)
                targets.append(y.unsqueeze(0))
            class_input = torch.cat(class_input, dim=0)
            test_input.append(class_input.unsqueeze(0))

        test_input = torch.cat(test_input, dim=0)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)

        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        fn(test_input=test_input,
           targets=targets,
           predictions=predictions,
           classes=plot['classes'],
           out_path=out_path,
           **plot['params'])

        gc.collect()
        if revert:
            self.train()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.forward(real_img).cpu()
        train_loss = self.classifier.loss_function(y, labels.cpu(),
                                                   **self.params.get('loss_params', {}))
        del real_img
        torch.cuda.empty_cache()
        self.logger.experiment.log({'train/' + key: val.item()
                                    for key, val in train_loss.items()})
        for plot, val_indices in zip(self.plots, self.val_indices):
            if self.global_step % plot['sample_every_n_steps'] == 0:
                self.sample_images(plot, val_indices)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.forward(real_img).cpu()
        val_loss = self.classifier.loss_function(y, labels.cpu(),
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
        optims = [optim.Adam(self.classifier.parameters(),
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

        
        # Persist separate validation indices for each plot
        val_indices = []
        for plot in self.plots:
            classes = plot['classes']
            examples_per_class = plot['examples_per_class']
            indices = []
            for obj in classes:
                class_indices = []
                for _ in range(examples_per_class):
                    idx = get_random_example_with_label(dataset,
                                                        torch.Tensor(obj['labels']),
                                                        all_=obj['all'],
                                                        exclude=class_indices)
                    class_indices.append(idx)
                indices.append(class_indices)
            val_indices.append(indices)
        self.val_indices = val_indices

        return self.sample_dataloader
