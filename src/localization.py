import gc
import os
import math
import torch
import io
import numpy as np
from torch import optim, Tensor
from torchvision import transforms
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
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
from models import Classifier, Localizer
from merge_strategy import strategy
from typing import List
from plot import get_labels
from linear_warmup import LinearWarmup
import boto3
from visdom import Visdom
from base_experiment import BaseExperiment
from models import create_model
from dataset import get_example_shape


class LocalizationExperiment(BaseExperiment):
    def __init__(self,
                 config: dict,
                 enable_tune: bool = False):
        super().__init__(config=config,
                         enable_tune=enable_tune)
        exp_params = config['exp_params']
        input_shape = get_example_shape(exp_params['data'])
        localizer = create_model(**config['model_params'],
                                 input_shape=input_shape)
        self.localizer = localizer

    def sample_images(self, plot: dict, batch: Tensor):
        test_input = []
        pred_params = []
        target_params = []
        for item in batch:
            x, target_label, target_param = item
            x = x.unsqueeze(0)
            test_input.append(x)
            pred_param = self.localizer(x.to(self.curr_device))
            pred_params.append(pred_param.detach().cpu())
            target_params.append(target_param.unsqueeze(0))
        test_input = torch.cat(test_input, dim=0).cpu()
        pred_params = torch.cat(pred_params, dim=0)
        target_params = torch.cat(target_params, dim=0)

        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        fn(test_input=test_input,
           pred_params=pred_params,
           target_params=target_params,
           out_path=out_path,
           vis=self.visdom(),
           **plot['params'])

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_params = self.localizer(real_img).cpu()
        train_loss = self.localizer.loss_function(pred_params,
                                                  targ_params,
                                                  **self.params.get('loss_params', {}))
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_params = self.localizer(real_img).cpu()
        val_loss = self.localizer.loss_function(pred_params,
                                                targ_params,
                                                **self.params.get('loss_params', {}))
        return val_loss

    def configure_optimizers(self):
        optims = [optim.Adam(self.localizer.parameters(),
                             **self.params['optimizer'])]
        scheds = self.configure_schedulers(optims)
        return optims, scheds

    def get_val_batches(self, dataset: Dataset) -> list:
        val_batches = []
        for plot in self.plots:
            batch = [get_positive_example(dataset)
                     for _ in range(plot['batch_size'])]
            for _, label, _ in batch:
                assert torch.is_nonzero(label)
            val_batches.append(batch)
        return val_batches


def get_positive_example(ds):
    try:
        return ds.get_positive_example()
    except:
        return get_positive_example(ds.dataset)
