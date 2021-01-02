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
from typing import Callable, Optional, List
from plot import get_plot_fn
from models import Localizer
from merge_strategy import deep_merge
from typing import List
from plot import get_labels
from linear_warmup import LinearWarmup
import boto3
from visdom import Visdom


class BaseExperiment(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 enable_tune: bool = False):
        super().__init__()

        self.save_hyperparameters(config)

        params = config['exp_params']
        self.params = params
        self.curr_device = None
        self.enable_tune = enable_tune

        if 'plot' in self.params:
            plots = self.params['plot']
            if type(plots) is not list:
                plots = [plots]
            self.plots = plots
        else:
            self.plots = []

    def visdom(self):
        if 'visdom' in self.params:
            params = self.params['visdom']
            username = os.environ.get('VISDOM_USERNAME', None)
            password = os.environ.get('VISDOM_PASSWORD', None)
            return Visdom(server=params['host'],
                          port=params['port'],
                          env=params['env'],
                          username=username,
                          password=password)
        else:
            return None

    @abstractmethod
    def sample_images(self, plot: dict, batch: Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_val_batches(self, dataset: Dataset) -> list:
        raise NotImplementedError

    def save_weights(self):
        params = self.params['save_weights']
        if 'local' in params:
            checkpoint_dir = os.path.join(self.logger.save_dir,
                                          self.logger.name,
                                          f"version_{self.logger.version}",
                                          "checkpoints")
            path = os.path.join(checkpoint_dir, f'step{self.global_step}.pt')
            torch.save(self.state_dict(), path)
            if params.get('delete_old', True):
                try:
                    old_checkpoint = f"step{self.global_step - params['every_n_steps']}.pt"
                    os.remove(os.path.join(checkpoint_dir, old_checkpoint))
                except:
                    pass
        if 's3' in params:
            buf = io.BytesIO()
            torch.save(self.state_dict(), buf)
            buf.seek(0)
            s3_params = params['s3']
            prefix = s3_params.get('prefix', 'logs/')
            key = prefix + \
                f"{self.logger.name}/version_{self.logger.version}/checkpoints/step{self.global_step}.pt"
            bucket = s3_params['bucket']
            s3 = boto3.client('s3',
                              endpoint_url=s3_params['endpoint'])
            s3.put_object(Body=buf,
                          Bucket=bucket,
                          Key=key)
            if params.get('delete_old', True):
                old_key = prefix + \
                    f"{self.logger.name}/version_{self.logger.version}/checkpoints/step{self.global_step - params['every_n_steps']}.pt"
                try:
                    s3.delete_object(Bucket=bucket, Key=old_key)
                except:
                    pass

    def log_train_step(self, train_loss: dict):
        if self.enable_tune:
            from ray import tune
            tune.report(**{key: val.item()
                           for key, val in train_loss.items()})
        self.logger.experiment.log({'train/' + key: val.item()
                                    for key, val in train_loss.items()})
        revert = self.training
        if revert:
            self.eval()
        if self.global_step > 0 and 'save_weights' in self.params:
            if self.global_step % self.params['save_weights']['every_n_steps'] == 0:
                self.save_weights()
        for plot, val_batch in zip(self.plots, self.val_batches):
            if self.global_step % plot['sample_every_n_steps'] == 0:
                self.sample_images(plot, val_batch)
                gc.collect()
        if revert:
            self.train()

    def validation_epoch_end(self, outputs: list):
        avg = {}
        for output in outputs:
            for k, v in output.items():
                items = avg.get(k, [])
                items.append(v)
                avg[k] = items
        for metric, values in avg.items():
            self.log('val/' + metric, torch.Tensor(values).mean())

    def configure_schedulers(self, optims: List[Optimizer]) -> list:
        scheds = []
        if 'warmup_steps' in self.params:
            scheds.append(LinearWarmup(optims[0],
                                       lr=self.params['optimizer']['lr'],
                                       num_steps=self.params['warmup_steps']))
        return scheds

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
        ds_params = deep_merge(
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
        self.val_batches = self.get_val_batches(dataset)
        return self.sample_dataloader
