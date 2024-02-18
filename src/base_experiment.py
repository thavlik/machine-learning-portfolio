import gc
import io
import os
import torch
from abc import abstractmethod
from torch import Tensor, optim
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ChainedScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Iterator, List

import boto3
import numpy as np
import pytorch_lightning as pl
from visdom import Visdom

from dataset import balanced_sampler, get_dataset
from linear_warmup import LinearWarmup
from merge_strategy import deep_merge


class BaseExperiment(pl.LightningModule):

    def __init__(self, config: dict, enable_tune: bool = False, **kwargs):
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

    @abstractmethod
    def trainable_parameters(self) -> Iterator[Parameter]:
        raise NotImplementedError

    def save_weights(self):
        params = self.params['save_weights']
        if 'local' in params:
            checkpoint_dir = os.path.join(self.logger.save_dir,
                                          self.logger.name,
                                          f"version_{self.logger.version}",
                                          "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
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
            s3 = boto3.client('s3', endpoint_url=s3_params['endpoint'])
            s3.put_object(Body=buf, Bucket=bucket, Key=key)
            if params.get('delete_old', True):
                old_key = prefix + \
                    f"{self.logger.name}/version_{self.logger.version}/checkpoints/step{self.global_step - params['every_n_steps']}.pt"
                try:
                    s3.delete_object(Bucket=bucket, Key=old_key)
                except:
                    pass

    def log_train_step(self, train_loss: dict):
        for key, val in train_loss.items():
            self.log('train/' + key, val)
        revert = self.training
        if revert:
            self.eval()
        if self.global_step > 0 and 'save_weights' in self.params:
            save_interval = self.params['save_weights']['every_n_steps']
            if save_interval != 0 and self.global_step % save_interval == 0:
                self.save_weights()
        for plot, val_batch in zip(self.plots, self.val_batches):
            if self.global_step % plot['sample_every_n_steps'] == 0:
                self.sample_images(plot, val_batch)
                gc.collect()
        if revert:
            self.train()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.on_validation_epoch_end(*args, **kwargs)

    def log_val_step(self, val_loss: dict):
        self.val_outputs += [val_loss]

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = []

    def on_validation_epoch_end(self):
        avg = {}
        outputs = self.val_outputs
        for output in outputs:
            for k, v in output.items():
                items = avg.get(k, [])
                items.append(v.numpy())
                avg[k] = items
        for metric, values in avg.items():
            key = 'val/' + metric
            mean = np.array(values).mean()
            self.log(key, mean)
            if self.enable_tune:
                from ray import train
                train.report({key: mean})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainable_parameters(),
                               **self.params['optimizer'])
        scheds = self.configure_schedulers(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheds,
            'monitor': 'train/loss',
        }

    def configure_schedulers(self, optimizer: Optimizer) -> list:
        scheds = []
        if 'warmup_steps' in self.params:
            scheds.append(
                LinearWarmup(optimizer,
                             lr=self.params['optimizer']['lr'],
                             num_steps=self.params['warmup_steps']))
        if 'reduce_lr_on_plateau' in self.params:
            scheds.append(
                ReduceLROnPlateau(optimizer,
                                  **self.params['reduce_lr_on_plateau']))
        return ChainedScheduler(scheds)

    def train_dataloader(self):
        ds_params = self.params['data'].get('training', {})
        dataset = get_dataset(self.params['data']['name'],
                              ds_params,
                              split=self.params['data'].get('split', None),
                              safe=self.params['data'].get('safe', True),
                              train=True)
        params = self.params['data'].get('loader', {}).copy()
        if self.params['data'].get('balanced', False) == True:
            params['sampler'] = balanced_sampler(dataset)
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          **params)

    def val_dataloader(self):
        ds_params = deep_merge(self.params['data'].get('training', {}).copy(),
                               self.params['data'].get('validation', {}))
        dataset = get_dataset(self.params['data']['name'],
                              ds_params,
                              split=self.params['data'].get('split', None),
                              safe=self.params['data'].get('safe', True),
                              train=False)
        params = self.params['data'].get('loader', {}).copy()
        if 'balanced' in self.params['data']:
            params['sampler'] = balanced_sampler(
                dataset, **self.params['data']['balanced'])
        self.sample_dataloader = DataLoader(
            dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            **params)
        self.num_val_imgs = len(self.sample_dataloader)
        self.val_batches = self.get_val_batches(dataset)
        return self.sample_dataloader
