import gc
import io
import os
import torch
from abc import abstractmethod
from torch import Tensor, optim
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ChainedScheduler, LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import Iterator, Optional

import boto3
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from ray import train
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
        self.reduce_lr_on_plateau = None

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

    def _save_weights_local(self):
        checkpoint_dir = os.path.join(self.logger.save_dir, self.logger.name,
                                      f"version_{self.logger.version}",
                                      "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'step_{self.global_step}.pt')
        torch.save(self.state_dict(), path)

    def _save_weights_s3(self,
                         bucket: str,
                         endpoint: str,
                         prefix: str = 'logs/'):
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        buf.seek(0)
        key = prefix + \
            f"{self.logger.name}/version_{self.logger.version}/checkpoints/step_{self.global_step}.pt"
        s3 = boto3.client('s3', endpoint_url=endpoint)
        s3.put_object(Body=buf, Bucket=bucket, Key=key)

    def save_weights(self):
        params = self.params['save_weights']
        if 'local' in params:
            self._save_weights_local(**params['local'])
        if 's3' in params:
            self._save_weights_s3(**params['s3'])

    def log_train_step(self, train_loss: dict):
        for key, val in train_loss.items():
            self.log(f'train/{key}', val)
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
                gc.collect()  # Free up memory
        if revert:
            self.train()

    def on_validation_epoch_start(self) -> None:
        # Reset the validation losses.
        self.val_losses = []

    def log_val_step(self, val_loss: dict):
        # Store the validation losses for the epoch.
        self.val_losses += [val_loss]

    def on_validation_epoch_end(self):
        # Calculate the average validation losses for the epoch.
        avgs = {
            k: np.array([v[k].numpy() for v in self.val_losses]).mean()
            for k in self.val_losses[0].keys()
        }
        # Report the averages to the logger(s).
        for metric, mean in avgs.items():
            self.log(f'val/{metric}', mean, on_epoch=True, on_step=False)
        # Report the averages to ray/tune for hparam optimization.
        if self.enable_tune:
            train.report({
                f'val/{metric}': mean
                for metric, mean in avgs.items()
            })

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.trainable_parameters(),
                               **self.params['optimizer'])
        sched = self.configure_scheduler(optimizer)
        result = {'optimizer': optimizer, 'monitor': 'train/loss'}
        if sched is not None:
            result['lr_scheduler'] = {
                'scheduler': sched,
                # Provide sensible defaults for the scheduler params.
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/loss',
                'strict': True,
                # Override the scheduler params with the specified config.
                **self.params.get('scheduler', {}).get('params', {})
            }
        return result

    def configure_scheduler(self,
                            optimizer: Optimizer) -> Optional[LRScheduler]:
        sched_params = self.params['scheduler']
        if sched_params is None:
            return None
        scheds = []
        if 'warmup_steps' in sched_params:
            scheds.append(
                LinearWarmup(optimizer,
                             lr=self.params['optimizer']['lr'],
                             num_steps=sched_params['warmup_steps']))
        if 'reduce_lr_on_plateau' in sched_params:
            # Because ReduceLROnPlateau requires a `metric` argument
            # to its `step` method, we must step it manually within
            # the `on_train_epoch_end` hook. Schedulers managed by
            # pytorch lightning cannot require any arguments to their
            # `step` method.
            self.reduce_lr_on_plateau = ReduceLROnPlateau(
                optimizer, **sched_params['reduce_lr_on_plateau'])
        if len(scheds) == 0:
            return None
        if len(scheds) == 1:
            return scheds[0]
        return ChainedScheduler(scheds)

    def on_train_epoch_end(self, *args, **kwargs):
        if self.reduce_lr_on_plateau is not None:
            self.reduce_lr_on_plateau.step(
                self.trainer.callback_metrics['train/loss'])

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
