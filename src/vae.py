import os
import gc
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from typing import Tuple, Iterator
from plot import get_plot_fn
from base_experiment import BaseExperiment
from models import create_model
from dataset import get_example_shape


class VAEExperiment(BaseExperiment):

    def __init__(self,
                 config: dict,
                 enable_tune: bool = False,
                 **kwargs) -> None:
        super(VAEExperiment, self).__init__(config=config,
                                            enable_tune=enable_tune,
                                            **kwargs)
        params = config['exp_params']
        c, h, w = get_example_shape(params['data'])
        self.model = create_model(**config['model_params'],
                                  width=w,
                                  height=h,
                                  channels=c,
                                  enable_fid='fid_weight' in params,
                                  progressive_growing=len(
                                      params['progressive_growing'])
                                  if 'progressive_growing' in params else 0)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def get_lod(self) -> Tuple[int, float]:
        schedule = self.params['progressive_growing']
        for i, step in enumerate(schedule):
            if self.trainer.global_step >= step:
                lod = len(schedule) - i - 1
                next_step = schedule[i + 1]
                alpha = (self.trainer.global_step - step) / (next_step - step)
                return lod, alpha
        return (0, 0.0)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        results = self.forward(real_img, labels=labels)
        kwargs = dict(optimizer_idx=optimizer_idx,
                      batch_idx=batch_idx,
                      kld_weight=self.params.get('kld_weight', 0.0) *
                      self.params['batch_size'] / self.num_train_imgs)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        train_loss = self.model.loss_function(*results, **kwargs)
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
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
                      self.params['batch_size'] / self.num_val_imgs,
                      **kwargs)
        if 'fid_weight' in self.params:
            kwargs['fid_weight'] = self.params['fid_weight']
        val_loss = self.model.loss_function(*results, **kwargs)
        self.log_val_step(val_loss)
        return val_loss

    def sample_images(self, plot: dict, batch: Tensor):
        revert = self.training
        if revert:
            self.eval()
        test_input = []
        recons = []
        for x in batch:
            x = x.unsqueeze(0)
            test_input.append(x)
            x = self.model.generate(x.to(self.curr_device),
                                    labels=[]).detach().cpu()
            recons.append(x)
        test_input = torch.cat(test_input, dim=0)
        recons = torch.cat(recons, dim=0)
        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(
            self.logger.save_dir, self.logger.name,
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

    def trainable_parameters(self) -> Iterator[Parameter]:
        return self.model.parameters()

    def get_val_batches(self, dataset: Dataset) -> list:
        val_batches = []
        n = len(dataset)
        for plot in self.plots:
            indices = torch.randint(low=0,
                                    high=n,
                                    size=(plot['batch_size'], 1)).squeeze()
            batch = [dataset[i][0] for i in indices]
            val_batches.append(batch)
        return val_batches

    """
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
        ds_params = deep_merge(
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
    """
