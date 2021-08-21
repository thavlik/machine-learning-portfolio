import os
import gc
import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import get_dataset
from plot import get_plot_fn
from merge_strategy import deep_merge
from models import create_model, BaseRenderer
from linear_warmup import LinearWarmup

class NeuralGBufferExperiment(pl.LightningModule):
    def __init__(self,
                 model: BaseRenderer,
                 params: dict) -> None:
        super().__init__()
        self.model = model
        self.params = params
        self.curr_device = None
        plots = self.params['plot']
        if type(plots) is not list:
            plots = [plots]
        self.plots = plots

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.curr_device = self.device
        orig, labels = batch
        recons = self.model(*labels)
        train_loss = self.model.loss_function(recons,
                                              orig,
                                              **self.params.get('loss_params', {}))
        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})
        if self.global_step > 0:
            for plot, val_indices in zip(self.plots, self.val_indices):
                if self.global_step % plot['sample_every_n_steps'] == 0:
                    self.sample_images(plot, val_indices)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        orig, labels = batch
        recons = self.model(*labels)
        val_loss = self.model.loss_function(recons,
                                            orig,
                                            **self.params.get('loss_params', {}))
        return val_loss

    def configure_optimizers(self):
        optims = [optim.Adam(self.model.parameters(),
                             **self.params['optimizer'])]
        scheds = []
        if 'warmup_steps' in self.params:
            scheds.append(LinearWarmup(optims[0],
                                       lr=self.params['optimizer']['lr'],
                                       num_steps=self.params['warmup_steps']))
        return optims, scheds

    def train_dataloader(self):
        dataset = get_dataset(self.params['data']['name'],
                              self.params['data'].get('training', {}))
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          **self.params['data'].get('loader', {}))

    def val_dataloader(self):
        ds_params = deep_merge(
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

    def sample_images(self, plot: dict, val_indices: Tensor):
        revert = self.training
        if revert:
            self.eval()
        test_input = []
        recons = []

        batch = [self.sample_dataloader.dataset[int(i)]
                 for i in val_indices]
        for x, transform in batch:
            test_input.append(x.unsqueeze(0))
            out = self.model(*[a.unsqueeze(0) for a in transform])
            recons.append(out)
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


def neural_gbuffer(config: dict, run_args: dict) -> pl.LightningModule:
    exp_params = config['exp_params']
    image_size = exp_params['data']['training']['rasterization_settings']['image_size']
    model = create_model(**config['model_params'],
                         width=image_size,
                         height=image_size,
                         channels=3,
                         enable_fid=True)
    return NeuralGBufferExperiment(model, exp_params)
