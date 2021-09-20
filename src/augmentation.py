import os
import torch
from torch import optim, Tensor
from torch.utils.data import Dataset
from models import create_model
from plot import get_plot_fn
from base_experiment import BaseExperiment
from models import create_model
from dataset import get_example_shape


class AugmentationExperiment(BaseExperiment):
    def __init__(self,
                 config: dict,
                 enable_tune: bool = False,
                 **kwargs):
        super().__init__(config=config,
                         enable_tune=enable_tune,
                         **kwargs)
        input_shape = get_example_shape(config['exp_params']['data'])
        self.constraint = create_model(**config['constraint_params'],
                                       input_shape=input_shape)
        self.constraint.requires_grad = False
        self.constraint.eval()
        self.model = create_model(**config['model_params'],
                                  input_shape=input_shape)

    def sample_images(self, plot: dict, batch: Tensor):
        transformed = self.model(batch)
        out_path = os.path.join(self.logger.save_dir,
                                self.logger.name,
                                f"version_{self.logger.version}",
                                f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        fn(x=batch,
           y=transformed,
           out_path=out_path,
           vis=self.visdom(),
           **plot['params'])

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

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, _ = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        train_loss = self.model.loss_function(real_img,
                                              constraint=self.constraint,
                                              **self.params.get('loss_params', {}))
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, _ = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        val_loss = self.model.loss_function(real_img,
                                            constraint=self.constraint,
                                            **self.params.get('loss_params', {}))
        self.log_val_step(val_loss)
        return val_loss

    def configure_optimizers(self):
        optims = [optim.Adam(self.model.parameters(),
                             **self.params['optimizer'])]
        scheds = self.configure_schedulers(optims)
        return optims, scheds
