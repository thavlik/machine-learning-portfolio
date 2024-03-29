import os
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from typing import Iterator

from base_experiment import BaseExperiment
from dataset import get_example_shape
from models import create_model
from plot import get_plot_fn


class LocalizationExperiment(BaseExperiment):

    def __init__(self, config: dict, enable_tune: bool = False, **kwargs):
        super().__init__(config=config, enable_tune=enable_tune, **kwargs)
        exp_params = config['exp_params']
        input_shape = get_example_shape(exp_params['data'])
        localizer = create_model(**config['model_params'],
                                 input_shape=input_shape)
        self.localizer = localizer

    def trainable_parameters(self) -> Iterator[Parameter]:
        return self.localizer.parameters()

    def sample_images(self, plot: dict, batch: list):
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
        out_path = os.path.join(
            self.logger.save_dir, self.logger.name,
            f"version_{self.logger.version}",
            f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        image = fn(test_input=test_input,
                   pred_params=pred_params,
                   target_params=target_params,
                   out_path=out_path,
                   **plot['params'])
        self.logger.experiment.add_image(plot['fn'], image, self.global_step)
        vis = self.visdom()
        if vis is not None:
            vis.image(image, win=plot['fn'])

    def training_step(self, batch, batch_idx):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_params = self.localizer(real_img).cpu()
        train_loss = self.localizer.loss_function(
            pred_params, targ_params.cpu(),
            **self.params.get('loss_params', {}))
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        real_img, targ_labels, targ_params = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        pred_params = self.localizer(real_img).cpu()
        val_loss = self.localizer.loss_function(
            pred_params, targ_params.cpu(),
            **self.params.get('loss_params', {}))
        self.log_val_step(val_loss)
        return val_loss

    def get_val_batches(self, dataset: Dataset) -> list:
        val_batches = []
        for plot in self.plots:
            batch = [
                get_positive_example(dataset)
                for _ in range(plot['batch_size'])
            ]
            for _, label, _ in batch:
                assert torch.is_nonzero(label)
            val_batches.append(batch)
        return val_batches


def get_positive_example(ds):
    try:
        return ds.get_positive_example()
    except:
        return get_positive_example(ds.dataset)
