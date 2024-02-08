import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset

import os
from typing import Iterator, List

from base_experiment import BaseExperiment
from dataset import get_example_shape
from models import create_model
from plot import get_plot_fn, get_random_example_with_label


class ClassificationExperiment(BaseExperiment):

    def __init__(self, config: dict, enable_tune: bool = False, **kwargs):
        super().__init__(config=config, enable_tune=enable_tune, **kwargs)
        self.classifier = create_model(**config['model_params'],
                                       input_shape=get_example_shape(
                                           config['exp_params']['data']))

    def sample_images(self, plot: dict, batches: List[Tensor]):
        test_input = []
        predictions = []
        targets = []
        for class_batch in batches:
            class_input = []
            class_predictions = []
            class_targets = []
            for x, y in class_batch:
                x = x.unsqueeze(0)
                class_input.append(x)
                x = self.classifier(x.to(self.curr_device)).detach().cpu()
                class_predictions.append(x)
                class_targets.append(y.unsqueeze(0))
            class_input = torch.cat(class_input, dim=0)
            test_input.append(class_input.unsqueeze(0))
            predictions.append(
                torch.cat(class_predictions, dim=0).unsqueeze(0))
            targets.append(torch.cat(class_targets, dim=0).unsqueeze(0))

        test_input = torch.cat(test_input, dim=0)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)

        # Extensionless output path (let plotting function choose extension)
        out_path = os.path.join(
            self.logger.save_dir, self.logger.name,
            f"version_{self.logger.version}",
            f"{self.logger.name}_{plot['fn']}_{self.global_step}")
        fn = get_plot_fn(plot['fn'])
        fn(test_input=test_input,
           targets=targets,
           predictions=predictions,
           classes=plot['classes'],
           out_path=out_path,
           vis=self.visdom(),
           **plot['params'])

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.classifier(real_img)
        train_loss = self.classifier.loss_function(
            y.cpu(), labels.cpu(), **self.params.get('loss_params', {}))
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.classifier(real_img)
        val_loss = self.classifier.loss_function(
            y.cpu(), labels.cpu(), **self.params.get('loss_params', {}))
        self.log_val_step(val_loss)
        return val_loss

    def trainable_parameters(self) -> Iterator[Parameter]:
        return self.classifier.parameters()

    def get_val_batches(self, dataset: Dataset) -> list:
        val_batches = []
        for plot in self.plots:
            classes = plot['classes']
            examples_per_class = plot['examples_per_class']
            class_batches = []
            for obj in classes:
                batch = []
                class_indices = []
                for _ in range(examples_per_class):
                    idx = get_random_example_with_label(dataset,
                                                        torch.Tensor(
                                                            obj['labels']),
                                                        all_=obj['all'],
                                                        exclude=class_indices)
                    batch.append(dataset[idx])
                    class_indices.append(idx)
                class_batches.append(batch)
            val_batches.append(class_batches)
        return val_batches
