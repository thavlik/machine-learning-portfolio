import gc
import os
import math
import torch
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
from models import Classifier
from merge_strategy import strategy
from typing import List
from plot import get_random_example_with_label
from linear_warmup import LinearWarmup
from base_experiment import BaseExperiment

class ClassificationExperiment(BaseExperiment):

    def __init__(self,
                 classifier: Classifier,
                 params: dict) -> None:
        super().__init__(params)
        self.classifier = classifier

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
            predictions.append(torch.cat(class_predictions, dim=0).unsqueeze(0))
            targets.append(torch.cat(class_targets, dim=0).unsqueeze(0))

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
           vis=self.vis,
           **plot['params'])


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.classifier(real_img).cpu()
        train_loss = self.classifier.loss_function(y, labels,
                                                   **self.params.get('loss_params', {}))
        self.log_train_step(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = self.device
        real_img = real_img.to(self.curr_device)
        y = self.classifier(real_img).cpu()
        val_loss = self.classifier.loss_function(y, labels.cpu(),
                                                 **self.params.get('loss_params', {}))
        return val_loss

    def configure_optimizers(self):
        optims = [optim.Adam(self.classifier.parameters(),
                             **self.params['optimizer'])]
        scheds = self.configure_schedulers(optims)
        return optims, scheds
    
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
                                                        torch.Tensor(obj['labels']),
                                                        all_=obj['all'],
                                                        exclude=class_indices)
                    batch.append(dataset[idx])
                    class_indices.append(idx)
                class_batches.append(batch)
            val_batches.append(class_batches)
        return val_batches

