import math
import torch
from torch import optim, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import pytorch_lightning as pl
from models.base import BaseVAE
from utils import data_loader
from dataset import DoomDataset


def get_dataset(dataset_params):
    if dataset_params['loader'] == 'doom':
        return DoomDataset(**dataset_params['options'])
    else:
        raise ValueError(f"unknown loader {dataset_params['loader']}")


class BasicExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(BasicExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        kld_weight = self.params['kld_weight'] * self.params['batch_size']/self.num_train_imgs
        train_loss = self.model.loss_function(*results,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx,
                                              kld_weight=kld_weight)

        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        kld_weight = self.params['kld_weight'] * self.params['batch_size']/self.num_train_imgs
        val_loss = self.model.loss_function(*results,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx,
                                            kld_weight=kld_weight)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.sample_dataloader))
        test_input = test_input[:8]
        test_input = test_input.to(self.curr_device)
        recons = self.model.generate(test_input, labels=[])
        out_path = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/recons_{self.logger.name}_{self.current_epoch}.png"
        orig = test_input.data.cpu()
        recons = recons.data.cpu()
        fig = self.plot(orig, recons)
        del test_input, recons

    def plot(self, orig, recon):
        raise NotImplementedError

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['lr'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['lr_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['lr_2'])
                optims.append(optimizer2)
        except:
            pass
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        dataset = get_dataset(self.params['dataset'])
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        dataset = get_dataset(self.params['dataset'])
        self.sample_dataloader = DataLoader(dataset,
                                            batch_size=self.params['batch_size'],
                                            shuffle=True,
                                            drop_last=True)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader
