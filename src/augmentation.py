from torch import optim
from models import create_model
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
        self.model = create_model(**config['model_params'],
                                  input_shape=input_shape)

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
