from torch import nn
from models import Classifier
from torch.nn import functional as F
from vae import VAEExperiment


class VAEClassifierExperiment(VAEExperiment):
    def __init__(self,
                 classifier: Classifier,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.classifier = classifier
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # First move to device so it doesn't happen twice
        real_img, labels = batch
        real_img = real_img.to(self.curr_device)
        batch = (real_img, labels)
        loss, results = super().training_step_raw(batch, batch_idx, optimizer_idx)
        z = results[4] # Check BaseVAE.loss_function()
        prediction = self.classifier(z)
        classifier_loss = self.classifier.loss_function(prediction, labels)
        loss['loss'] += classifier_loss 
        loss['Classifier_Loss'] = classifier_loss 
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        real_img = real_img.to(self.curr_device)
        batch = (real_img, labels)
        loss, results = super().validation_step_raw(batch, batch_idx, optimizer_idx)
        z = results[4] # Check BaseVAE.loss_function()
        prediction = self.classifier(z)
        classifier_loss = self.classifier.loss_function(prediction, labels)
        loss['loss'] += classifier_loss 
        loss['Classifier_Loss'] = classifier_loss 
        return loss
