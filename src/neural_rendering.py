import os
import torch
from models import create_model
from vae import VAEExperiment
from dataset import ReferenceDataset, get_example_shape
import pytorch_lightning as pl

def neural_gbuffer(config: dict, run_args: dict) -> pl.LightningModule:
    pass