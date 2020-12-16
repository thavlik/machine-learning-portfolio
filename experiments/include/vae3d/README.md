# 3D Variational Autoencoders
Adding another dimension to the classic experiment, 3D VAE is capable of handling voxel and video data. 

## Flavors
Several derivatives VAE experiments are available in the separate yaml files.
- **mse.yaml** is the original experiment with Mean Square Error and [KLD](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) loss
- **adversarial.yaml** leverages adversarial regularization inspired by [Goodfellow 2014](https://arxiv.org/abs/1406.2661)

## Compatible Datasets
This experiment can be applied to the following datasets:
- video data
- any kind of sMRI

## Results
(TODO: insert picture of validation results)