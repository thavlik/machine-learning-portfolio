# 1D Variational Autoencoders
Time series can be transformed with convolutional networks just like image data. 

## Flavors
Several derivatives VAE experiments are available in the separate yaml files.
- **mse.yaml** is the original experiment with Mean Square Error and [KLD](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) loss
- **fid.yaml** is identical to mse.yaml, but [FID(original, reconstruction)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) is included in loss
- **adversarial.yaml** leverages adversarial regularization inspired by [Goodfellow 2014](https://arxiv.org/abs/1406.2661)

## Compatible Datasets
This experiment can be applied to the following datasets:
- [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)
