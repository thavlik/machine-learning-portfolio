# thavlik ML portfolio
This is a repository of my personal machine learning projects.

## Experiments
This repository was designed to showcase well-designed project structure. Configurations are defined in yaml files, which can be composed via the `base` directive to conveniently form derivative experiments with minimal boilerplate. An experiment can be run by passing the path to the input yaml as the `--config` flag to `src/main.py`:

`python3 src/main.py --config experiments/rsna-intracranial/vae_mse.yaml`

**Note: script assumes the current working directory is the root of this repository**. By convention, all file and directory paths in yaml files are given relative to the repo root.

### Unsupervised
These experiments showcase unsupervised modeling tasks on a variety of both medical and non-medical datasets. Variational Autoencoders (VAE) embed higher dimensional data into a compact latent space by modeling the principle components as a multivariate gaussian, a la [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114). Unsupervised modeling tasks are distinguished by their use of plentiful, unlabeled data. Unsupervised pretraining confers an exponential boost in data efficiency on relevant supervised tasks [2][3], rendering these experiments highly relevant to [few-/one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning).
- [1D Variational Autoencoder](experiments/abstract/vae1d/README.md), used for time series
- [2D Variational Autoencoder](experiments/abstract/vae2d/README.md), used for 2D imagery
- [3D Variational Autoencoder](experiments/abstract/vae3d/README.md), used for video and 3D imagery
- [4D Variational Autoencoder](experiments/abstract/vae4d/README.md), used for fMRI

### Supervised
These experiments make use of ground truth provided with the data. Ground truth for medical data typically constitutes that of an attending physician.
- [RSNA Intracranial Hemorrhage Prediction](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)
- [CQ500](http://headctstudy.qure.ai/dataset)
- [TReNDS fMRI](https://www.kaggle.com/c/trends-assessment-prediction/data)
- [DeepLesion](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images)
- [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)
- [torchvision datasets](https://pytorch.org/docs/stable/torchvision/datasets.html), for non-medical reference
  - CelebA (human faces)
  - MNIST (handwritten digits)
  - etc...

## Relevant Literature
Many of the ideas implemented in this repository were first detailed in the following papers:

1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
2. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
3. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
4. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
5. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
6. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
7. [Towards Photographic Image Manipulation with Balanced Growing of Generative Autoencoders](https://arxiv.org/abs/1904.06145)

## TODO
- Progressive growing decoder
- Implement beta loss term

## Contributing
Please open an issue or pull request if you would like to contribute.

# License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
