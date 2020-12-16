# thavlik ML portfolio
This is a repository of my personal deep learning projects.

## Experiments
This repository was designed to showcase well-designed project structure. Configurations are defined in yaml files, which can be composed via the `base` directive to conveniently form derivative experiments with minimal boilerplate. An experiment can be run by passing the path to the input yaml as the `--config` flag to `src/main.py`:

`python3 src/main.py --config experiments/mnist/vae/fid.yaml`

**Note: the script assumes the current working directory is the root of this repository**. By convention, all file and directory paths in yaml files are given relative to the repo root.

### Unsupervised
These experiments showcase unsupervised modeling tasks on a variety of both medical and non-medical datasets. Variational Autoencoders (VAE) embed higher dimensional data into a compact latent space by modeling the principle components as a multivariate gaussian, a la [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114). Unsupervised modeling tasks are distinguished by their use of plentiful, unlabeled data. Pretraining a network on an unsupervised task confers an exponential boost in data efficiency on relevant supervised tasks [2][3], rendering these experiments highly relevant to [few-/one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning).
- [1D Variational Autoencoder](experiments/include/vae1d/README.md), used for EEG and other time series
- [2D Variational Autoencoder](experiments/include/vae2d/README.md), used for 2D images
- [3D Variational Autoencoder](experiments/include/vae3d/README.md), used for video and structural MRI
- [4D Variational Autoencoder](experiments/include/vae4d/README.md), used for fMRI

### Supervised
These experiments make use of ground truth provided with the data. Ground truth for medical data typically constitutes the judgment of one or more attending physicians.
- [Neural Rendering](experiments/graphics/README.md)
- [RSNA Intracranial Hemorrhage Prediction](experiments/rsna-intracranial/README.md)
- [CQ500](http://headctstudy.qure.ai/dataset)
- [TReNDS fMRI](https://www.kaggle.com/c/trends-assessment-prediction/data)
- [DeepLesion](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images)
- [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)
- [torchvision datasets](https://pytorch.org/docs/stable/torchvision/datasets.html), for non-medical reference
  - CelebA (human faces)
  - MNIST (handwritten digits)
  - etc...

## Datasets
These are datasets that I have authored/compiled personally.

- [Doom Gameplay Dataset](https://github.com/thavlik/doom-gameplay-dataset)
- [Quake Gameplay Dataset](https://github.com/thavlik/quake-gameplay-dataset)

## Relevant Literature
Many of the ideas implemented in this repository were first detailed in the following papers:

1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
2. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
3. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
4. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
5. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
6. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
7. [Towards Photographic Image Manipulation with Balanced Growing of Generative Autoencoders](https://arxiv.org/abs/1904.06145)
8. [Understanding disentangling in β-VAE](https://arxiv.org/pdf/1804.03599.pdf)

## TODO
- Second optimizer for adversarial training

## Contributing
Please open an issue or pull request if you would like to contribute.

# License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
