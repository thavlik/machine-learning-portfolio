# thavlik ML portfolio
This is a repository of my personal machine learning projects.

## Experiments
This repository was designed to showcase well-designed project structure. Configurations are defined in yaml files, which can be composed via the `base` directive to conveniently form derivative experiments with minimal boilerplate.

### Unsupervised
These experiments showcase unsupervised modeling tasks on a variety of both medical and non-medical datasets.
- [1D Variational Autoencoder](experiments/abstract/vae1d/)
- [2D Variational Autoencoder](experiments/abstract/vae2d/)
- [3D Variational Autoencoder](experiments/abstract/vae3d/)
- [4D Variational Autoencoder](experiments/abstract/vae4d/)

### Supervised
These experiments make use of ground truth provided with the data. Ground truth for medical imagery typically constitutes that of an attending physician.
- [RSNA Intracranial Hemorrhage Prediction](experiments/rsna-intracranial/)
- [CQ500](experiments/cq500/)
- [TReNDS fMRI](experiments/trends-fmri/)
- [DeepLesion](experiments/deeplesion/)
- [Grasp-and-Lift EEG Detectiton](experiments/eeg/)
- [torchvision datasets](https://pytorch.org/docs/stable/torchvision/datasets.html), for non-medical reference

## Relevant Literature
Many of the ideas implemented in this repository were first detailed in the following papers:

1. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
2. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
3. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
4. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
5. [Towards Photographic Image Manipulation with Balanced Growing of Generative Autoencoders](https://arxiv.org/abs/1904.06145)

## TODO
- Progressive growing decoder a la [3]
- Implement beta loss term from [4]

## Contributing
Please open an issue or pull request if you would like to contribute.

# License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
