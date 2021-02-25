# thavlik machine learning portfolio
This is a repository of my personal deep learning projects.

## Supervised Experiments
These experiments make use of ground truth provided with the data. Ground truth for medical data typically constitutes the judgment of one or more attending physicians.

- [RSNA Intracranial Hemorrhage Prediction](experiments/rsna-intracranial/README.md): classification of brain CT scans by hemorrhage type
- [DeepLesion](experiments/deeplesion/README.md): localization of tumors on abdominal CT scans
- [Grasp-and-Lift EEG Detection](experiments/grasp_and_lift_eeg/README.md): fine-grain detection of motor activity using EEG

Works in progress:
- [Neural Rendering](experiments/graphics/README.md) (WIP): single-shot, differentiable 3D rendering

<!---
### Unsupervised Modeling
These experiments showcase unsupervised modeling tasks on a variety of both medical and non-medical datasets. Variational Autoencoders (VAE) embed higher dimensional data into a compact latent space by modeling the principle components as a multivariate gaussian, a la [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114). Unsupervised modeling tasks are distinguished by their use of plentiful, unlabeled data. Pretraining a network on an unsupervised task confers an exponential boost in data efficiency on relevant supervised tasks [[2](https://arxiv.org/abs/1911.10448)] [[3](https://arxiv.org/abs/1707.08475)], rendering these experiments highly relevant to [few-/one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning).

- [1D Variational Autoencoder](experiments/include/vae1d/README.md), used for EEG and other time series
- [2D Variational Autoencoder](experiments/include/vae2d/README.md), used for 2D images
- [3D Variational Autoencoder](experiments/include/vae3d/README.md), used for video and structural MRI
- [4D Variational Autoencoder](experiments/include/vae4d/README.md), used for fMRI
-->

## Datasets
These are datasets that I have authored/compiled personally.

- [Doom Gameplay Dataset](https://github.com/thavlik/doom-gameplay-dataset)
- [Quake Gameplay Dataset](https://github.com/thavlik/quake-gameplay-dataset)

## Running Code
Configurations are defined in yaml files, which can be composed via the `include` directive to conveniently form derivative experiments with minimal boilerplate. An experiment can be run by passing the path to the input yaml as the `--config` flag to `src/main.py`:

`python3 src/main.py --config experiments/mnist/vae/fid.yaml`

**Note: the script assumes the current working directory is the root of this repository**. By convention, all file and directory paths in yaml files are given relative to the repository root.

If an experiment hangs during the initial validation pass, it is likely because [nonechucks](https://github.com/msamogh/nonechucks) is suppressing exceptions thrown by the dataset. This behavior improves fault tolerance, but can complicate debugging.

## Relevant Literature
Many of the ideas implemented in this repository were first detailed in the following papers:

1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
2. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
3. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
4. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
5. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
6. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
7. [Towards Photographic Image Manipulation with Balanced Growing of Generative Autoencoders](https://arxiv.org/abs/1904.06145)
8. [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)
9. [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
10. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
 
## Contributing
This repository was intended to be repurposed. As part of the open source community, I do not share the perception that minor contributions from others dilutes the claim that this repository is fully representative of my research capabilities. On the contrary, publishing software good enough for others to justify spending their time and effort improving is a nontrivial achievement.

Please open an issue or pull request if you would like to contribute.

## License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
