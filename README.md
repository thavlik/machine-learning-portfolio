# thavlik ML portfolio
This is a repository of my personal machine learning projects.

## Experiments
This repository was designed to showcase how well-designed ML boilerplate encourages its reuse. The same unsupervised modeling experiment can run on any of the datasets. Some datasets may not have any supervised tasks implemented, even if ground truth is available.

### Medical Imaging
- [RSNA Intracranial Hemorrhage](experiments/rsna-intracranial/)
- [CQ500](experiments/cq500/)
- [TReNDS fMRI](experiments/trends-fmri/)
- [DeepLesion](experiments/deeplesion/)

### Video Games
- [doom](experiments/doom/)

### Reference
Additional reference datasets are pulled from [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html).
TODO: organize experiments around reference datasets

## Relevant Literature
Many of the ideas implemented in this repository were first detailed in the following papers:

1. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
2. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
3. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
4. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## TODO
- Progressive growing decoder a la [3]
- Implement beta loss term from [4]
- Implement FID(orig, recons) loss
- ~~Implement entrypoints~~
- ~~Implement datasets~~
- ~~Resnet boilerplate~~

## Contributing
Please open an issue or pull request if you would like to contribute.

# License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
