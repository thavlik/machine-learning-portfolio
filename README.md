# thavlik ML portfolio
This is a repository of my personal machine learning projects.

## Experiments
Many experiments with unrelated datasets share almost all of their code. This repository was designed to showcase how well-designed ML boilerplate facilitates its own reuse.
- [rsna-intracranial](experiments/rsna-intracranial/)
- [tends-fmri](experiments/trends-fmri/)
- [doom](experiments/doom/)

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
- Dataset compiler
- ~~Doom gameplay video links~~
- ~~Implement entrypoints~~
- ~~Implement datasets~~
- ~~Resnet boilerplate~~

## Contributing
Please open an issue or pull request if you would like to contribute.

# License
Everything is released under MIT / Apache 2.0 dual license, which is extremely permissive. Open an issue if somehow neither is sufficient.
