# EARLY WORK IN PROGRESS

Variational autoencoders (VAE) trained on DOOM 1/2 gameplay videos

## Motivation
Latent representations and unsupervised pretraining boost data efficiency on more challenging supervised [1] and reinforcement learning tasks [2]. The goal of this project is to provide both the Doom and machine learning communities with:
- High quality datasets comprised of Doom gameplay
- Various ready-to-run VAE experiments
- Suitable boilerplate for derivative projects

### Relevant Literature
1. [3FabRec: Fast Few-shot Face alignment by Reconstruction](https://arxiv.org/abs/1911.10448)
2. [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
3. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
4. [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## The Data
Gameplay videos are sourced from Youtube. Special thanks to the following creators for their contributions to the community and this dataset - these individuals are truly the lifeblood of the Doom community:
- [Timothy Brown](https://www.youtube.com/user/mArt1And00m3r11339)
- [decino](https://www.youtube.com/c/decino)
- [Zero Master](https://www.youtube.com/channel/UCiVZWY9LmrJFOg3hWGjyBbw)

This project will seek permission from the video authors before distributing the videos directly, e.g. from an S3 bucket. Currently, [youtube_dl](https://pypi.org/project/youtube_dl/) is used to download the videos to a local cache. Note: code such as this providing access to copyrighted content is [explicitly recognized as fair use by Github](https://github.blog/2020-11-16-standing-up-for-developers-youtube-dl-is-back/). If your content has made its way into the dataset and you would prefer it be omitted, please open an issue.

## Contributing
Please open an issue or pull request if you would like to contribute.

## TODO
- Progressive growing decoder a la [3]
- Implement beta loss term from [4]
- Implement FID(orig, recons) loss
- Dataset compiler
- ~~Doom gameplay video links~~
- ~~Implement entrypoints~~
- ~~Implement datasets~~
- ~~Resnet boilerplate~~

