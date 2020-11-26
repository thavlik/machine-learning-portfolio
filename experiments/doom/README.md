Variational autoencoders (VAE) and other experiments trained on DOOM 1/2 gameplay videos

## Motivation
Latent representations and unsupervised pretraining boost data efficiency on more challenging supervised [1] and reinforcement learning tasks [2]. The goal of this project is to provide both the Doom and machine learning communities with:
- High quality datasets comprised of Doom gameplay
- Various ready-to-run experiments
- Suitable boilerplate for derivative projects

## The Data
Gameplay videos are sourced from YouTube with permission. Special thanks to the following creators for their contributions to the community and this dataset - these individuals are the lifeblood of the Doom community:
- [Timothy Brown](https://www.youtube.com/user/mArt1And00m3r11339)
- [decino](https://www.youtube.com/c/decino)
- [Zero Master](https://www.youtube.com/channel/UCiVZWY9LmrJFOg3hWGjyBbw)

The dataset is compiled by taking the original videos (~167 Gb worth) and re-encoding them into fixed-duration, 320x240 @ 10fps videos. This allows data loaders to retrieve random frames at training speeds (128-1024 frames/sec) whereas the raw 1080p video can only be sampled at ~3 frames/sec (i7 7800X @ 3.5 GHz).

TODO: The compiler is currently a work in progress. When the dataset is compiled, this page will link to it.

