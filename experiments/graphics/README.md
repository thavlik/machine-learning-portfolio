# Neural Rendering (work in progress)
Graphics rendering pipelines are becoming exponentially more complicated. Generative adversarial networks (GANs) are able to produce realistic imagery ([Goodfellow 2014](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf), [Karras et al 2019](https://arxiv.org/abs/1912.04958)), providing an alternative means of achieving computer graphics indistinguishable from reality.

My interest in AI graphics is motivated by the goal of seeing the technology put to use in surgical training tools. While the applications are innumerable, surgery simulators with differentiable patient models - allowing the educator to gradually increase the difficulty of a case - are particularly interesting to me.

## The Basics
The most basic neural rendering experiment attempts to reproduce the pixels drawn by a standard rasterization- based renderer according to a bounded transform.

In this first experiment, the model is given a 4x4 transformation matrix as input, and is tasked with rendering the mesh with no further variation. For simplicity's sake, [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) was used to calculate reconstruction loss.

![](images/NeuralGBuffer_plot2d_55000.png)

Target image is on the left, model output is on the right.

Unsurprisingly, use of MSE without a progressive growing strategy yields only blurry messes. Further work is required to simplify image synthesis early in training.

## TODO
- implement progressive growing
- implement FID loss

## License
### Data
Cow model is part of [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and is [licensed under BSD](https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/LICENSE).

### Code
[Apache 2.0](../../LICENSE-Apache) / [MIT](../../LICENSE-MIT) dual-license. Please contact me if this is somehow not permissive enough and we'll add whatever free license is necessary for your project.
