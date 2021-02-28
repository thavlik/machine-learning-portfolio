# Forrest Gump fMRI
These experiments utilize the Forrest Gump fMRI dataset ([paper](https://www.nature.com/articles/sdata20143), [data](https://openneuro.org/datasets/ds000113/versions/1.3.0)). 20 participants were stimulated by the audio feature film *Forrest Gump* and brain activity was recorded with [BOLD imaging](https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging) on a 7-tesla MRI, yielding a 3D image (160x160x36) every two seconds for the duration of the film. The dataset includes start/end times and class labels for each of the film's 198 scenes. Researchers attempt to correlate features of the fMRI scans with each scene's class labels.

## Results
TODO

### Discussion
TODO

## Materials & Methods
### Experiment Design
Any `N` contiguous frames from a scene are assigned the indoor (0) or outdoor (1) label. The label is predicted by feeding the fMRI frames through a 4D convolutional network. Training/validation split is by scene, with 70% of scenes randomly selected for training and the remaining 30% used for validation.

While deep learning and fMRI are a powerful combination, the high dimensionality of fMRI complicates the practice. fMRI input tensors are 4D, and [pytorch](https://pytorch.org/) only implements up to [3D convolutions](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html). TODO: explain how this problem was overcome

### Experiment Files
| File                                                                       | Frame Dimensions       | Temporal Resolution | Notes
| -------------------------------------------------------------------------- | ---------------------- | ------------------- | -----
| [classification/basic.yaml](classification/basic.yaml)                     | 160x160x36             | 16 frames @ 0.5 Hz  | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)     | 160x160x36             | 16 frames @ 0.5 Hz  | 16 frames @ 0.5 Hz  | Hyperparameter search for `basic.yaml`
| [classification/strided.yaml](classification/strided.yaml)                 | 160x160x36             | 16 frames @ 0.25 Hz | Every other input frame is skipped
| [classification/strided_hparams.yaml](classification/strided_hparams.yaml) | 160x160x36             | 16 frames @ 0.25 Hz | Hyperparameter search for `strided.yam

