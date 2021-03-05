# Forrest Gump fMRI (work in progress)
These experiments utilize the Forrest Gump fMRI dataset ([paper](https://www.nature.com/articles/sdata20143), [data](https://openneuro.org/datasets/ds000113/versions/1.3.0)). 20 participants were stimulated by the audio feature film *Forrest Gump* and brain activity was recorded with [BOLD imaging](https://en.wikipedia.org/wiki/Blood-oxygen-level-dependent_imaging) on a 7-tesla MRI, yielding a 3D image (36x160x160) every two seconds for the duration of the film. The dataset includes start/end times and class labels for each of the film's 198 scenes. Researchers attempt to correlate features of the fMRI scans with each scene's class labels.

## Results
TODO

## Materials & Methods
### Experiment Design
Each scene of the film is given DAY/NIGHT and INTERIOR/EXTERIOR labels. Individual BOLD frames (two seconds in duration) are assigned *soft labels* by taking a weighted average of the scenes' labels. 

BOLD has a "built-in" acquisition delay due to the biological processes underlying changes in the brain's blood flow. The apparent lag between the stimulus (audiofilm) and response (fMRI) is on the order of seconds ([Liao et al 2005](https://www.math.mcgill.ca/keith/delay/delay.pdf)). Model performance - the correlation between stimulus and apparent BOLD activity - can be improved by applying a constant delay between stimulus and response. An ideal offset duration can be determined with a hyperparameter search. TODO: hyperparameter search

Input BOLD frames are fed to 3D convolutional layers with residual connections. A linear output layer then predicts the frame's soft labels. [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) is used to calculate multiclass loss.

The dataset authors provided two alignment derivatives: linear and nonlinear. 

### Experiment Files
| File                                                                              | Frame Dimensions | Temporal Resolution | Notes
| --------------------------------------------------------------------------------- | ---------------- | ------------------- | -----
| [classification/linear.yaml](classification/linear.yaml)                          | 48x132x175       | 1 frame @ 0.5 Hz    | Linear alignment
| [classification/linear_hparams.yaml](classification/linear_hparams.yaml)          | 48x132x175       | 1 frame @ 0.5 Hz    | Hyperparameter search for `linear.yaml`
| [classification/nonlinear.yaml](classification/nonlinear.yaml)                    | 48x132x175       | 1 frame @ 0.5 Hz    | Nonlinear alignment
| [classification/nonlinear_hparams.yaml](classification/nonlinear_hparams.yaml)    | 48x132x175       | 1 frame @ 0.5 Hz    | Hyperparameter search for `nonlinear.yaml`
| [classification/unaligned.yaml](classification/unaligned.yaml)                    | 36x160x160       | 1 frame @ 0.5 Hz    | No alignment (raw), base experiment
| [classification/unaligned_hparams.yaml](classification/unaligned_hparams.yaml)    | 36x160x160       | 1 frame @ 0.5 Hz    | Hyperparameter search for `unaligned.yaml`

### Source Files
| File                                                                          | Notes
| ----------------------------------------------------------------------------- | ----- 
| [src/classification.py](/src/classification.py)                               | Base classification experiment
| [src/dataset/forrestgump.py](/src/dataset/forrestgump.py)                     | Dataset class for converted data (fast)
| [src/dataset/forrestgumpraw.py](/src/dataset/forrestgumpraw.py)               | Dataset class for raw BIDS data (very slow)
| [src/dataset/forrestgump_converter.py](/src/dataset/forrestgump_converter.py) | Data converting utilities
| [src/models/resnet_classifier3d.py](/src/models/resnet_classifier3d.py)       | 3D ResNet classifier model

## Future Direction
While deep learning and fMRI are a powerful combination, the high dimensionality of fMRI complicates the practice. fMRI input tensors are 4D, and [pytorch](https://pytorch.org/) only implements up to [3D convolutions](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html). This is sufficient when the model considers a single BOLD frame, but problematic when dealing with multiple. It is supposed that multi-frame input with a 4D (likely convolutional) kernel has potential to be outperform single-frame models built upon 3D convolutions, especially on more complex tasks. For example, instead of predicting class labels for a single BOLD frame, the model can predict class labels at a higher resolution: one BOLD frame (acquired in 2.0 seconds) is used to predict a sequence of eight 0.25s-long labels, corresponding to the scene labels.

The authors of the data included several additional experiment results with many of the participants. These experiments present an opportunity to train a model on more than one task - a technique that can improve generalization on a single task. 
