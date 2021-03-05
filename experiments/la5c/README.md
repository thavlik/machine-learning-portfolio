# LA5c Study (work in progress)
These experiments utilize the [LA5c Study](https://openneuro.org/datasets/ds000030/versions/1.0.0) from the [Preprocessed Consortium for Neuropsychiatric Phenomics dataset](https://f1000research.com/articles/6-1262/v2). 265 participants completed extensive psychometric and neuroimaging examinations. A relatively simple modeling task was chosen: predict participants' questionnaire answers given their T1w MRI.

## Results
TODO: pick interesting questions we want to model with T1w

TODO: show a case where overtly visible differences in brain structure were exploited to infer correct answers 

## Materials & Methods
### Experiment Files
| File                                                                     | Input Size (CxDxHxW)  | Notes
| ------------------------------------------------------------------------ | --------------------- | ------
| [classification/basic.yaml](classification/basic.yaml)                   | 1x176x256x256         | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)   | 1x176x256x256         | Hyperparameter search for `basic.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ----- 
| [src/classification.py](/src/classification.py)                          | Base classification experiment
| [src/dataset/la5c.py](/src/dataset/la5c.py)                              | LA5c dataset
| [src/models/resnet_classifier3d.py](/src/models/resnet_classifier3d.py)  | 3D ResNet classifier model
