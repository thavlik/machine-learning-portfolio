# UCLA Consortium for Neuropsychiatric Phenomics LA5c Study
TODO: pick phenotypes we want to model with T1w

### Experiment Files
| File                                                                     | Input Resolution      | Notes
| ------------------------------------------------------------------------ | --------------------- | ------
| [classification/basic.yaml](classification/basic.yaml)                   | TODO                  | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)   | TODO                  | Hyperparameter search for `basic.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ----- 
| [src/classification.py](/src/classification.py)                          | Base classification experiment
| [src/dataset/la5c.py](/src/dataset/la5c.py)                              | LA5c dataset
| [src/models/resnet_classifier3d.py](/src/models/resnet_classifier3d.py)  | 3D ResNet classifier model
