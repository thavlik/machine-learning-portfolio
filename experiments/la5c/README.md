# LA5c Study (work in progress)
These experiments utilize the [LA5c Study](https://openneuro.org/datasets/ds000030/versions/1.0.0) from the [Preprocessed Consortium for Neuropsychiatric Phenomics dataset](https://f1000research.com/articles/6-1262/v2). 265 participants completed extensive psychometric and neuroimaging examinations. A relatively simple modeling task was chosen: predict participants' questionnaire answers given their T1w MRI.

## Results
TODO: pick interesting questions we want to model with T1w

TODO: show a case where overtly visible differences in brain structure were exploited to infer correct answers 

## Materials & Methods
### Choice of Questions
The dataset comes with hundreds of self-report answers for an amalgam of questionnaires. Participants' answers are modeled by the structural T1-weighted MRI of each participant. TODO: explain process behind selecting questions

TODO: show which questions were selected

### Experiment Files
| File                                                                             | Input Size (CxDxHxW)  | Notes
| -------------------------------------------------------------------------------- | --------------------- | ------
| [classification/bilingual.yaml](classification/bilingual.yaml)                   | 1x176x256x256         | Model bilingual Y/N in terms of T1w
| [classification/bilingual_hparams.yaml](classification/bilingual_hparams.yaml)   | 1x176x256x256         | Hyperparameter search for `bilingual.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ------------------------------
| [src/classification.py](/src/classification.py)                          | Base classification experiment
| [src/dataset/la5c.py](/src/dataset/la5c.py)                              | LA5c dataset class
| [src/models/resnet_classifier3d.py](/src/models/resnet_classifier3d.py)  | 3D ResNet classifier model

## License
### Data
LA5c is released under the [Creative Commons Zero (CC0)](https://creativecommons.org/choose/zero/) license.

### Code
[Apache 2.0](../../LICENSE-Apache) / [MIT](../../LICENSE-MIT) dual-license. Please contact me if this is somehow not permissive enough and we'll add whatever free license is necessary for your project.
