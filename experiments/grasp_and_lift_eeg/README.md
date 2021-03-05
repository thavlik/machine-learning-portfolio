# Grasp-and-Lift EEG Detection
These experiments utilize the dataset from [Luciw et al 2014](https://www.nature.com/articles/sdata201447). The subjects wear a 32-channel, 500 Hz EEG cap and perform a task involving grasping and lifting an object ([video 1](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM69_ESM.avi), [video 2](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM70_ESM.avi)). The model is trained to classify the last observed sample according to its associated part of the motor task. That is, the model is fed EEG data "as it is made", and can therefore be used in real time. Each of the twelve participants performed eight trials.

![](images/data_example.png)

There are three ways to approach this supervised experiment: by splitting train/test examples randomly, by splitting the subjects into train/test groups, and by splitting the trials into train/test groups. EEG is notorious for its sensitivity to electrode placement, making generalization across subjects (and even trials) nontrivial. 

## Results
The classification task was easily solved when training on a random split, but failed to show any degree generalization when trained on a split of subjects or trials - as indicated by strictly decreasing validation accuracy. This outcome reflects disparate levels of difficulty for each aforementioned method.

Regardless of the split method, training accuracy increases logarithmically. Near-100% training accuracy is achievable after a week of training on a single 1080 Ti. Random split achieves high validation accuracy. Subject and trial splits exhibit unfavorable training dynamics.

The following depicts >80% relative accuracy* achieved within two days of training - a trend observed with all splitting methods:

![](images/training_acc.jpg)

However, when splitting the data by subject or trial, the validation accuracy (measured here at around 6 and 12 hours of training) is strictly decreasing:

![](images/validation_acc.jpg)

This trend reflects immediate overfitting on the training data. Further work is justified to determine if the trial split can be solved with alternative methods.

### *Relative Accuracy
Because >97% of all the dataset's samples are not associated with any class label, the model's overall accuracy does not intuitively reflect how well it performs above baseline performance. The model quickly learns the optimal strategy of outputting mostly zeros - achieving accuracy over 97% - then slowly learns to selectively predict class labels based on features of the input data. Normalizing accuracy above the baseline to [0, 1] in a quantity termed *relative accuracy* elegantly represents model performance after it learns the approximate frequencies of the labels.

### Experiment Files
| File                                                                     | Input Resolution      | Notes
| ------------------------------------------------------------------------ | --------------------- | ------
| [classification/basic.yaml](classification/basic.yaml)                   | 2048 samples @ 500 Hz | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)   | 2048 samples @ 500 Hz | Hyperparameter search for `basic.yaml`
| [classification/halfres.yaml](classification/halfres.yaml)               | 1024 samples @ 250 Hz | Half-resolution input
| [classification/halfres_hparams.yaml](classification/basic_hparams.yaml) | 1024 samples @ 250 Hz | Hyperparameter search for `halfres.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ----- 
| [src/dataset/grasp_and_lift_eeg.py](/src/dataset/grasp_and_lift_eeg.py)  | Grasp-and-Lift EEG dataset
| [src/models/resnet_classifier1d.py](/src/models/resnet_classifier1d.py)  | 1D ResNet classifier model
| [src/classification.py](/src/classification.py)                          | Classification experiment

## Future Direction
The random splitting method almost trains with the same data as it uses for validation, so it is unsurprising that it yields high validation accuracy. Because there does not appear to be enough data to generalize across subject or trial splits, further efforts could examine the effect of the random split's proportion. It is estimated that relatively few training examples (<50% of the dataset) would be necessary to maintain high validation accuracy.

