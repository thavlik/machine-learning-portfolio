# Grasp-and-Lift EEG Detection
These experiments utilize the dataset from [Luciw et al 2014](https://www.nature.com/articles/sdata201447). The subjects wear a 32-channel, 500 Hz EEG cap and perform a task involving grasping and lifting an object ([video 1](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM69_ESM.avi), [video 2](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM70_ESM.avi)). The model is trained to classify the last observed sample according to its associated part of the motor task. That is, the model is fed EEG data "as it is made", and can therefore be used in real time. Each of the twelve participants performed eight trials.

![](images/data_example.png)

## Results
The classification task was solved across a split of subjects (1-8 for training, 10-12 for validation) with clear evidence of generalization. This performance was achieved by training a 4.4 million parameter ResNet-style model for about three hours:

![](images/balanced_val_accuracy_small.png)

For reference, here is the training accuracy:

![](images/balanced_train_accuracy_small.png)

## Discussion
### Balanced Labels
I was only able to achieve generalization across subjects after balancing each minibatch's distribution of labels. With unbalanced labels, the network would converge to ~99% accuracy very quickly, making it necessary to compute a *relative accuracy* metric that normalizes performance above baseline accuracy. This presented issues with precision, as the optimizer would be forced to work on numbers within a very small range - differences of <2% accuracy in this case. There were issues with optimization more broadly, as batches often lacked labeled examples entirely. ResNet's extensive use of batch normalization is reported to perform poorly with unbalanced labels, so the balancing appears critical to this experiment's success.

## Materials & Methods
There are three ways to approach this supervised experiment: by splitting train/test examples randomly, by splitting the subjects into train/test groups, and by splitting the trials into train/test groups. EEG is notorious for its sensitivity to electrode placement, making generalization across subjects (and even trials) nontrivial. Each method was investigated.

### Random Split
Randomly assigning examples from the dataset to be withheld for validation is the simplest and least sophisticated way to create a training split. It has the distinct advantage of ensuring the validation examples are substantially similar (but not identical) to training examples. This reliably induces overfitting, which can be leveraged in the early stages of an experiment to ensure a working implementation. The following training dynamics serve as evidence that various aspects of the experiment function as expected:

![](images/training_acc.jpg)

### Subject/Series Split
The holy grail of this EEG classification task is the ability to generalize across novel subjects. Though this was one of the first medical AI projects I started working on, such results proved elusive for nearly a year. Most of time, training accuracy would improve as shown above, but validation accuracy would decrease over time:

![](images/validation_acc.jpg)

Determining whether this was due to the data - i.e. training/validation data being too different - or my own code wasn't straightforward. The solution of balancing labels occured to me when I inquired about the practice in my professional work.

### Future Direction
GraspAndLiftEEGDetection has grown to be a sort of stock dataset for me. Because my familiarity with it is so high, I find it ideal for multi-dataset experiments. It's a practical guarantee that a model from this experiment will appear in a future experiment.

Right now my goal is to contribute this dataset to [Lightning Bolts](https://github.com/PyTorchLightning/lightning-bolts/pull/742). While the authors [greenlit its inclusion in late 2020](https://github.com/PyTorchLightning/lightning-bolts/pull/446), I opted to hold off until I could confirm cross-subject generalization was feasible.

### Experiment Files
| File                                                                         | Input Size (CxW) | Sample Rate | Notes
| ---------------------------------------------------------------------------- | ---------------- | ----------- | -----
| [classification/basic.yaml](classification/basic.yaml)                       | 32x2048          | 500 Hz      | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)       | 32x2048          | 500 Hz      | Hyperparameter search for `basic.yaml`
| [classification/halfres.yaml](classification/halfres.yaml)                   | 32x1024          | 250 Hz      | Half-resolution input
| [classification/halfres_hparams.yaml](classification/basic_hparams.yaml)     | 32x1024          | 250 Hz      | Hyperparameter search for `halfres.yaml`
| [classification/balanced.yaml](classification/balanced.yaml)                 | 32x2048          | 500 Hz      | Minibatches have equal label distribution
| [classification/balanced_hparams.yaml](classification/balanced_hparams.yaml) | 32x2048          | 500 Hz      | Hyperparameter search for `balanced.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ----- 
| [src/classification.py](/src/classification.py)                          | Base classification experiment
| [src/dataset/grasp_and_lift_eeg.py](/src/dataset/grasp_and_lift_eeg.py)  | Grasp-and-Lift EEG dataset
| [src/models/resnet_classifier1d.py](/src/models/resnet_classifier1d.py)  | 1D ResNet classifier model

