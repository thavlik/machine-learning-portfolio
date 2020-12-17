# Grasp-and-Lift EEG Detection
These experiments utilize the dataset from [Luciw et al 2014](https://www.nature.com/articles/sdata201447). The subjects wear a 32-channel EEG and perform a task involving grasping and lifting an object ([video 1](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM69_ESM.avi), [video 2](https://grasplifteeg.nyc3.digitaloceanspaces.com/41597_2014_BFsdata201447_MOESM70_ESM.avi)). The model is trained to classify the last observed sample according to its associated part of the motor task. That is, the model is fed EEG data "as it is made", and can therefore be used in real time.

(TODO: insert picture of example data)

## Results
(TODO: insert picture of validation results)

### Relative Accuracy
Because >97% of all the dataset's samples are not associated with any class label, the model's overall accuracy does not intuitively reflect how well it performs above baseline performance. The model quickly learns the optimal strategy of outputting mostly zeros - achieving accuracy in the high 90%'s - then slowly learns to selectively predict class labels based on features of the input data. Normalizing accuracy above the baseline to [0, 1] in a quantity termed *relative accuracy* elegantly represents model performance after it learns the approximate frequencies of the labels.

**TODO**: show training dynamics

## TODO
- Finish training model
- Provide per-class breakdown for each event
- Show off training dynamics