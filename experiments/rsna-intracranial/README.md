# RSNA Intracranial Hemorrhage Detection

These experiments utilize the [RSNA Intracranial Hemorrhage Detection](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019) dataset released by the Radiological Society of North America in 2019. It proposes a relatively simple task where one or more classes of ICH are assigned to individual CT slices. 

## Results
~95% validation accuracy was achieved with <12 hours of training on a single 1080 Ti. I believe near-100% accuracy could be achieved with additional work.

> ![](images/RSNA_HalfRes_classifier2d_20000.jpg)  
***Figure 1: Visualization of data and model performance. Columns include examples with at least the designated class label, and may have more labels (meaning some examples could have been displayed in multiple columns). This was done because some classes of hemorrhage do not occur in isolation within the dataset. Concretely, the "Epidural" column includes any examples positive for epidural hemhorrage, some of which are also positive for intraparenchymal, subarachnoid, etc. The "Control" column features exclusively healthy subjects. The left indicator bar represents per-class accuracy and the right represents accuracy across all classes. Visualizing these metrics separately helps gauge performance on a per-class basis. Low (baseline) performance is mapped to red, and 100% accuracy is bright green.***

Because model accuracy is high, all indicators bars in ***Figure 1*** are green. However, the `Control` column's per-class indicators are all a perceivably lighter shade of green than those of other columns, suggesting the model is biased in favor of [type 1 error](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors). This is considered desirable for such medical technology because *type 2 errors*—false negative—are far more likely to be fatal; human attention should *always* be directed in edge cases.

## Materials & Methods
### Training Dynamics
Stable training dynamics were exhibited with the [mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss function. Label balancing was not necessary, likely because of the dataset's relatively even distribution of class labels.

> ![training tensorboard](images/training-dynamics.jpg)  
***Training accuracy and loss, respectively. Aggressive smoothing is applied.***

Training beyond one epoch did not increase validation accuracy. Optimal performance was achieved around step 19,000—prior to the end of the first epoch—suggesting further training at this learning rate results in overfitting.

> ![overfitting tensorboard](images/overfitting.jpg)  
***Validation accuracy does not improve with more training when the model is excessively biased towards the training data. This is a sign of overfitting.***

Because such high validation accuracy was observed so early into training, it is possible that the validation set is composed of categorically easier examples. It is difficult to make this qualitative judgment without the ability to interpret CTs.

### Half-Resolution Training
The weights from any fully convolutional network are able to be applied to images of arbitrary dimensions—even dimensions differing from those used for training. While larger convolutional kernel sizes are able to more easily capture large-scale details, they come with a significant increase in compute overhead. For simplicity, the 3x3 and 1x1 kernels are used exclusively, and other aspects of the experiment (model architecture, input resolution, etc.) are optimized against this small kernel size.

Training at half the input resolution (256x256) can be seen as comparable to doubling the dimensions of the kernel (up to 6x6) and enabling the model to more easily capture detail at larger scales. Reduced memory overhead and increased batch sizes can be appreciated with this single change, leading to considerable performance gains. The resultant weights can then be used to improve training at higher resolutions.

### Experiment Files
| File                                                                     | Input Size (CxHxW) | Notes
| ------------------------------------------------------------------------ | ------------------ | ------
| [classification/basic.yaml](classification/basic.yaml)                   | 1x512x512          | "Vanilla" experiment setup
| [classification/basic_hparams.yaml](classification/basic_hparams.yaml)   | 1x512x512          | Hyperparameter search for `basic.yaml`
| [classification/halfres.yaml](classification/halfres.yaml)               | 1x256x256          | Half-resolution input slices
| [classification/halfres_hparams.yaml](classification/basic_hparams.yaml) | 1x256x256          | Hyperparameter search for `halfres.yaml`

### Source Files
| File                                                                     | Notes
| ------------------------------------------------------------------------ | ----- 
| [src/classification.py](/src/classification.py)                          | Base classification experiment
| [src/dataset/rsna_intracranial.py](/src/dataset/rsna_intracranial.py)    | RSNA Intracranial Hemorrhage dataset
| [src/models/resnet_classifier2d.py](/src/models/resnet_classifier2d.py)  | 2D ResNet classifier model

## Download Results
The weights, configs, and logs for two runs are available for download: [Link](https://nyc3.digitaloceanspaces.com/rsna-ich/results/RSNA_HalfRes.zip)

## License
### Data
RSNA Intracranial Hemorrhage Detection Challenge is property of the Radiological Society of North America and is publicly available at no cost.  

### Code
[Apache 2.0](../../LICENSE-Apache) / [MIT](../../LICENSE-MIT) dual-license. Please contact me if this is somehow not permissive enough and we'll add whatever free license is necessary for your project.
