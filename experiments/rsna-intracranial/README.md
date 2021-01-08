These experiments utilize the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) dataset released by the [Radiological Society of North America](https://www.rsna.org/).

## Results
![](images/img.png)

The left indicator bar represents per-class accuracy and the right represents accuracy across all classes. Visualizing these metrics separately helps gauge performance on a per-class basis. Low (baseline) performance is mapped to red, and 100% accuracy is signified with bright green.

Columns include examples with at least the designated class label, and may have more labels (meaning some examples could have been displayed in multiple columns). This was done because some classes of hemorrhage do not occur isolation within the dataset. Concretely, the "Epidural" column includes any examples positive for epidural hemhorrage, some of which are also positive for intraparenchymal, subarachnoid, etc. The "Control" column features exclusively healthy subjects. 

## Half-Resolution Training
The weights from any fully convolutional network are able to be applied to images of arbitrary dimensions - even dimensions that differ from those with which it was trained. While larger convolutional kernel sizes are able to more easily capture large-scale details, they come with a significant increase in compute overhead. For simplicity, my portfolio uses the 3x3 and 1x1 kernels exclusively, and other aspects of the experiment (model architecture, input resolution, etc.) are optimized against this small kernel size.

Training at half the input resolution (256x256) can be seen as analagous to doubling the dimensions of the kernel (up to 6x6) and enabling the model to more easily capture detail at those larger scales. Reduced memory overhead and larger batch sizes can be appreciated with this single change, leading to dramatically higher training performance and ~95% validation accuracy within <12 hours of training.

## Training Dynamics
Stable training dynamics were exhibited with the [mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss function.

![training tensorboard](images/training-dynamics.jpg)

Training beyond one epoch did not increase validation accuracy. Optimal performance was achieved around step 19,000, suggesting further training at this learning rate results in overfitting:

![overfitting tensorboard](images/overfitting.jpg)

