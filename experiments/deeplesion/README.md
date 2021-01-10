# DeepLesion
These experiments utilize the [DeepLesion](https://nihcc.app.box.com/v/DeepLesion) dataset released by the [National Institute of Health](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images) in 2018. The modeling task entails detecting and localizing the bounding boxes of abdominal lesions for individually labeled CT slices.

## Results
After several days of training, there was some evidence of generalization to the validation split (prediction is yellow, ground truth is green):

![](images/initial_localization.png)

The model appears to be making mistakes characteristic of non-experts by inaccurately localizing the lesion to any "lesion-like" blob, such as a cross section of intestine or aorta. Instances where the model fails to localize to anything remotely lesion-like (top left) suggest these examples may be "harder" than those in the training split.

Because these examples were never exposed to the model during training, partial overlaps reflect weak generalization. This is hypothesized to be caused by the model granting excess saliency to the features of nearby tissue, as opposed to focusing more on the actual tumor pixels. This would be caused by discrepancies in surrounding tissue deformation between training and validation splits.

## Materials & Methods
### Initial Attempt
The simplest architecture entails modeling the location of lesions directly:

```python
class MyLocalizationModel(nn.Module):
    ...

    def forward(self, x: Tensor) -> Tensor:
        # The model directly predicts the minX/maxX
        # and minY/maxY of the lesion
        bbox = self.predict(x)        
        return bbox
```

### Half-Resolution Training
Due to perceptual limitations with the 3x3 convolutional kernel, a large number of filters for each layer must be used to extract details from full resolution inputs. Halving the input resolution results in an effective doubling of kernel dimensions with no effect on parameter count. By increasing the model's receptive field, large / low frequency details can be detected with fewer parameters, conferring larger batch sizes and improved training performance. 

### Batch Normalization
A hyperparameter search was carried out to determine the effect of batch normalization, which indicated superior training performance in its absence. This is likely due to small batch sizes, which are necessary even when halving the input resolution.

### Multivariate Guassian
To add sophistication, the next iteration attempts to model the lesion's bounding box as a multivariate gaussian. Concretely, this means that instead of the model directly predicting the class labels, it predicts mean and standard deviation parameters that are then used to sample a normal distribution. This is also known as the *reparametrization trick*, and its use in was heavily inspired by [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114). Unlike with variational autoencoders - which use a log normal distribution - this implementation uses the classic normal distribution:

```python
from torch.distributions import Normal

class MyLocalizationModel(nn.Module):
    ...

    def forward(self, x: Tensor) -> Tensor:
        # The model predicts parameters of the distribution
        mu, std_dev = self.predict(x)

        # Sample from the normal distribution to produce the
        # final bounding box estimate.
        bbox = Normal(mu, std_dev).rsample()
        
        return bbox
```

The goal was to capture information about lesion margins, with overfitting occuring as the standard deviation approaches zero. Sigmoidal activation was used for all output activation layers to normalize predictions to [0, 1]. This permits the introduction of another hyperparameter, `kappa`, that scales the normalized standard deviation to an even smaller, more appropriate range:

```python
class MyLocalizationModel(nn.Module):
    ...

    def forward(self, x: Tensor, kappa: float) -> Tensor:
        mu, std_dev = self.predict(x)

        # Scale the standard deviation from [0, 1] to [0, kappa]
        # where kappa is typically a very small value (e.g. 0.05)
        std_dev *= kappa

        bbox = Normal(mu, std_dev).rsample()
        
        return bbox
```

This effectively limits the influence of the distribution and favors the central value with lower values of `kappa`, allowing this technique to blended with the direct approach.

Unsurprisingly, the multivariate gaussian architecture performs comparably to the direct approach. Visualization of lesion margins has not yet been performed.

### Experiment Files
| File                                                                 | Input Resolution | Notes
| -------------------------------------------------------------------- | ---------------- | ------
| [localization/basic.yaml](localization/basic.yaml)                   | 512x512          | "Vanilla" experiment setup
| [localization/basic_hparams.yaml](localization/basic_hparams.yaml)   | 512x512          | Hyperparameter search for `basic.yaml`
| [localization/halfres.yaml](localization/halfres.yaml)               | 256x256          | Half-resolution input slices
| [localization/halfres_hparams.yaml](localization/basic_hparams.yaml) | 256x256          | Hyperparameter search for `halfres.yaml`

## Future Direction
DeepLesion appears to be a "hard" problem, manifesting as poor convergence and data efficiency. [Browatzki & Wallraven 2019](https://arxiv.org/abs/1911.10448) addresses similar problems with facial landmark prediction by sandwiching freshly initialized, trainable layers with frozen layers pre-trained on an unsupervised task, increasing data efficiency by several orders of magnitude, furthermore leading to convergence - even in cases where before it was not possible. Different flavors of this technique (creative unsupervised pre-training) are critical for nontrivial deep learning problems. Incidentally, their use is widespread (refer to the *Relevant Literature* section of the repository root).

