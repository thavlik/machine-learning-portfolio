These experiments utilize the [DeepLesion](https://nihcc.app.box.com/v/DeepLesion) dataset released by the [National Institute of Health](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images) in 2018. The modeling task entails detecting and localizing visible lesions.

## Initial Attempt
My first effort entailed modeling the probability of a lesion being present and the lesion's bounding box as a multivariate  gaussian. Concretely, this means that instead of the model directly predicting the class labels, it predicts mean and variance parameters that are then used to sample a normal distribution. This is also known as the *reparametrization trick*, and its use in was heavily inspired by [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114).

```python
def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

class MyLocalizationModel(nn.Module):
    ...

    def forward(self, input: Tensor) -> Tensor:
        mu, log_var = self.predict(input)
        pred = reparameterize(mu, log_var)
        return pred
```

## Results
(TODO: insert picture of validation results)

(TODO: insert picture of training dynamics)

## TODO
- Visualize results