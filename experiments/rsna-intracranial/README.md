These experiments utilize the [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) dataset released by the [Radiological Society of North America](https://www.rsna.org/).

## Results
![](images/img.png)

Left indicator bar represents relative per-class accuracy, right indicator bar represents accuracy across all classes. Columns include examples with at least the designated class label, and may have more labels (meaning some examples could have been displayed in multiple columns). This was done because some classes of hemorrhage do not occur isolation within the dataset. Concretely, the "Epidural" column includes any examples positive for epidural hemhorrage, some of which are also positive for intraparenchymal, subarachnoid, etc. The "Control" column features exclusively healthy subjects. 

TODO: show training dynamics

## TODO
- Improve visualization
  - Add class labels to each column
  - Offer explanation of color indicators
  - Improve contrast
- use s3cmd to make rsna-intracranial bucket public
