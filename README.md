# BatchDetect

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8a3865af38e440e1aa5eaf421392fac3)](https://www.codacy.com/gh/marrlab/BatchDetect/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=marrlab/BatchDetect&amp;utm_campaign=Badge_Grade)

This repository contain the code for detecting batch effects in different
datasets.

For different examples, you can follow our examples in the [docs](docs) folder.

## How to install the package

For installing the package, you can simply clone the repository and run the following command:

```bash
pip -q install <PATH TO THE FOLDER>
```

## How to use the package

You need to have two pandas dataframes. One should include the metadata such as
annotations, donors, hospitals, etc. The other one should include the features of
the dataset. You can initalize the module simply by running:

```python
from batchdetect.batchdetect import BatchDetect
bd = BatchDetect(metadata, features)
```

After that you can run different methods as you would like. These methods include:

-   `bd.low_dim_visualization("pca")`
-   `bd.low_dim_visualization("tsne")`
-   `bd.low_dim_visualization("umap")`
-   `bd.classification_test()`

For more examples or more information about the possible options,
please refere to our [docs](docs) folder.

## Automatic feature extraction

The package also considers that there is no available feature set. We also provided
an automatic feature extraction based on first and second Order image features.
The only requirement is to provide a `metadata` dataframe with the column `file`.

You can create a feature dataframe by simply using the following:

```python
from batchdetect.image import automatic_feature_extraction

df_features = automatic_feature_extraction(metadata)
```

The rest would be similar to the previous part.

## How to cite this work

coming soon

## Metrics to use:

* Silhoutte Score (UMAP) (Rushin)
* Mean local diversity (Sophia)
* Shannon’s equitability (Sophia)
* Jensen-Shannon distance (Sophia)
* KL-Divergence (Manuel)
* other distribution metrics (Ali)
* Look at scIB (Ali)
* FID score (Manuel)
* Moran's I (Rushin)
…..

Add datasets before Wednesday (21st)

LUNG: 3 Cohorts: TCGA, CPTAC, UCL (cis).
