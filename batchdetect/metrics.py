import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.datasets import make_multilabel_classification
import pandas as pd


# template for a new metric:
# --------------------------
# def metric_function(metadata, batch_list, df_features):
# """
# inputs:
#     metadata:   pandas dataframe with all the possible covariates
#     batch_list: a list including the covariates to be used for the
#                 batch effect correction. Thse covariates should be
#                 a column name in the metadata dataframe
#     df_features: pandas dataframe with the numerical features

# outputs:
#     metrics (dict): dictionary with the calculated metrics
# """
# ...
# return metrics


def mean_local_diversity(metadata, batch_list, df_features, k=5):
    """
    computes the Shannon diversity for each point in the dataset over its k nearest neighbors
    
    inputs:
        metadata:   pandas dataframe with all the possible covariates
        batch_list: a list including the covariates to be used for the
                    batch effect correction. Thse covariates should be
                    a column name in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        metrics (dict): dictionary with the calculated metrics
    """
    num_points = len(df_features)
    mean_local_diversity = {bl: [] for bl in batch_list}

    for i in range(num_points):
        # Compute distances between the point and all other points
        distances = cdist([df_features.iloc[i]], df_features)[0]

        # Sort distances and labels in ascending order
        sorted_indices = np.argsort(distances)
        sorted_metadata = metadata.reindex(sorted_indices)

        # Exclude the point itself from the neighborhood
        local_metadata = sorted_metadata[1:k+1]

        # Compute Shannon diversity for the local neighborhood
        for bl in batch_list:
            p = np.bincount(local_metadata[bl]) / float(len(local_metadata))
            local_diversity = entropy(p)

            mean_local_diversity[bl].append(local_diversity)

    # Average local diversity over all points
    for bl in batch_list:
        mean_local_diversity[bl] = np.mean(mean_local_diversity[bl])
        
    return mean_local_diversity

# test metrics above
# ------------------
# X , y = make_multilabel_classification(n_samples=1000, n_classes=5)
# df_features = pd.DataFrame(X)
# metadata = pd.DataFrame(y, columns=["col1", "col2", "col3", "col4", "col5"])
# batch_list = ["col3", "col4", "col5"]

# print(mean_local_diversity(metadata, batch_list, df_features, k=10))