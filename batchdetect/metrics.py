import itertools

import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn import metrics
from sklearn.datasets import make_multilabel_classification

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


def mean_local_diversity(metadata, batch_list, df_features, k=10):
    """
    computes the Shannon diversity for each point in the dataset over its k nearest neighbors
    
    inputs:
        metadata:   pandas dataframe with all the possible covariates
        batch_list: a list including the covariates to be used for the
                    batch effect correction. These covariates should be
                    a column name in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        metrics (dict): dictionary with the calculated metrics
    """
    num_points = len(df_features)
    mean_local_diversity = {bl: [] for bl in batch_list}
    
    # Convert batch labels to integers
    for bl in batch_list:
        label_mapping = {label: i for i, label in enumerate(metadata[bl].unique())}
        metadata[bl] = metadata[bl].map(label_mapping)

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


def silhouette_score(metadata, batch_list, df_features):
    """
    Compute the mean Silhouette Coefficient of all samples.
    
    inputs:
        metadata:   pandas dataframe with all the possible covariates
        batch_list: a list including the covariates to be used for the
                    batch effect correction. Thse covariates should be
                    a column name in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        metrics (dict): dictionary with the calculated metrics
    """
    covariate_specific_silhoutte_score = {batch_label: [] for batch_label in batch_list}

    # compute the silhoutte score for each batch
    for batch_label in batch_list:
        covariate_specific_silhoutte_score[batch_label] = metrics.silhouette_score(df_features, metadata[batch_label])

    return covariate_specific_silhoutte_score


def kullback_leibler_divergence(metadata, batch_list, df_features):
    """
    calculates the pairwise kl-div between all given covariates

    inputs:
        metadata:    pandas dataframe with all the possible covariates
        batch_list:  a list including the covariates to be used for the
                     batch effect correction. These covariates should be
                     column names in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        metrics (dict): dictionary with the calculated pairwise metircs
    """
    metrics = {}

    for var1, var2 in itertools.combinations(batch_list, 2):

        # extract feature subsets based on metadata batch labels
        subset1 = df_features[metadata[var1] == 1]
        subset2 = df_features[metadata[var2] == 1]

        # calculate the empirical probabilities
        p = subset1.value_counts(normalize=True)
        q = subset2.value_counts(normalize=True)

        # make sure that p and q have the same index
        p, q = p.align(q, fill_value=0)
        p, q = p.values, q.values

        # compute KL divergence
        mask = (p != 0) & (q != 0)
        kl_div = np.sum(p[mask] * np.log(p[mask] / q[mask]))
        metrics[(var1, var2)] = kl_div

    return metrics
    
    
def jensen_shannon_divergence(metadata, batch_list, df_features):
    """
    calculates the pairwise js-div between all given covariates

    inputs:
        metadata:    pandas dataframe with all the possible covariates
        batch_list:  a list including the covariates to be used for the
                     batch effect correction. These covariates should be
                     column names in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        metrics (dict): dictionary with the calculated pairwise metrics
    """
    metrics = {}

    for var1, var2 in itertools.combinations(batch_list, 2):

        # extract feature subsets based on metadata batch labels
        subset1 = df_features[metadata[var1] == 1]
        subset2 = df_features[metadata[var2] == 1]

        # calculate the empirical probabilities
        p = subset1.value_counts(normalize=True)
        q = subset2.value_counts(normalize=True)

        # make sure that p and q have the same index
        p, q = p.align(q, fill_value=0)
        p, q = p.values, q.values

        # calculate m distribution
        m = (p + q) / 2

        # compute JS divergence
        mask = (p != 0) & (q != 0) & (m != 0)
        
        kl_pm = np.sum(p[mask] * np.log(p[mask] / m[mask]))
        kl_qm = np.sum(q[mask] * np.log(q[mask] / m[mask]))
        
        js_div = (kl_pm + kl_qm) / 2
        metrics[(var1, var2)] = js_div

    return metrics


def frechet_inception_distance(metadata, batch_list, df_features):
    """
    calculates the pairwise fid between all given covariates

    inputs:
        metadata:    pandas dataframe with all the possible covariates
        batch_list:  a list of covariates to be used for the
                     batch effect correction. These covariates should be
                     column names in the metadata dataframe
        df_features: pandas dataframe with the numerical features

    outputs:
        pairwise_fid (dict): dictionary with the calculated pairwise metrics
    """

    pairwise_fid = {}

    for var1, var2 in itertools.combinations(batch_list, 2):

        # extract feature subsets based on metadata batch labels
        subset1 = df_features[metadata[var1] == 1]
        subset2 = df_features[metadata[var2] == 1]

        # calculate mean and covariance statistics for subset1
        mu1, sigma1 = subset1.mean(axis=0), np.cov(subset1, rowvar=False)
        
        # calculate mean and covariance statistics for subset2
        mu2, sigma2 = subset2.mean(axis=0), np.cov(subset2, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)

        # calculate sqrt of product between covariances
        covmean = sqrtm(sigma1.dot(sigma2))

        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # calculate the FID score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
        pairwise_fid[(var1, var2)] = fid

    return pairwise_fid


# minimal example of how to extract features for FID
model = torchvision.models.inception_v3(weights='DEFAULT')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(batch, model):
    batch = batch.to(device)
    with torch.no_grad():
        pred = model(batch)
    return pred.cpu().numpy()


# test metrics above
# ------------------
X , y = make_multilabel_classification(n_samples=1000, n_classes=5)
df_features = pd.DataFrame(X)
metadata = pd.DataFrame(y, columns=["col1", "col2", "col3", "col4", "col5"])
batch_list = ["col3", "col4", "col5"]

mean_local_diversity(metadata, batch_list, df_features, k=10)
silhouette_score(metadata, batch_list, df_features)
kullback_leibler_divergence(metadata, batch_list, df_features)
jensen_shannon_divergence(metadata, batch_list, df_features)
frechet_inception_distance(metadata, batch_list, df_features)
