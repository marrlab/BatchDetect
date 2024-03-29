import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import mannwhitneyu
from umap import UMAP

import anndata
import scanpy as sc


def pca_emb(features, n_components=2):
    X_emb = StandardScaler().fit_transform(features)
    X_emb = PCA(n_components=n_components).fit_transform(X_emb)
    return numpy_to_dataframe(X_emb)


def tsne_emb(features, n_components=2):
    X_emb = StandardScaler().fit_transform(features)
    X_emb = TSNE(n_components).fit_transform(X_emb)
    return numpy_to_dataframe(X_emb)


def umap_emb(features, n_components=2):
    X_emb = StandardScaler().fit_transform(features)
    X_emb = UMAP(n_components=n_components).fit_transform(X_emb)
    return numpy_to_dataframe(X_emb)


def numpy_to_dataframe(X_emb):
    n_col = X_emb.shape[1]
    X_emb = pd.DataFrame(X_emb,
                         columns=["PC" + str(i+1) for i in range(n_col)])
    return X_emb


class BatchDetect():
    """
    BatchDetect
    This class takes the metadata and features from any dataset
    and analyze whether there is a clear batch effect in the data.
    """
    def __init__(self, metadata, features):
        """
        Initializing the class with metadata and features.
        inputs:
            metadata (pd.DataFrame): The metadata needs to have dimension
            of (N,M) where N is the number of sampels and M is
            the number covariates.

            features (pd.DataFrame): The features need to have the dimension of
            (N,F) where N is the number of sampels and
            F is the number of features.
        """
        self.metadata = metadata
        self.features = features

        # create an anndata with features as the main matrix and each column of metadata as an obs column
        self.adata = anndata.AnnData(X=self.features)
        for col in self.metadata.columns:
            self.adata.obs[col] = self.metadata.loc[:, col].tolist()


    def low_dim_visualization(self, method="pca", save=None):
        """
        low dimensional visualization creates a 2D representation of the
        provided features and generates scatterplots based on the covariates
        in the metadata
        inputs:
            method (str): one of three options ["pca", "tsne", "umap"]
            save (str): filename to save the plot
        """
        assert method.lower() in ["pca", "tsne", "umap"]
        # sc.set_figure_params(figsize=(4, 4))

        if method == "pca":
            # compute pca embedding for self.adata using scanpy
            sc.tl.pca(self.adata)
            sc.pl.pca(self.adata, color=self.metadata.columns.tolist(), save=save)
        elif method == "tsne":
            # compute tsne embedding for self.adata using scanpy
            sc.tl.tsne(self.adata)
            sc.pl.tsne(self.adata, color=self.metadata.columns.tolist(), save=save)
        elif method == "umap":
            # compute umap embedding for self.adata using scanpy
            sc.pp.neighbors(self.adata)
            sc.tl.umap(self.adata)
            sc.pl.umap(self.adata, color=self.metadata.columns.tolist(), save=save)

    def prince_plot(self, n_components=5):
        """
        prince plot first generates a PCA embedding of the features. then
        it runs an ANOVA test of every principal componenet (PC) vs. the
        covariates of the metadata
        inputs:
            n_components (int): number of PCs in the PCA output
        """

        X_emb = pca_emb(self.features,
                        n_components=n_components)

        ncols = self.metadata.shape[1]
        nrows = n_components

        column_names = ["PC" + str(i) for i in range(n_components)]
        heatmap_mat = pd.DataFrame(0.,
                                   columns=self.metadata.columns,
                                   index=column_names)

        for i in range(nrows):
            for j in range(ncols):
                mask = self.metadata.iloc[:, j].notnull()  # mask for nan values
                p_value = f_classif(X_emb.iloc[:, i:i+1][mask],
                                    self.metadata.iloc[:, j][mask])[1][0]
                heatmap_mat.iloc[i, j] = p_value

        fig, ax = plt.subplots(figsize=(1*ncols, 1*nrows))

        ax = sns.heatmap(heatmap_mat,
                         cmap="Greys_r",
                         annot=True,
                         fmt=".3f",
                         ax=ax,
                         vmin=0,
                         vmax=0.05)

    def classification_test(self,
                            scorer="accuracy",
                            n_splits=5,
                            n_repeats=10):
        """
        classification test runs two classifiers. One is a DummyClassifier from
        sklearn with uniform strategy and the next one is a
        RandomForestClassifier. The target variables are based on the covariate
        in the metadata dataframe. If the Randomforst works better than the
        DummyClassifier, it means that there is a systematic batch effect for
        that specific covariate.
        inputs:
            scorer (str): the metric for the classification performance
            n_splits (int): number of splits of the data for
            RepeatedStratifiedKFold
            n_repeats (int): number of repeats of the data for
            RepeatedStratifiedKFold

        """

        results = dict()

        results["covariate"] = []
        results[scorer] = []
        results["method"] = []

        rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                                       n_repeats=n_repeats)

        for col in self.metadata.columns:
            mask = self.metadata.loc[:, col].notnull()
            dc = DummyClassifier(strategy="uniform")
            random_results = cross_val_score(estimator=dc,
                                             X=self.features[mask],
                                             y=self.metadata.loc[:, col][mask],
                                             cv=rskf,
                                             scoring=scorer,
                                             verbose=1)

            results["covariate"] += [col]*n_splits*n_repeats
            results["method"] += ["random"]*n_splits*n_repeats
            results[scorer] += random_results.tolist()

            rf = RandomForestClassifier(n_jobs=-1)
            rf_results = cross_val_score(estimator=rf,
                                         X=self.features[mask],
                                         y=self.metadata.loc[:, col][mask],
                                         cv=rskf,
                                         scoring=scorer,
                                         verbose=1)

            results["covariate"] += [col]*n_splits*n_repeats
            results["method"] += ["random forest"]*n_splits*n_repeats
            results[scorer] += rf_results.tolist()

        results = pd.DataFrame(results)

        for cov in results.covariate.unique():
            row_index_random = results["covariate"] == cov
            row_index_random = row_index_random & (results["method"]=="random")
            random_scores = results.loc[row_index_random, scorer].tolist()

            row_index_rf = results["covariate"] == cov
            row_index_rf = row_index_rf & (results["method"]=="random forest")
            rf_scores = results.loc[row_index_rf, scorer].tolist()

            print("comparing", cov, ":", mannwhitneyu(random_scores, rf_scores))

        fig, ax = plt.subplots(figsize=(len(self.metadata.columns), 3))

        ax = sns.boxplot(data=results,
                         x="covariate",
                         y=scorer,
                         hue="method",
                         ax=ax)
        ax.legend(loc=(1.04, 0))
