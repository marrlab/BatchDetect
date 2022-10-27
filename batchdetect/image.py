import pandas as pd
from imageio import imread
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy


def list_of_dict_to_dict(list_of_dicts):
    new_dict = dict()
    for one_dict in list_of_dicts:
        new_dict.update(one_dict)
    return new_dict


def automatic_feature_extraction(metadata):
    feature_union = FeatureUnion([
                                ("MaskBasedFeatures",
                                 FirstAndSecondLevelFeatures())
                                ])
    pipeline = Pipeline([("features", feature_union)], verbose=3)

    feature_extractor = FeatureExtractor(pipeline)
    list_of_features = feature_extractor.extract_features(metadata)
    df_features = pd.DataFrame(list_of_features)
    return df_features


class FeatureExtractor(object):
    def __init__(self, feature_unions):
        self.feature_unions = feature_unions

    def extract_(self, f):
        image = imread(f)
        features = self.feature_unions.transform([image]).copy()
        features = list_of_dict_to_dict(features)
        return features

    def extract_features(self, metadata, n_jobs=-1):
        file_list = metadata["file"].tolist()
        results = Parallel(n_jobs=n_jobs)(delayed(self.extract_)(f)
                                          for f in tqdm(file_list, position=0,
                                          leave=True))
        return results


class FirstAndSecondLevelFeatures(BaseEstimator, TransformerMixin):

    def __init__(self,
                 distances=[5],
                 angles=[0],
                 levels=256):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        image = X[0].copy()
        # storing the feature values
        features = dict()
        for ch in range(image.shape[2]):
            # First Order Features
            features["mean_intensity_Ch" + str(ch+1)] = image[:, :, ch].mean()
            features["std_intensity_Ch" + str(ch+1)] = image[:, :, ch].std()
            features["kurtosis_intensity_Ch" + str(ch+1)] = kurtosis(image[:,:,ch].ravel())
            features["skew_intensity_Ch" + str(ch+1)] = skew(image[:, :, ch].ravel())
            features["min_intensity_Ch" + str(ch+1)] = image[:, :, ch].min()
            features["max_intensity_Ch" + str(ch+1)] = image[:, :, ch].max()
            features["shannon_entropy_Ch" + str(ch+1)] = shannon_entropy(image[:,:,ch])

            # Second Order Features
            # create a 2D temp image
            temp_image = image[:, :, ch].copy()
            # use 8bit pixel values for GLCM
            temp_image = (temp_image/temp_image.max())*255
            # convert to unsigned for GLCM
            temp_image = temp_image.astype('uint8')
            # calculating glcm
            glcm = greycomatrix(temp_image,
                                distances=self.distances,
                                angles=self.angles,
                                levels=self.levels)
            # storing the glcm values
            features["contrast_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='contrast')[0, 0]
            features["dissimilarity_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='dissimilarity')[0, 0]
            features["homogeneity_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='homogeneity')[0, 0]
            features["ASM_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='ASM')[0, 0]
            features["energy_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='energy')[0, 0]
            features["correlation_Ch" + str(ch+1)] = greycoprops(
                                                            glcm,
                                                            prop='correlation')[0, 0]
        return features
