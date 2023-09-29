import pandas as pd
from imageio.v2 import imread
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
from PIL import Image

from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.nn as nn

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from batchdetect.swin_transformer import swin_tiny_patch4_window7_224, ConvStem



def list_of_dict_to_dict(list_of_dicts):
    new_dict = dict()
    for one_dict in list_of_dicts:
        new_dict.update(one_dict)
    return new_dict


def first_and_second_order(metadata):
    feature_union = FeatureUnion([
                                ("MaskBasedFeatures",
                                 FirstAndSecondLevelFeatures())
                                ])
    pipeline = Pipeline([("features", feature_union)], verbose=3)

    feature_extractor = FeatureExtractor(pipeline, model)
    list_of_features = feature_extractor.extract_features(metadata)
    df_features = pd.DataFrame(list_of_features)
    return df_features


class FeatureExtractor(object):
    def __init__(self, feature_unions, model=None):
        self.feature_unions = feature_unions
        
        self.model = model

    def extract_(self, f):
        image = Image.read(f)
        if self.model is not None:
            lowres = F.interpolate(image, size=(128, 128))
            z_content, (mu, _) = model.encoder(lowres)
            z_random = torch.randn_like(mu) * 2
            image = self.model.generator(image, z_content, z_random)
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
    def __init__(self, distances=[5], angles=[0], levels=256):
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
            glcm = graycomatrix(temp_image,
                                distances=self.distances,
                                angles=self.angles,
                                levels=self.levels)
            # storing the glcm values
            features["contrast_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='contrast')[0, 0]
            features["dissimilarity_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='dissimilarity')[0, 0]
            features["homogeneity_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='homogeneity')[0, 0]
            features["ASM_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='ASM')[0, 0]
            features["energy_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='energy')[0, 0]
            features["correlation_Ch" + str(ch+1)] = graycoprops(
                                                            glcm,
                                                            prop='correlation')[0, 0]
        return features


def feature_extraction(metadata, model, transform, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    file_list = metadata["file"].tolist()
    features = []
    for f in tqdm(file_list):
        image = Image.open(f).convert('RGB')

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image = transform(image).unsqueeze(0)
                image = image.to(device)
                features.append(model(image).squeeze().cpu().numpy())
    
    return pd.DataFrame(features)


def resnet(metadata, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load pre-trained ResNet model
    model = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
    model.to(device)
    model.eval()
    
    # Apply appropriate transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    return feature_extraction(metadata, model, transform, device)


def ctranspath(metadata, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load pre-trained ResNet model
    feature_extractor = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    feature_extractor.head = nn.Identity()

    ctranspath = torch.load('/home/ubuntu/models/ctranspath.pth')
    feature_extractor.load_state_dict(ctranspath['model'], strict=True)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Apply appropriate transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return feature_extraction(metadata, feature_extractor, transform, device)


# test the functions above
# ------------------------
# from pathlib import Path
# dataset = 'CRC'
# base_dir = Path(f'/home/ubuntu/data/BatchDetectData/BatchDetect{dataset}')
# metadata_path = Path(base_dir / 'metadata.csv')
# metadata = pd.read_csv(metadata_path)

# first_and_second_order(metadata)
# resnet(metadata)
# ctranspath(metadata)
