from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import numpy as np


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

def compute_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_feature_names(transformer, feature_names_in):
    return ["ratio"]
# Pipelines for different feature types
def make_ratio_pipeline():
    """
    Pipeline for computing ratios between two features with imputation and scaling.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(compute_ratio, feature_names_out=ratio_feature_names),
        StandardScaler()
    )


def make_log_pipeline():
    """
    Pipeline for log-transforming features with imputation and scaling.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),  # log1p for better stability
        StandardScaler()
    )


def make_categorical_pipeline():
    """
    Pipeline for categorical features: imputation and one-hot encoding.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )


def make_default_numeric_pipeline():
    """
    Default numeric preprocessing pipeline with imputation and scaling.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
