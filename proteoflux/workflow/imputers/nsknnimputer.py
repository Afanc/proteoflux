import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances


class NSkNNImputer(BaseEstimator, TransformerMixin):
    """
    No-Skip kNN Imputer (NS-kNN)

    This imputer uses the average of k nearest neighbors to fill missing values.
    Unlike standard kNN, it includes neighbors with missing values and replaces their missing feature
    with the minimum observed value of that feature. Distances are computed on autoscaled data.

    Attributes:
        n_neighbors (int): Number of neighbors to use.
    """

    def __init__(self, n_neighbors=6):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self._X = X.copy()

        # Compute feature-wise statistics
        self._min_per_feature = np.nanmin(X, axis=0)
        self._mean_per_feature = np.nanmean(X, axis=0)
        self._std_per_feature = np.nanstd(X, axis=0)
        self._std_per_feature[self._std_per_feature == 0] = 1.0  # avoid division by zero

        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nan_mask = np.isnan(X)
        n_samples, n_features = X.shape

        # Initialize an empty distance matrix
        distances = np.empty((n_samples, n_samples))

        # Compute pairwise distances while ignoring features with missing values in either sample
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    distances[i, j] = np.inf  # Set self-distance to infinity
                else:
                    # Identify common indices where both samples have observed data
                    common = ~nan_mask[i] & ~nan_mask[j]
                    n_common = np.sum(common)
                    if n_common == 0:
                        # If no features are in common, set distance to infinity
                        distances[i, j] = np.inf
                    else:
                        # Autoscale only the common features
                        xi = (X[i, common] - self._mean_per_feature[common]) / self._std_per_feature[common]
                        xj = (X[j, common] - self._mean_per_feature[common]) / self._std_per_feature[common]
                        # Compute Euclidean distance normalized by sqrt(n_common)
                        d = np.linalg.norm(xi - xj) / np.sqrt(n_common)
                        distances[i, j] = d

        # Impute missing values using the k-nearest neighbors
        for i in range(n_samples):
            # Get the indices of the n_neighbors smallest distances
            neighbors_idx = np.argsort(distances[i])[:self.n_neighbors]
            for j in range(n_features):
                if nan_mask[i, j]:
                    neighbor_vals = []
                    for n_idx in neighbors_idx:
                        neighbor_val = X[n_idx, j]
                        if np.isnan(neighbor_val):
                            # Use the precomputed minimum value for this feature
                            neighbor_vals.append(self._min_per_feature[j])
                        else:
                            neighbor_vals.append(neighbor_val)
                    X_imputed[i, j] = np.mean(neighbor_vals)

        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
