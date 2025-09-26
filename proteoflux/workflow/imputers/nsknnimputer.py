import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NSkNNImputer(BaseEstimator, TransformerMixin):
    """
    No-Skip kNN Imputer (NS-kNN), adapted to ProteoFlux matrix shape.

    Matrix shape: X is (n_features, n_samples)  [rows = proteins/features, cols = runs/samples]

    Idea (kept from your original code):
      - Compute pairwise distances between SAMPLES (columns), autoscaling per FEATURE.
      - Include neighbors even if they have missing at the target feature; for those, use
        the feature's minimum observed value as a stand-in.
      - Impute X[feature, sample] with the mean of that feature across the k nearest neighbor SAMPLES.

    Attributes:
        n_neighbors (int): Number of neighbors to use.
    """

    def __init__(self, n_neighbors=6):
        self.n_neighbors = int(n_neighbors)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)                 # (n_features, n_samples)
        self.n_features_, self.n_samples_ = X.shape

        # Per-FEATURE stats (across samples) — needed for autoscaling & fallback value
        # axis=1 because rows are features
        self._min_per_feature  = np.nanmin(X, axis=1)       # shape (n_features,)
        self._mean_per_feature = np.nanmean(X, axis=1)      # shape (n_features,)
        self._std_per_feature  = np.nanstd(X, axis=1)       # shape (n_features,)
        self._std_per_feature[self._std_per_feature == 0] = 1.0  # avoid division by zero
        self._std_per_feature[np.isnan(self._std_per_feature)] = 1.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)                 # (n_features, n_samples)
        n_features, n_samples = X.shape
        if getattr(self, "n_features_", None) is None:
            # allow transform without a prior fit
            self.fit(X)

        if n_features != self.n_features_ or n_samples != self.n_samples_:
            # be explicit; distances depend on sample count
            raise ValueError(f"X shape changed: expected {(self.n_features_, self.n_samples_)}, got {X.shape}")

        X_imputed = X.copy()
        nan_mask = np.isnan(X)                              # (n_features, n_samples)

        # --- Pairwise distances BETWEEN SAMPLES (columns) ---
        # distance[i, j] is distance between sample i and sample j
        distances = np.empty((n_samples, n_samples), dtype=np.float64)

        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    distances[i, j] = np.inf
                    continue

                # common features observed in both samples i and j (column-wise selection)
                common = (~nan_mask[:, i]) & (~nan_mask[:, j])
                n_common = int(np.sum(common))
                if n_common == 0:
                    distances[i, j] = np.inf
                    continue

                # autoscale per FEATURE using precomputed row means/stds
                xi = (X[common, i] - self._mean_per_feature[common]) / self._std_per_feature[common]
                xj = (X[common, j] - self._mean_per_feature[common]) / self._std_per_feature[common]

                # Euclidean over common features, normalized by sqrt(n_common)
                d = np.linalg.norm(xi - xj) / np.sqrt(n_common)
                distances[i, j] = d

        # --- Impute missing cells using k nearest SAMPLES ---
        for s in range(n_samples):                          # target sample (column)
            # indices of k nearest neighbor samples to s
            neighbors_idx = np.argsort(distances[s])[:self.n_neighbors]
            for f in range(n_features):                     # feature (row)
                if nan_mask[f, s]:
                    # collect this feature's value from neighbor samples
                    neighbor_vals = []
                    for n_s in neighbors_idx:
                        v = X[f, n_s]
                        if np.isnan(v):
                            # stand-in: feature's min observed across samples (left-censor-ish)
                            neighbor_vals.append(self._min_per_feature[f])
                        else:
                            neighbor_vals.append(v)
                    X_imputed[f, s] = float(np.mean(neighbor_vals)) if neighbor_vals else self._min_per_feature[f]

        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

