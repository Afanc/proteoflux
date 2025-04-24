import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import truncnorm

class KNNTNImputer(BaseEstimator, TransformerMixin):
    """
    KNN-Truncation (KNN-TN) Imputer.

    Scales data using truncated normal estimates (via MLE) per feature,
    computes correlation-based distances between genes (features),
    and imputes missing values with a weighted average of k nearest neighbors.
    """

    def __init__(self, n_neighbors=6, perc=1.0):
        """
        Args:
            n_neighbors (int): Number of nearest neighbors to use.
            perc (float): Threshold for using sample mean/SD instead of truncated MLE (if missingness > perc).
        """
        self.n_neighbors = n_neighbors
        self.perc = perc

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_, self.n_samples_ = X.shape
        self.lod_ = np.nanmin(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.genemeans_ = np.nanmean(X, axis=1)
            self.genesd_ = np.nanstd(X, axis=1)

        # If a row is *completely* NaN
        all_na_mask = np.isnan(self.genemeans_)
        self.genemeans_[all_na_mask] = self.lod_  # fallback: set mean to LOD
        self.genesd_[all_na_mask] = 1.0           # assume 1.0 to not inflate/deflate

        # If std is 0 (or near), fallback to small noise
        self.genesd_[self.genesd_ == 0] = 1e-8
        self.genesd_[np.isnan(self.genesd_)] = 1e-8

        # Identify rows needing truncated MLE
        na_fraction = np.isnan(X).sum(axis=1) / self.n_samples_
        far_above_lod = self.genemeans_ > (3 * self.genesd_ + self.lod_)
        overly_missing = na_fraction >= self.perc
        mle_mask = ~(overly_missing | far_above_lod)

        self.trunc_mu_ = self.genemeans_.copy()
        self.trunc_sd_ = self.genesd_.copy()

        for i in np.where(mle_mask)[0]:
            observed = X[i, ~np.isnan(X[i])]
            if len(observed) < 3:
                continue
            std = np.std(observed)
            if std == 0:
                continue
            a, b = (self.lod_ - observed.mean()) / std, np.inf
            try:
                mu, sigma = truncnorm.fit(observed, floc=observed.mean(), fscale=std, a=a, b=b)
                self.trunc_mu_[i] = mu
                self.trunc_sd_[i] = sigma
            except Exception:
                continue

        # Fallback to sample mean/sd if MLE fails
        fallback_mu = self.genemeans_
        fallback_sd = self.genesd_

        self.trunc_mu_ = np.where(np.isnan(self.trunc_mu_), fallback_mu, self.trunc_mu_)
        self.trunc_sd_ = np.where((self.trunc_sd_ == 0) | np.isnan(self.trunc_sd_), 1e-8, self.trunc_sd_)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.trunc_mu_[:, None]) / self.trunc_sd_[:, None]

        imputed = X_scaled.copy()
        mask = np.isnan(X_scaled)

        # Replace constant rows (zero stddev) with small noise to avoid NaNs in corr
        with warnings.catch_warnings(): #we'll get warnings because no variance if all NAs
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_std = np.nanstd(X_scaled, axis=1)
        X_scaled[row_std == 0] += 1e-8

        # Compute correlation safely
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(np.nan_to_num(X_scaled))

        # Replace self-correlation and invalid rows
        np.fill_diagonal(corr_matrix, 0)
        corr_matrix[np.isnan(corr_matrix)] = 0

        for i in range(self.n_features_):
            for j in range(self.n_samples_):
                if mask[i, j]:
                    dists = 1 - np.abs(corr_matrix[i])
                    dists[np.isnan(dists)] = np.inf

                    k_idx = np.argsort(dists)[:self.n_neighbors]
                    values = X_scaled[k_idx, j]

                    # Handle edge case: all neighbors are nan
                    valid = ~np.isnan(values)
                    if valid.sum() == 0:
                        #imputed[i, j] = 0
                        imputed[i, j] = (self.lod_ - self.trunc_mu_[i]) / self.trunc_sd_[i]
                        #print(values, imputed[i,j])
                    else:
                        weights = 1 / (dists[k_idx][valid] + 1e-6)
                        weights /= weights.sum()
                        imputed[i, j] = np.dot(weights, values[valid])

                    #if imputed[i,j] < 0:
                    #    print(values)

        # Rescale back to original space
        imputed_rescaled = (imputed * self.trunc_sd_[:, None]) + self.trunc_mu_[:, None]

        return imputed_rescaled

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

