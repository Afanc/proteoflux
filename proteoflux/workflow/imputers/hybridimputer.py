import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances
from scipy.stats import truncnorm
from typing import Optional, List, Tuple

class HybridImputer:
    def __init__(self,
                 condition_map: pd.DataFrame,
                 group_column: str = "CONDITION",
                 sample_index: Optional[List[str]] = None,
                 knn_k: int = 6,
                 knn_weights: str = "distance",
                 knn_metric: str = "nan_euclidean",
                 gaussian_clip: float = 0.5,
                 gaussian_left_shift: float = 1.8,
                 random_state: int = 42):

        self.knn_k = knn_k
        self.knn_weights = knn_weights
        self.knn_metric = knn_metric
        self.gaussian_clip = gaussian_clip
        self.gaussian_left_shift = gaussian_left_shift
        self.random_state = random_state

        self.sample_index = sample_index
        self.group_column = group_column

        if sample_index is None:
            raise ValueError("sample_index is required to match sample order to condition map")

        condition_map = condition_map.set_index("Sample")
        self.conditions = np.array([
            condition_map.loc[sample, group_column] for sample in sample_index
        ])

        self.groups = np.unique(self.conditions)
        self.knn_model = None

    def fit(self, X):
        # Fit a standard KNN imputer
        self.knn_model = KNNImputer(
            n_neighbors=self.knn_k,
            metric=self.knn_metric,
            weights=self.knn_weights,
        )
        self.knn_model.fit(X.T)
        return self

    def transform(self, X):
        """
        Perform hybrid imputation: Perseus‐style truncated Gaussian for MNAR,
        (optionally QRILC), then KNN for the rest.
        """
        X = X.copy()
        n_features, n_samples = X.shape

        mask = np.isnan(X)

        # Perseus‐style MNAR fill (feature‐wise down‐shifted, narrow Gaussian)
        for i in range(n_features):
            mask_i = mask[i, :]
            if not mask_i.any():
                continue

            vals = X[i, ~mask_i]
            if len(vals) < 2:
                continue

            μ = np.mean(vals)
            σ = np.std(vals, ddof=1)
            lod = np.min(vals)
            shift = self.gaussian_left_shift
            width = self.gaussian_clip

            # guard against zero or non‐finite σ
            if σ <= 0 or not np.isfinite(σ):
                continue

            loc = μ - shift * σ
            scale = width * σ
            # if scale still somehow non‐positive, skip
            if scale <= 0:
                continue

            a = (lod - loc) / scale
            b = np.inf

            imputations = truncnorm.rvs(a,
                                        b,
                                        loc=loc,
                                        scale=scale,
                                        size=mask_i.sum())
            X[i, mask_i] = imputations

        # QRILC fallback - TODO check if good
        if getattr(self, "use_qrilc", False):
            try:
                # if you have an impyute‐style QRILC:
                from impyute.imputation.cs import QRILC
                X = QRILC(X)
            except ImportError:
                # or via rpy2 into R’s DEP::QRILC
                import rpy2.robjects as ro
                ro.r('library(DEP)')
                ro.globalenv['Xmat'] = ro.r['as.matrix'](ro.FloatVector(X.flatten()))
                ro.r('res <- DEP::QRILC(matrix(Xmat, nrow=%d, byrow=TRUE))' % n_features)
                X = np.array(ro.r('res$imputed')[0]).reshape(n_features, n_samples)

        # finally KNN for whatever’s still missing
        X_imputed = self.knn_model.transform(X.T).T

        # sanity check
        assert X_imputed.shape == X.shape, \
               f"imputed shape {X_imputed.shape} != input shape {X.shape}"
        return X_imputed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

