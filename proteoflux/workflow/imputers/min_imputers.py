import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-8

class MinDetImputer(BaseEstimator, TransformerMixin):
    """
    Deterministic left-censored imputation (MinDet).
    Replaces NaNs per column with a low value (column-wise LoD) minus a small shift.
    Assumes X is log-scale, shape = (n_features, n_samples).
    """

    def __init__(self, quantile=0.01, shift=0.2):
        """
        Args:
            quantile (float): Low-tail quantile per column used as LoD (e.g., 0.01).
            shift (float): Subtract this from LoD on the log scale.
        """
        self.quantile = float(quantile)
        self.shift = float(shift)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        # Column-wise low quantile as LoD; fallback to global min if column is all-NaN
        col_has_val = np.any(~np.isnan(X), axis=0)
        lod = np.full(X.shape[1], np.nan, dtype=np.float64)
        if np.any(col_has_val):
            lod[col_has_val] = np.nanquantile(X[:, col_has_val], self.quantile, axis=0)
        # Fallbacks
        global_min = np.nanmin(X) if np.isnan(lod).any() else None
        lod[~col_has_val] = global_min if global_min is not None else 0.0
        # Final fill value per column
        self.col_fill_ = lod - self.shift
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imp = X.copy()
        rows, cols = np.where(np.isnan(X_imp))
        if len(rows):
            X_imp[rows, cols] = self.col_fill_[cols]
        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinProbImputer(BaseEstimator, TransformerMixin):
    """
    Probabilistic left-censored imputation (MinProb).
    Replaces NaNs per column by drawing from N(mu_c, sigma_c), where
    mu_c = q_low_c - mu_shift * sd_c, and sigma_c = sd_width * sd_c.
    Assumes X is log-scale, shape = (n_features, n_samples).
    """

    def __init__(self, quantile=0.01, mu_shift=1.8, sd_width=0.3, random_state=42, clip_to_quantile=True):
        """
        Args:
            quantile (float): Low-tail quantile per column (e.g., 0.01).
            mu_shift (float): How far below the low-tail mean to center (in SDs).
            sd_width (float): Fraction of column SD used as sampling SD.
            random_state (int|None): RNG seed for reproducibility.
            clip_to_quantile (bool): If True, clip draws to <= the low quantile.
        """
        self.quantile = float(quantile)
        self.mu_shift = float(mu_shift)
        self.sd_width = float(sd_width)
        self.random_state = random_state
        self.clip_to_quantile = bool(clip_to_quantile)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        col_has_val = np.any(~np.isnan(X), axis=0)
        qlow = np.full(X.shape[1], np.nan, dtype=np.float64)
        sd   = np.full(X.shape[1], np.nan, dtype=np.float64)

        if np.any(col_has_val):
            qlow[col_has_val] = np.nanquantile(X[:, col_has_val], self.quantile, axis=0)
            with np.errstate(invalid="ignore"):
                sd[col_has_val] = np.nanstd(X[:, col_has_val], axis=0, ddof=1)

        # Fallbacks for degenerate columns
        global_min = np.nanmin(X) if np.isnan(qlow).any() else 0.0
        qlow[~col_has_val] = global_min
        sd[np.isnan(sd) | (sd < _EPS)] = np.nanstd(X, ddof=1) if np.isfinite(np.nanstd(X, ddof=1)) else 1.0
        sd[(~np.isnan(sd)) & (sd < _EPS)] = 1.0

        mu = qlow - self.mu_shift * sd
        sigma = np.maximum(self.sd_width * sd, _EPS)

        self.col_mu_ = mu
        self.col_sigma_ = sigma
        self.col_qlow_ = qlow
        self._rng = rng
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imp = X.copy()
        rows, cols = np.where(np.isnan(X_imp))
        if len(rows):
            draws = self._rng.normal(loc=self.col_mu_[cols], scale=self.col_sigma_[cols], size=len(rows))
            if self.clip_to_quantile:
                # keep imputations at/below the estimated low quantile to respect left-censor idea
                draws = np.minimum(draws, self.col_qlow_[cols])
            X_imp[rows, cols] = draws
        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

