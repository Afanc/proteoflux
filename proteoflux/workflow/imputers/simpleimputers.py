import numpy as np

class RowMedianImputer:
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.row_medians_ = np.nanmedian(X, axis=1)
        # Fallback for rows with all NaNs
        self.row_medians_[np.isnan(self.row_medians_)] = 0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nan_mask = np.isnan(X_imputed)
        for i in range(X.shape[0]):
            X_imputed[i, nan_mask[i]] = self.row_medians_[i]
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class RowMeanImputer:
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.row_means_ = np.nanmean(X, axis=1)
        self.row_means_[np.isnan(self.row_means_)] = 0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imputed = X.copy()
        nan_mask = np.isnan(X_imputed)
        for i in range(X.shape[0]):
            X_imputed[i, nan_mask[i]] = self.row_means_[i]
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

