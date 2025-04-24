import numpy as np
from sklearn.linear_model import RANSACRegressor
from skmisc.loess import loess
# from statsmodels.nonparametric.smoothers_lowess import lowess  # if needed

def regression_normalization(mat: np.ndarray, scale: str = "global", regression_type: str = "linear", span: float = 0.9):
    """
    Perform regression-based normalization on a matrix.

    Parameters:
        mat (np.ndarray): 2D array (proteins x samples) with NaNs.
        scale (str): "global" uses the global mean per protein; otherwise, use first sample as reference.
        regression_type (str): "linear" (RANSAC) or "loess".
        span (float): Smoothing parameter for loess.

    Returns:
        Tuple[np.ndarray, list]: Normalized matrix (same shape as input) and list of regression models.
    """
    # Compute reference values:
    if scale == "global":
        reference_values = np.nanmean(mat, axis=1)  # per protein
    else:
        reference_values = mat[:, 0]
        reference_values = np.where(np.isnan(reference_values), np.nanmean(mat, axis=1), reference_values)

    # Impute NaNs using per-protein medians:
    protein_medians = np.nanmedian(mat, axis=1, keepdims=True)
    mat_filled = np.where(np.isnan(mat), protein_medians, mat)

    # Initialize output matrix and model list
    mat_normalized = np.zeros_like(mat)
    models = []

    for i in range(mat.shape[1]):
        # Mean intensity per protein:
        X = (reference_values + mat_filled[:, i]) / 2
        Y = mat_filled[:, i] - reference_values

        if regression_type == "linear":
            model = RANSACRegressor(random_state=42)
            model.fit(X.reshape(-1, 1), Y)
            predicted = model.predict(X.reshape(-1, 1))
        elif regression_type == "loess":
            # Implement or wrap your loess routine here
            model = loess(X, Y, span=span)
            model.fit()
            predicted = model.outputs.fitted_values
        else:
            raise ValueError("Invalid regression_type. Choose 'linear' or 'loess'.")

        models.append(model)
        # Normalize: subtract predicted bias while retaining missing values
        mat_normalized[:, i] = np.where(np.isnan(mat[:, i]), np.nan, mat[:, i] - predicted)

    return mat_normalized, models


