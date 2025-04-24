import numpy as np
from sklearn.linear_model import RANSACRegressor
from skmisc.loess import loess
from typing import Tuple
# from statsmodels.nonparametric.smoothers_lowess import lowess  # if needed

def compute_reference_values(mat: np.ndarray, scale: str) -> np.ndarray:
    if scale == "global":
        return np.nanmean(mat, axis=0)
    ref = mat[0, :]
    return np.where(np.isnan(ref), np.nanmean(mat, axis=0), ref)

def impute_nan_with_median(mat: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(mat, axis=0, keepdims=True)
    return np.where(np.isnan(mat), medians, mat)

def compute_MA(sample: np.ndarray, reference_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = (reference_values + sample) / 2
    A = sample - reference_values
    return M, A

def fit_regression(X: np.ndarray, Y: np.ndarray, regression_type: str, span: float):
    if regression_type == "linear":
        model = RANSACRegressor(random_state=42)
        model.fit(X.reshape(-1, 1), Y)
        predicted = model.predict(X.reshape(-1, 1))
    elif regression_type == "loess":
        sort_idx = np.argsort(X)
        X_sorted, Y_sorted = X[sort_idx], Y[sort_idx]
        model = loess(X_sorted, Y_sorted, span=span)
        model.fit()
        pred_sorted = model.outputs.fitted_values
        predicted = np.empty_like(pred_sorted)
        predicted[sort_idx] = pred_sorted
    else:
        raise ValueError("Invalid regression_type.")
    return model, predicted

def regression_normalization(
    mat:np.ndarray,
    scale="global",
    regression_type="linear",
    span=0.9
    ):
    reference_values = compute_reference_values(mat, scale)
    mat_filled = impute_nan_with_median(mat)

    mat_normalized = np.zeros_like(mat)
    models = []

    for i in range(mat.shape[0]):
        X, Y = compute_MA(mat_filled[i, :], reference_values)
        model, predicted = fit_regression(X, Y, regression_type, span)
        models.append(model)

        mat_normalized[i, :] = np.where(
            np.isnan(mat[i, :]), np.nan, mat[i, :] - predicted
        )

    return mat_normalized, models

def regression_normalization2(
    mat: np.ndarray,
    scale: str = "global",
    regression_type: str = "linear",
    span: float = 0.9
    ):
    """
    Perform regression-based normalization on a matrix with shape (samples x proteins).

    Parameters:
        mat (np.ndarray): 2D array (samples x proteins) with NaNs.
        scale (str): "global" uses the global mean per protein; otherwise, use first sample as reference.
        regression_type (str): "linear" (RANSAC) or "loess".
        span (float): Smoothing parameter for loess.

    Returns:
        Tuple[np.ndarray, list]: Normalized matrix (same shape as input) and list of regression models.
    """
    # Compute reference values per protein:
    if scale == "global":
        reference_values = np.nanmean(mat, axis=0)  # now per protein (columns)
    else:
        reference_values = mat[0, :]
        reference_values = np.where(np.isnan(reference_values), np.nanmean(mat, axis=0), reference_values)

    # Impute NaNs using per-protein medians:
    protein_medians = np.nanmedian(mat, axis=0, keepdims=True)
    mat_filled = np.where(np.isnan(mat), protein_medians, mat)

    # Initialize output matrix and model list
    mat_normalized = np.zeros_like(mat)
    models = []

    # Loop over each sample (rows)
    for i in range(mat.shape[0]):
        # Calculate the mean intensity per protein for the sample:
        X = (reference_values + mat_filled[i, :]) / 2
        Y = mat_filled[i, :] - reference_values

        if regression_type == "linear":
            model = RANSACRegressor(random_state=42)
            model.fit(X.reshape(-1, 1), Y)
            predicted = model.predict(X.reshape(-1, 1))
        elif regression_type == "loess":
            # For loess, sort the X values (and corresponding Y values) first
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            Y_sorted = Y[sort_idx]
            model = loess(X_sorted, Y_sorted, span=span)
            model.fit()
            pred_sorted = model.outputs.fitted_values
            # Unsort predictions to the original order:
            predicted = np.empty_like(pred_sorted)
            predicted[sort_idx] = pred_sorted
        else:
            raise ValueError("Invalid regression_type. Choose 'linear' or 'loess'.")

        models.append(model)
        # Subtract the predicted bias from the raw values, preserving NaNs
        mat_normalized[i, :] = np.where(np.isnan(mat[i, :]), np.nan, mat[i, :] - predicted)

    return mat_normalized, models
