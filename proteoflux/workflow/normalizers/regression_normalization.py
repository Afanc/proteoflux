import numpy as np
from typing import Tuple, List
from sklearn.linear_model import RANSACRegressor
from skmisc.loess import loess

def compute_reference_values(
    mat: np.ndarray,
    scale: str = None,
    condition_labels: List[str] = None
) -> np.ndarray:
    """
    Returns an (n_prots x n_samps) reference matrix.
      - Global (scale is None or 'global'): tile the per-protein mean.
      - Local  (scale == 'local') : for each sample j, pick the first-sample of that sample’s condition.
    """
    n_prots, n_samps = mat.shape

    # Precompute per-protein means (for global tile and local‐fallback)
    prot_means = np.nanmean(mat, axis=1)

    # GLOBAL / NONE: tile the vector
    if scale is None or scale == "global":
        # shape (n_prots, n_samps)
        return np.repeat(prot_means[:, None], n_samps, axis=1)

    # LOCAL: require condition_labels
    if scale == "local":
        if condition_labels is None:
            raise ValueError("When scale='local', you must pass condition_labels")

        # Map each condition to its first-sample index
        cond2first: Dict[str,int] = {}
        for idx, cond in enumerate(condition_labels):
            cond2first.setdefault(cond, idx)

        # Build the reference matrix
        ref = np.empty((n_prots, n_samps), dtype=float)
        for j, cond in enumerate(condition_labels):
            first_idx = cond2first[cond]
            col = mat[:, first_idx]
            # fallback to prot_means where col is NaN
            ref[:, j] = np.where(np.isnan(col), prot_means, col)
        return ref

    raise ValueError(f"Unknown scale '{scale}'")

def impute_nan_with_median(mat: np.ndarray) -> np.ndarray:
    """
    Impute missing values using the per-protein (row-wise) median,
    without triggering warnings on all-NaN rows.
    """
    n_prots, n_samps = mat.shape

    # 1) Figure out which proteins have at least one real value
    has_data = np.any(~np.isnan(mat), axis=1)

    # 2) Prepare a column vector of medians, default NaN
    medians = np.full((n_prots, 1), np.nan, dtype=float)

    # 3) Compute the median only for those rows with data
    medians[has_data, 0] = np.nanmedian(mat[has_data, :], axis=1)

    # 4) Wherever mat is NaN, fill in the per-protein median
    return np.where(np.isnan(mat), medians, mat)

def compute_MA(sample: np.ndarray, reference_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MA coordinates: M = (sample + ref)/2, A = sample - ref
    """
    M = (sample + reference_values) / 2
    A = sample - reference_values
    return M, A

def fit_regression(X: np.ndarray, Y: np.ndarray, regression_type: str, span: float = 0.9):
    """
    Fit regression model to (X, Y). Supports 'linear' or 'loess'.
    Masks out any NaNs in X or Y before fitting, but still returns
    a preds array the same length as Y (with NaN for masked positions).
    """
    # 1) Build mask of valid points
    mask = (~np.isnan(X)) & (~np.isnan(Y))
    Xg, Yg = X[mask], Y[mask]

    # 2) Prep full-length preds
    preds = np.full_like(Y, np.nan)

    # 3) Fit on the good data
    if regression_type == "linear":
        model = RANSACRegressor(random_state=42)
        model.fit(Xg.reshape(-1, 1), Yg)
        preds[mask] = model.predict(Xg.reshape(-1, 1))

    elif regression_type == "loess":
        model = loess(Xg, Yg, span=span, surface="direct")
        model.fit()
        preds[mask] = model.outputs.fitted_values

    else:
        raise ValueError(f"Unsupported regression type: {regression_type}")

    return model, preds

def regression_normalization(
    mat: np.ndarray,
    scale: str = "global",
    regression_type: str = "loess",
    span: float = 0.9,
    condition_labels: List[str] = None
) -> Tuple[np.ndarray, list]:
    """
    Perform regression-based normalization on a matrix (proteins × samples):
      - scale="global" or None: each column uses the per-protein mean (tiled)
      - scale="local": each column uses its condition’s first-sample values
    You must pass `condition_labels` when using local scaling.
    Returns:
      mat_normalized: same shape as mat
      models: list of fitted regression models (one per sample)
    """
    # build full (n_prots × n_samps) reference matrix
    reference_matrix = compute_reference_values(mat, scale, condition_labels)

    # impute missing values for fitting
    mat_filled = impute_nan_with_median(mat)

    n_prots, n_samps = mat.shape
    mat_normalized = np.zeros_like(mat)
    models = []

    for j in range(n_samps):
        sample_vals = mat_filled[:, j]
        # get the j-th column of the reference
        ref_vals = reference_matrix[:, j]

        # MA-plot coords
        X, Y = compute_MA(sample_vals, ref_vals)

        # fit either loess or RANSAC linear
        model, predicted = fit_regression(X, Y, regression_type, span)
        models.append(model)

        # subtract the fit in original space, preserving NaNs
        mat_normalized[:, j] = np.where(
            np.isnan(mat[:, j]),
            np.nan,
            mat[:, j] - predicted
        )

    return mat_normalized, models

