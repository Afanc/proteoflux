import numpy as np
import pandas as pd
from proteoflux.utils.utils import logger, log_time

@log_time("Apply Contrasts")
def apply_contrasts(fit_results, contrast_matrix, contrast_names=None):
    """
    Applies contrast matrix to fitted model results.

    Parameters:
    - fit_results: output of LinearModelFitter.get_results()
    - contrast_matrix: shape (p x m) → p = design coefficients, m = contrasts
    - contrast_names: list of contrast names, optional

    Returns:
    - contrast_df: DataFrame of shape (n_proteins x n_contrasts) with log2FC
    - se_df: DataFrame of same shape with standard errors
    """
    B = fit_results["coefficients"]      # (n_proteins x p)
    sigma2 = fit_results["residual_variance"]  # (n_proteins,)
    XtX_inv = fit_results["xtx_inv"]     # (p x p)

    C = contrast_matrix                  # (p x m)
    V = np.array([C.T @ XtX_inv @ C])    # shape: (1 x m x m), same for all

    # Log2FCs
    beta_contrasts = B @ C  # (n_proteins x m)

    # SEs
    se_contrasts = []
    for j in range(C.shape[1]):
        vj = C[:, j].T @ XtX_inv @ C[:, j]  # scalar
        se_j = np.sqrt(sigma2 * vj)         # vector (n_proteins,)
        se_contrasts.append(se_j)

    se_matrix = np.stack(se_contrasts, axis=1)  # (n_proteins x m)
    contrast_variances = np.array([             # (m, )
        C[:, j].T @ XtX_inv @ C[:, j]
        for j in range(C.shape[1])
    ])

    # Wrap into DataFrames
    protein_index = getattr(fit_results.get("protein_index", None), "copy", lambda: None)() or np.arange(B.shape[0])
    contrast_names = contrast_names or [f"contrast_{i}" for i in range(C.shape[1])]

    return beta_contrasts, se_matrix, contrast_variances


