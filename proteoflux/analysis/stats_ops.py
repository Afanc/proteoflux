from __future__ import annotations

import numpy as np
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests


def raw_stats_from_fit(
    *,
    coefs: np.ndarray,
    stdu: np.ndarray,
    sigma: np.ndarray,
    df_res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mechanical shared primitive:
      se = stdu * sigma[:, None]
      t  = coefs / se
      p  = 2 * t.sf(|t|, df=df_res[:, None])
    """
    se = stdu * sigma[:, None]
    t = coefs / se
    p = 2 * t_dist.sf(np.abs(t), df=df_res[:, None])
    return se, t, p


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values, applied per contrast/column (mechanical extraction)."""
    if p.ndim != 2:
        raise ValueError(f"Expected 2D p-value array (n_features x n_contrasts), got shape {p.shape}")
    q = np.vstack([multipletests(p[:, j], method="fdr_bh")[1] for j in range(p.shape[1])]).T
    return q

