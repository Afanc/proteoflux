from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import anndata as ad


@dataclass(frozen=True)
class MissingnessResult:
    df: pd.DataFrame
    source: str
    rule: str
    feature_indices: np.ndarray  # always full range (for provenance)


def _missingness_counts(intensity_matrix_GxN: np.ndarray, conditions: list[str]) -> dict[str, np.ndarray]:
    cond_arr = np.asarray(conditions, dtype=str)
    uniq = np.unique(cond_arr)
    out: dict[str, np.ndarray] = {}
    for cond in uniq:
        mask = cond_arr == cond
        sub = intensity_matrix_GxN[:, mask]
        out[cond] = np.isnan(sub).sum(axis=1)
    return out


def compute_missingness(
    adata: ad.AnnData,
    *,
    max_features: None = None,
    random_seed: int = 0,
) -> MissingnessResult:
    """
    Compute per-feature missingness counts per condition, using a deterministic random
    subset of features when max_features < n_vars.

    Source selection matches previous behavior:
      - layers['qvalue'] if present
      - else layers['raw'] if present
      - else adata.X
    """
    if "qvalue" in adata.layers:
        mat = np.asarray(adata.layers["qvalue"]).T  # (G x N)
        source = "qvalue"
    elif "raw" in adata.layers:
        mat = np.asarray(adata.layers["raw"]).T
        source = "raw"
    else:
        mat = np.asarray(adata.X).T
        source = "X"

    feat_idx = np.arange(adata.n_vars, dtype=int)
    mat = mat[feat_idx, :]

    counts = _missingness_counts(mat, adata.obs["CONDITION"].astype(str).tolist())
    df = pd.DataFrame(counts, index=adata.var_names.tolist())
    return MissingnessResult(df=df, source=source, rule="nan-is-missing", feature_indices=feat_idx)
