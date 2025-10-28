# TODO adapt limma pipeline to (re)use this file instead of doing things there, cleaner.
# And check whether we should put missingness back here or keep in it clustering (no sense? idk)

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import fdrcorrection

from proteoflux.utils.utils import log_time


class StatisticalTester:
    """
    Perform statistical testing for differential expression contrasts.

    Computes t-statistics, p-values, and adjusted q-values (FDR) for each contrast,
    as well as feature-wise missingness summaries.
    """

    def __init__(
        self,
        log2fc: np.ndarray,
        se: np.ndarray,
        df_residual: Union[np.ndarray, float],
        contrast_names: List[str],
        protein_ids: List[str],
        df_prior: Optional[float] = None,
    ) -> None:
        """
        Initialize the StatisticalTester.

        Args:
            log2fc: Array of log2 fold-changes (n_features × n_contrasts).
            se: Array of standard errors, same shape as log2fc.
            df_residual: Residual degrees of freedom, scalar or array of length n_features.
            contrast_names: Names of each contrast.
            protein_ids: List of feature/ gene IDs of length n_features.
            df_prior: Optional prior degrees of freedom (e.g., from eBayes).
        """
        self.log2fc: np.ndarray = log2fc
        self.se: np.ndarray = se
        self.df_residual: np.ndarray = (
            np.array(df_residual) if not isinstance(df_residual, np.ndarray) else df_residual
        )
        self.df_prior: Optional[float] = df_prior
        self.contrast_names: List[str] = contrast_names
        self.protein_ids: List[str] = protein_ids

    @log_time("Compute statistics")
    def compute(self) -> Dict[str, np.ndarray]:
        """
        Compute t-statistics, p-values, and FDR-adjusted q-values.

        Returns:
            A dict containing:
              - 'log2fc': log2 fold-changes array
              - 't': t-statistics array
              - 'p': p-values array
              - 'q': FDR-adjusted q-values array
        """
        df_raw = self.df_residual
        log2fc = self.log2fc
        se = self.se

        # Ensure df_raw has shape (n_features, 1)
        if df_raw.ndim == 1:
            df_raw = df_raw[:, None]

        # Compute t-statistics with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = log2fc / se

        # Mask invalid or zero SE entries
        bad_se = ~np.isfinite(se) | (se == 0)
        t_stat[bad_se] = np.nan

        # Mask zero degrees of freedom
        zero_df = df_raw[:, 0] == 0
        t_stat[zero_df, :] = np.nan

        # Compute two-sided p-values
        with np.errstate(invalid='ignore'):
            p_val = 2 * t_dist.sf(np.abs(t_stat), df=df_raw)

        # Adjust for multiple testing (FDR) per contrast
        q_val = np.stack(
            [fdrcorrection(p_val[:, j])[1] for j in range(p_val.shape[1])],
            axis=1
        )

        return {"log2fc": log2fc, "t": t_stat, "p": p_val, "q": q_val}

    @staticmethod
    def compute_missingness(
        intensity_matrix: np.ndarray,
        conditions: List[str],
        feature_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compute per-feature missingness ratio for each condition.

        Args:
            intensity_matrix: Numeric array (n_features × n_samples).
            conditions: List of condition labels per sample.
            feature_ids: List of feature IDs for index.

        Returns:
            DataFrame of missingness ratios, shape (n_features, n_conditions).
        """
        condition_array = np.array(conditions)
        unique_conditions = np.unique(condition_array)

        counts: Dict[str, np.ndarray] = {}
        for cond in unique_conditions:
            mask = condition_array == cond
            sub = intensity_matrix[:, mask]
            # Proportion of missing per feature
            counts[cond] = np.isnan(sub).sum(axis=1)

        return pd.DataFrame(counts, index=feature_ids)

    def export_to_anndata(
        self,
        adata: AnnData,
        stats_dict: Dict[str, np.ndarray],
        missingness_df: pd.DataFrame
    ) -> AnnData:
        """
        Export statistics and missingness to AnnData.

        Stats go to .varm, missingness to .uns['missingness'].

        Args:
            adata: AnnData object to modify.
            stats_dict: Dict from compute(), with keys 'log2fc', 't', 'p', 'q'.
            missingness_df: DataFrame from compute_missingness().

        Returns:
            The modified AnnData.
        """
        for key, arr in stats_dict.items():
            adata.varm[key] = arr

        adata.uns["contrast_names"] = self.contrast_names
        adata.uns["missingness"] = missingness_df.to_dict(orient="index")
        return adata
