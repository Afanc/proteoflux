import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests, fdrcorrection
from typing import List
from proteoflux.utils.utils import logger, log_time

class StatisticalTester:
    def __init__(self, log2fc, se, df_residual, contrast_names, protein_ids, df_prior = None):
        """
        Parameters:
        - log2fc: (n_proteins x n_contrasts) NumPy array
        - se: same shape
        - df_residual: scalar (or vector in future)
        - contrast_names: list of contrast names
        - protein_ids: list/array of protein names
        """
        self.log2fc = log2fc
        self.se = se
        self.df_residual = df_residual
        self.df_prior = df_prior
        self.contrast_names = contrast_names
        self.protein_ids = protein_ids

    def compute(self):

        df_raw = self.df_residual

        # compute t stat
        t_stat = self.log2fc / self.se
        p_val = 2 * t_dist.sf(np.abs(t_stat), df=df_raw)

        # BH on raw p's
        q_val = np.stack([
            fdrcorrection(p_val[:, j])[1]
            for j in range(p_val.shape[1])
        ], axis=1)
        t_stat = self.log2fc / self.se

        return {
            "log2fc": self.log2fc,
            "t": t_stat,
            "p": p_val,
            "q": q_val,
        }

    def compute_missingness(self, qvalue_matrix: np.ndarray, conditions: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame of shape (proteins x conditions) with ratio of missing q-values.
        """
        n_proteins, n_samples = qvalue_matrix.shape
        df = pd.DataFrame(qvalue_matrix, columns=conditions)

        ratios = {}
        for condition in np.unique(conditions):
            mask = (np.array(conditions) == condition)
            ratio = np.isnan(qvalue_matrix[:, mask]).sum(axis=1) / mask.sum()
            ratios[condition] = ratio

        return pd.DataFrame(ratios, index=self.protein_ids)

    def export_to_anndata(self, adata, stats_dict, missingness_dict):
        """
        Save outputs to adata.varm and adata.uns

        stats_dict keys: 'log2fc', 't', 'p', 'q'
        missingness_dict: dict from compute_missingness()
        """
        for key, arr in stats_dict.items():
            adata.varm[key] = arr

        adata.uns["contrast_names"] = self.contrast_names
        adata.uns["missingness"] = {
            k: v.to_dict(orient="index") for k, v in missingness_dict.items()
        }

        return adata

