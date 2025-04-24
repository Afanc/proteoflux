import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
from typing import List
from proteoflux.utils.utils import logger, log_time

class StatisticalTester:
    def __init__(self, log2fc, se, df_residual, contrast_names, protein_ids):
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
        self.df = df_residual
        self.contrast_names = contrast_names
        self.protein_ids = protein_ids

    def compute(self):

        # Perseus style variance flooring
        pos_se = self.se[self.se > 0]
        if len(pos_se) == 0:
            s0 = 1e-6  # fallback if somehow everything is zero
        else:
            s0 = np.percentile(pos_se, 10)

        # replace raw SE with sqrt(SE^2 + s0^2)
        self.se = np.sqrt(self.se**2 + s0**2)

        # compute t stat
        t_stat = self.log2fc / self.se
        p_val = 2 * t_dist.sf(np.abs(t_stat), df=self.df)

        # Adjusted p-values (FDR)
        q_val = np.zeros_like(p_val)
        for j in range(p_val.shape[1]):
            q_val[:, j] = multipletests(p_val[:, j], method="fdr_bh")[1]


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
            #adata.varm[key] = pd.DataFrame(arr, index=self.protein_ids, columns=self.contrast_names)
            adata.varm[key] = arr

        adata.uns["contrast_names"] = self.contrast_names
        adata.uns["missingness"] = {
            k: v.to_dict(orient="index") for k, v in missingness_dict.items()
        }

        return adata

