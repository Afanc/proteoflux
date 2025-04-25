import numpy as np
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
from proteoflux.utils.utils import logger, log_time
from proteoflux.analysis.ebayes_prior import fit_fdist, fit_fdist_robust, squeeze_var_input_filter, tmixture_vector

class EbayesModerator:
    def __init__(self, sigma2, df_residual, method="moments"):
        """
        Parameters:
        - sigma2: (n_proteins,) vector of residual variances
        - df_residual: scalar or array of degrees of freedom (per protein)
        """
        self.sigma2 = sigma2
        self.df_residual = df_residual
        self.d0 = None
        self.s0 = None
        self.method = method

    def fit(self):
        if self.method == "moments":
            return self._fit_moments()
        elif self.method == "limma":
            return self._fit_limma_like()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_limma_like(self):
        robust=False #TODO option

        filtered_s2, filtered_df = squeeze_var_input_filter(
            self.sigma2,
            self.df_residual)

        z = np.log(filtered_s2)

        if robust:
            raise NotImplementedError
            #s0, d0 = fit_fdist_robust(filtered_s2, filtered_df)
        else:
            s0, d0 = fit_fdist(filtered_s2, filtered_df)

        self.s0 = s0
        self.d0 = d0
        return d0, s0

    def _fit_moments(self):
        """
        Estimate prior variance s0² and prior df d0 using method of moments.
        Based on limma's `fitFDist()` approach.
        """
        s2 = self.sigma2
        d = self.df_residual

        # log variance
        lns2 = np.log(np.clip(self.sigma2, 1e-8, None))  # foolproof
        mean_ln = np.mean(lns2)
        var_ln = np.var(lns2, ddof=1)

        var_ln = max(var_ln, 1e-6) # foolproof again

        # Method-of-moments estimator for d0
        d0 = max(2 * ((1 / var_ln) - 1), 1.0)
        s0 = np.exp(mean_ln - np.log(d0 / max(d0 - 2, 1e-6)))

        self.d0 = d0
        self.s0 = s0
        return d0, s0

    def moderate(self):
        """
        Returns:
        - moderated variances
        - total degrees of freedom (dg + d0)
        """
        d = self.df_residual
        s2 = self.sigma2
        s02 = self.s0**2
        d0 = self.d0

        # shrink variances
        if np.isnan(self.s0):
            s2_moderated = self.sigma2
            df_total = self.df_residual
        else:
            s2_moderated = (self.d0 * self.s0 + self.df_residual * self.sigma2) / (self.d0 + self.df_residual)
            df_total = self.d0 + self.df_residual

        df_total = d0 + d
        self.df_total = df_total

        return s2_moderated, df_total

    @log_time("EBayes Computation")
    def apply_to_contrasts(self, log2fc, contrast_variances):
        """
        Recalculate t, p, q using moderated variances

        Parameters:
        - log2fc: (n_proteins x n_contrasts)
        - contrast_variances: (n_contrasts,)  — v_j = c^T (X'X)^-1 c for each contrast

        Returns:
        - dict: t, p, q (each of shape n_proteins x n_contrasts)
        """
        # Vectorized SEs
        s2_moderated, df_total = self.moderate()
        s2_moderated = np.clip(s2_moderated, 1e-8, None) #foolproofing
        se = np.sqrt(np.outer(s2_moderated, contrast_variances))  # (n_proteins x n_contrasts)

        t_stat = log2fc / se

        p_val = 2 * t_dist.sf(np.abs(t_stat), df=self.df_total)

        q_val = np.zeros_like(p_val)
        for j in range(p_val.shape[1]):
            q_val[:, j] = multipletests(p_val[:, j], method="fdr_bh")[1]

        return {
            "t_ebayes": t_stat,
            "p_ebayes": p_val,
            "q_ebayes": q_val,
        }
