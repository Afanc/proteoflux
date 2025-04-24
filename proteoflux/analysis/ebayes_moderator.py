import numpy as np
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
from proteoflux.utils.utils import logger, log_time

class EbayesModerator:
    def __init__(self, sigma2, df_residual):
        """
        Parameters:
        - sigma2: (n_proteins,) vector of residual variances
        - df_residual: scalar or array of degrees of freedom (per protein)
        """
        self.sigma2 = sigma2
        self.df_residual = df_residual
        self.d0 = None
        self.s0 = None

    def fit(self):
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
        s2_moderated = (d0 * s02 + d * s2) / (d0 + d)
        df_total = d0 + d

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
        p_val = 2 * t_dist.sf(np.abs(t_stat), df=df_total)

        q_val = np.zeros_like(p_val)
        for j in range(p_val.shape[1]):
            q_val[:, j] = multipletests(p_val[:, j], method="fdr_bh")[1]

        return {
            "t_ebayes": t_stat,
            "p_ebayes": p_val,
            "q_ebayes": q_val,
        }

if __name__ == "__main__":
    def test_ebayes_simple_case():
        # Simulate a toy case
        sigma2 = np.array([0.5, 1.0, 1.5, 2.0])
        df = 6  # constant for simplicity

        logfc = np.array([
            [1.0, 0.5],
            [2.0, 1.5],
            [-1.0, 0.2],
            [-0.5, 2.0]
        ])  # shape: (4 proteins × 2 contrasts)

        contrast_var = np.array([0.1, 0.2])  # two contrast variances

        moderator = EbayesModerator(sigma2=sigma2, df_residual=df)
        d0, s0 = moderator.fit()

        assert d0 > 0 and s0 > 0, "Prior parameters should be positive"

        result = moderator.apply_to_contrasts(logfc, contrast_var)

        t = result["t_ebayes"]
        p = result["p_ebayes"]
        q = result["q_ebayes"]

        assert t.shape == logfc.shape
        assert p.shape == logfc.shape
        assert q.shape == logfc.shape
        assert np.all(p >= 0) and np.all(p <= 1), "p-values must be [0, 1]"

        print("Test passed: shapes and value ranges are valid.")

    def test_ebayes_moderation_effect():
        """
        Simulate proteins with same log2FC but different variances.
        eBayes should increase t-stats and lower p/q for high-variance ones.
        """
        # Simulated variances
        sigma2 = np.array([0.1, 1.0, 0.1, 2.0])  # protein 2 and 4 are high-variance
        df = 6  # fixed residual df

        # All proteins same effect size (log2FC)
        logfc = np.array([
            [1.0],
            [1.0],
            [1.0],
            [1.0]
        ])

        # One contrast, so contrast variance is scalar
        contrast_var = np.array([0.1])  # small v_j

        moderator = EbayesModerator(sigma2=sigma2, df_residual=df)
        d0, s0 = moderator.fit()

        assert d0 > 0 and s0 > 0

        result = moderator.apply_to_contrasts(logfc, contrast_var)

        t_raw = logfc.flatten() / np.sqrt(sigma2 * contrast_var[0])
        t_moderated = result["t_ebayes"].flatten()

        print("Raw t-stats:      ", np.round(t_raw, 3))
        print("Moderated t-stats:", np.round(t_moderated, 3))

        # High-variance proteins should be *boosted*
        assert t_moderated[1] > t_raw[1], "eBayes should increase t-stat for protein 2"
        assert t_moderated[3] > t_raw[3], "eBayes should increase t-stat for protein 4"

        # And low-variance proteins should be relatively unchanged or slightly shrunk
        assert np.abs(t_moderated[0] - t_raw[0]) / t_raw[0] < 0.1
        assert np.abs(t_moderated[2] - t_raw[2]) / t_raw[2] < 0.1

        print("eBayes moderation test passed ✅")

    test_ebayes_simple_case()
    test_ebayes_moderation_effect()
