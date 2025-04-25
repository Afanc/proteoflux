import numpy as np

def tmixture_vector(t_stat: np.ndarray, stdev_unscaled: np.ndarray, df: float, proportion: float = 0.01) -> np.ndarray:
    """
    Empirical Bayes estimation of prior variance on logFCs (var.prior),
    matching limma's tmixture.vector()

    Parameters:
    - t_stat: moderated t-values (n_genes x n_contrasts)
    - stdev_unscaled: sqrt(cᵀ(X'X)⁻¹c) per contrast (n_genes x n_contrasts)
    - df: scalar degrees of freedom
    - proportion: assumed proportion of DE genes

    Returns:
    - var_prior: (n_contrasts,) estimated prior variance per contrast
    """
    t_stat = np.abs(t_stat)
    n_genes, n_contrasts = t_stat.shape
    var_prior = np.zeros(n_contrasts)

    for j in range(n_contrasts):
        t = t_stat[:, j]
        su = stdev_unscaled[:, j]
        v1 = su**2

        # Select top t-statistics
        r = max(1, int(proportion * n_genes))
        top_idx = np.argsort(t)[-r:]
        t_top = t[top_idx]
        v1_top = v1[top_idx]

        # Expected under null
        q = (np.arange(1, r + 1) - 0.5) / n_genes
        t0 = np.abs(norm.ppf((1 + q) / 2))
        expected = t0**2

        diff = (t_top**2 - expected)
        v0 = np.maximum(0, v1_top * (diff / expected))

        var_prior[j] = np.mean(v0)

    return var_prior


