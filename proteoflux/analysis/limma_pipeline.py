import warnings
import anndata as ad
import pandas as pd
import numpy as np
import patsy
import inmoose.limma as imo
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from proteoflux.utils.utils import log_time
from proteoflux.stats import StatisticalTester
from proteoflux.analysis.clustering import run_clustering, run_clustering_missingness

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in divide",
    category=RuntimeWarning,
    module=r"inmoose\.limma\.decidetests"
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"inmoose\.limma\.decidetests"
)

@log_time("pyLimma pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:

    # 2) Read design column & one-hot encode
    #group_col = config["design"]["group_column"]
    obs       = adata.obs.copy()
    levels    = sorted(obs["CONDITION"].unique())
    for lvl in levels:
        obs[lvl] = (obs["CONDITION"] == lvl).astype(int)

    # 3) Patsy design (one column per level, no intercept)
    formula   = "0 + " + " + ".join(levels)
    design_dm = patsy.dmatrix(formula, obs)

    # 4) Expression matrix: genes × samples
    df_X = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names
    ).T

    # 5) Fit linear model & contrasts
    fit_imo = imo.lmFit(df_X, design=design_dm)

    # auto-generate all pairwise contrasts
    contrast_defs = [f"{a} - {b}" for a, b in combinations(levels, 2)]
    contrast_df   = imo.makeContrasts(contrast_defs, levels=design_dm)
    # normalize to “_vs_” names
    contrast_df.columns = [
        c.replace(" - ", "_vs_") for c in contrast_df.columns
    ]

    fit_imo = imo.contrasts_fit(fit_imo, contrasts=contrast_df)

    # === RAW (pre-eBayes) statistics ===
    coefs  = fit_imo.coefficients.values          # (n_genes × n_contrasts)
    stdu   = fit_imo.stdev_unscaled.values        # same shape
    sigma  = fit_imo.sigma.to_numpy()             # (n_genes,)
    df_res = fit_imo.df_residual                  # (n_genes,)

    se_raw = stdu * sigma[:, np.newaxis]
    t_raw  = coefs / se_raw
    p_raw  = 2 * t_dist.sf(np.abs(t_raw), df=df_res[:, None])
    q_raw  = np.vstack([
        multipletests(p_raw[:, j], method="fdr_bh")[1]
        for j in range(p_raw.shape[1])
    ]).T

    # === MODERATED (post-eBayes) statistics ===
    fit_imo    = imo.eBayes(fit_imo)
    s2post     = fit_imo.s2_post.to_numpy()       # (n_genes,)
    se_ebayes  = stdu * np.sqrt(s2post[:, np.newaxis])
    t_ebayes   = fit_imo.t.values
    p_ebayes   = fit_imo.p_value.values
    q_ebayes   = np.vstack([
        multipletests(p_ebayes[:, j], method="fdr_bh")[1]
        for j in range(p_ebayes.shape[1])
    ]).T

    # === Assemble into AnnData ===
    out = adata.copy()

    out.varm["log2fc"]  = coefs
    out.varm["se_raw"]  = se_raw
    out.varm["t_raw"]   = t_raw
    #out.varm["p_raw"]   = p_raw #future
    out.varm["p"]   = p_raw
    #out.varm["q_raw"]   = q_raw #same
    out.varm["q"]   = q_raw
    # moderated statistics
    out.varm["se_ebayes"]  = se_ebayes
    out.varm["t_ebayes"]   = t_ebayes
    out.varm["p_ebayes"]   = p_ebayes
    out.varm["q_ebayes"]   = q_ebayes

    # metadata
    out.uns["contrast_names"] = list(contrast_df.columns)

    # === Missingness exactly as before ===
    missing_df = StatisticalTester.compute_missingness(
        intensity_matrix = adata.layers["qvalue"].T,
        conditions       = adata.obs['CONDITION'].tolist(),
        feature_ids      = adata.var_names.tolist(),
    )
    out.uns["missingness"] = missing_df

    # clustering
    out = run_clustering(out, n_pcs=out.X.shape[0]-1)
    out = run_clustering_missingness(out)


    return out
