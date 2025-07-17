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

@log_time("pyLimma pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    # 1) Optional: fall back to R-limma
    if config.get("analysis", {}).get("use_r_limma", False):
        from proteoflux.analysis.r_limma_pipeline import run_r_limma_pipeline
        return run_r_limma_pipeline(adata, config)

    # 2) Read design column & one-hot encode
    group_col = config["design"]["group_column"]
    obs       = adata.obs.copy()
    levels    = sorted(obs[group_col].unique())
    for lvl in levels:
        obs[lvl] = (obs[group_col] == lvl).astype(int)

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
        conditions       = adata.obs[group_col].tolist(),
        feature_ids      = adata.var_names.tolist(),
    )
    out.uns["missingness"] = missing_df

    return out

#import anndata as ad
#import pandas as pd
#import numpy as np
#import patsy
#import inmoose.limma as imo
#from scipy.stats import t as t_dist
#from statsmodels.stats.multitest import multipletests
#from itertools import combinations
#from proteoflux.utils.utils import log_time
#from proteoflux.stats import StatisticalTester
#
#@log_time("pyLimma pipeline")
#def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
#    # 1) Optional: fall back to R-limma
#    use_r = config.get("analysis", {}).get("use_r_limma", False)
#    if use_r:
#        from proteoflux.analysis.r_limma_pipeline import run_r_limma_pipeline
#        return run_r_limma_pipeline(adata, config)
#
#    # 2) Read config
#    group_col    = config["design"]["group_column"]
#    contrast_req = config["analysis"].get("contrast_name", None)
#    # (we don’t support voom here)
#
#    # 3) One-hot encode obs, build Patsy design
#    obs = adata.obs.copy()
#    levels = sorted(obs[group_col].unique())
#    for lvl in levels:
#        obs[lvl] = (obs[group_col] == lvl).astype(int)
#
#    formula   = "0 + " + " + ".join(levels)
#    design_dm = patsy.dmatrix(formula, obs)  # patsy.DesignMatrix :contentReference[oaicite:0]{index=0}
#
#    # 4) Expression matrix (genes × samples)
#    df_X = pd.DataFrame(
#        adata.X, index=adata.obs_names, columns=adata.var_names
#    ).T
#
#    # 5) Fit, build all pairwise contrasts, apply & moderate
#    fit_imo = imo.lmFit(df_X, design=design_dm)  # :contentReference[oaicite:1]{index=1}
#
#    # auto pairwise contrasts
#    contrast_defs = [f"{a} - {b}" for a, b in combinations(levels, 2)]
#    contrast_df   = imo.makeContrasts(contrast_defs, levels=design_dm)
#    new_cols = [c.replace(" - ", "_vs_") for c in contrast_df.columns]
#    contrast_df.columns = new_cols
#
#    fit_imo = imo.contrasts_fit(fit_imo, contrasts=contrast_df)  # :contentReference[oaicite:2]{index=2}
#    fit_imo = imo.eBayes(fit_imo)                                 # :contentReference[oaicite:3]{index=3}
#
#    # 6) Now use your StatisticalTester to compute stats + missingness in one go
#    #    It will fill .varm['log2fc','t','p','q'] and .uns['missingness','contrast_names']
#    stats = StatisticalTester(
#        log2fc       = fit_imo.coefficients.values,
#        se           = fit_imo.stdev_unscaled.values * fit_imo.sigma.values[:, None],
#        df_residual  = fit_imo.df_residual,
#        contrast_names = list(contrast_df.columns),
#        protein_ids  = adata.var_names.tolist(),
#        df_prior     = float(np.atleast_1d(fit_imo.df_prior)[0]),
#    )
#    stats_dict = stats.compute()
#    # missingness: use the same layer you used before (e.g. 'qvalue')
#    missing_df = StatisticalTester.compute_missingness(
#        intensity_matrix = adata.layers["qvalue"].T,
#        conditions       = adata.obs[group_col].tolist(),
#        feature_ids      = adata.var_names.tolist(),
#    )
#    out = stats.export_to_anndata(adata.copy(), stats_dict, missing_df)
#    print(out)
#    return out
#
##    # 6) Bake into a new AnnData
##    out = adata.copy()
##
##    coefs  = fit_imo.coefficients.values            # genes×contrasts
##    stdu   = fit_imo.stdev_unscaled.values          # genes×contrasts
##    sigma  = fit_imo.sigma.to_numpy()               # (genes,)
##    df_res = fit_imo.df_residual                    # (genes,)
##    s2post = fit_imo.s2_post.to_numpy()             # (genes,)
##
##    # compute raw t‐statistics
##    t_raw = coefs / (stdu * sigma[:, np.newaxis])
##
##    # compute two‐sided p‐values
##    # note broadcasting df_resid over contrasts
##    p_raw = 2 * t_dist.sf(np.abs(t_raw), df=df_res[:, np.newaxis])
##    q_raw = np.vstack([
##        multipletests(p, method="fdr_bh")[1] for p in p_raw.T
##    ]).T                                          # genes×contrasts
##
##
##    # Raw & moderated SE (convert Series → ndarray first)
##    se_raw    = stdu * sigma[:, np.newaxis]
##    se_ebayes = stdu * np.sqrt(s2post[:, np.newaxis])
##
##    # Statistics
##    t_ebayes = fit_imo.t.values                   # genes×contrasts
##    p_ebayes = fit_imo.p_value.values             # genes×contrasts
##    q_ebayes = np.vstack([
##        multipletests(p, method="fdr_bh")[1] for p in p_ebayes.T
##    ]).T                                          # genes×contrasts
##    B         = fit_imo.lods.values               # genes×contrasts
##
##    # Fill layers and varm
##    out.varm["log2fc"]     = coefs
##    out.varm["stdev"]      = stdu
##    out.varm["se_raw"]     = se_raw
##    out.varm["se_ebayes"]  = se_ebayes
##    out.varm["t_raw"] = t_raw
##    out.varm["p"] = p_raw #adapt key to raw
##    out.varm["q"] = q_raw #same
##    out.varm["t_ebayes"]   = t_ebayes
##    out.varm["p_ebayes"]   = p_ebayes
##    out.varm["q_ebayes"]   = q_ebayes
##    out.varm["B"]          = B
##
##    # Global priors & metadata
##    #out.uns["Amean"]           = fit_imo.Amean
##    #out.uns["s2_prior"]        = float(np.atleast_1d(fit_imo.s2_prior)[0])
##    #out.uns["s2_prior_full"]   = fit_imo.s2_prior
##    #out.uns["df_prior"]        = float(np.atleast_1d(fit_imo.df_prior)[0])
##    #out.uns["var_prior"]       = fit_imo.var_prior
##    #out.uns["df_total"]        = getattr(fit_imo, "df_total", None)
##    out.uns["contrast_names"]  = list(contrast_df.columns)
##    # --- exactly as in the old pylimma pipeline: fill missingness per group
##    protein_ids = adata.var_names
##    # proteoflux uses 'qvalue' layer for detection; transpose to samples×genes
##    intensity   = adata.layers["qvalue"].T
##    conditions  = adata.obs[group_col].values
##    out.uns["missingness"] = StatisticalTester.compute_missingness(
##        intensity_matrix=intensity,
##        conditions=conditions,
##        protein_ids=protein_ids
##    )
##
##    return out
##
###import anndata as ad
###from pylimma import fit
###from proteoflux.utils.utils import log_time
###
###@log_time("pyLimma pipeline")
###def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
###    use_r = config.get("analysis", {}).get("use_r_limma", False)
###    if use_r:
###        from proteoflux.analysis.r_limma_pipeline import run_r_limma_pipeline
###        return run_r_limma_pipeline(adata, config)
###
###    # Extract design-related config values
###    group = config["design"]["group_column"]
###    remove_batch = config["design"].get("remove_batch", None)
###    contrast_name = config["analysis"].get("contrast_name", None)
###    use_voom = config["analysis"].get("use_voom", False)
###
###    # Run full pyLimma pipeline
###    adata_out, _, _ = fit(
###        adata=adata,
###        group=group,
###        formula=None,
###        design_matrix=None,
###        contrast_name=contrast_name,
###        remove_batch=remove_batch,
###        use_voom=use_voom,
###    )
###
###    return adata_out
