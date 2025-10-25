import anndata as ad
import pandas as pd
import numpy as np
import patsy
from typing import Tuple
from itertools import combinations
import inmoose.limma as imo
from scipy.stats import t as t_dist
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from proteoflux.utils.utils import log_time, log_warning
from proteoflux.analysis.statisticaltester import StatisticalTester
from proteoflux.analysis.clustering import run_clustering, run_clustering_missingness

@log_time("Analysis pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    # If a covariate layer exists, run the covariate model path (simple OLS + BH)

    injected_runs = (config or {}).get("dataset", {}).get("inject_runs") or {}
    has_covariate_cfg = any(bool(run.get("is_covariate")) for _, run in injected_runs.items())

    # Pilot study mode: allow experiments where at least one condition has only 1 replicate.
    # We still fit OLS, compute logFC + (unmoderated) p/q, but SKIP eBayes.

    # 2) Read design column & one-hot encode
    obs    = adata.obs.copy()

    repl_counts = obs["CONDITION"].value_counts()
    pilot_mode = (repl_counts.min() <= 1)
    if pilot_mode:
        log_warning(
            "Pilot study mode: at least one condition has only 1 replicate, no statistical analysis done. "
        )

    if has_covariate_cfg:
        return run_limma_pipeline_covariate(adata, config, pilot_mode)

    levels = sorted(obs["CONDITION"].unique())
    if len(levels) < 2:
        raise ValueError(f"Need ≥2 conditions for contrasts; found {levels}")
    for lvl in levels:
        obs[lvl] = (obs["CONDITION"] == lvl).astype(int)

    # 3) Patsy design (no intercept) — keep as Patsy DesignMatrix
    formula   = "0 + " + " + ".join(levels)
    design_dm = patsy.dmatrix(formula, obs)

    # 4) Expression: genes × samples
    df_X = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T
    # debug
    #df_X.loc["P0AB71", "195_KaForAll_DIA_B22"] = 15.618
    #print(df_X.shape)
    #

    # 5) Fit
    fit_imo = imo.lmFit(df_X, design=design_dm)
    resid_var = np.asarray(fit_imo.sigma, dtype=np.float32) ** 2
    adata.uns["residual_variance"] = resid_var

    # 6) Contrasts (with optional only_against)
    base = (config or {}).get("analysis", {}).get("only_against", None)

    if base is not None:
        if base not in levels:
            raise ValueError(f"only_against='{base}' not found in Condition levels {levels}")
        contrast_defs = [f"{c} - {base}" for c in levels if c != base]
    else:
        contrast_defs = [f"{a} - {b}" for a, b in combinations(levels, 2)]

    # IMPORTANT: pass the Patsy DesignMatrix as 'levels' (exactly like before)
    contrast_df = imo.makeContrasts(contrast_defs, levels=design_dm)
    # (Optional) pretty names for downstream labeling
    contrast_df.columns = [c.replace(" - ", "_vs_") for c in contrast_df.columns]

    # Now coefficients @ contrasts lines up (both use the same unnamed basis)
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
    if not pilot_mode:
        with np.errstate(divide='ignore', invalid='ignore'):
            fit_imo    = imo.eBayes(fit_imo)

        s2post     = fit_imo.s2_post.to_numpy()       # (n_genes,)
        se_ebayes  = stdu * np.sqrt(s2post[:, np.newaxis])
        t_ebayes   = fit_imo.t.values
        p_ebayes   = fit_imo.p_value.values
        q_ebayes   = np.vstack([
            multipletests(p_ebayes[:, j], method="fdr_bh")[1]
            for j in range(p_ebayes.shape[1])
        ]).T

    else:
        # Placeholders (omit from .varm below)
        se_ebayes = t_ebayes = p_ebayes = q_ebayes = None

    ###
    #print(adata.var_names.get_loc("P0AB71"))
    #ix = adata.var_names.get_loc("P0AB71")
    #print(coefs[ix])
    #print(se_raw[ix])
    #print(t_raw[ix])
    #print(p_raw[ix])
    #print(q_ebayes[ix])
    #print(adata.X[:,ix])
    ##

    # === Assemble into AnnData ===
    out = adata.copy()

    out.varm["log2fc"]  = coefs

    if not pilot_mode:
        out.varm["se_raw"]  = se_raw
        out.varm["t_raw"]   = t_raw
        out.varm["p_raw"]   = p_raw
        out.varm["q_raw"]   = q_raw

        # moderated statistics
        out.varm["se_ebayes"]  = se_ebayes
        out.varm["t_ebayes"]   = t_ebayes
        out.varm["p_ebayes"]   = p_ebayes
        out.varm["q_ebayes"]   = q_ebayes

    # metadata
    out.uns["contrast_names"] = list(contrast_df.columns)
    out.uns["pilot_study_mode"] = bool(pilot_mode)

    # === Missingness exactly as before ===
    if "qvalue" in adata.layers:
        miss_mat = adata.layers["qvalue"].T
        miss_source = "qvalue"
    elif "raw" in adata.layers:
        # Use pre-imputation raw intensities; NaN means missing
        miss_mat = adata.layers["raw"].T
        miss_source = "raw"
    else:
        # Last resort: use X (may be imputed). Still better than crashing.
        miss_mat = adata.X.T
        miss_source = "X"

    missing_df = StatisticalTester.compute_missingness(
        intensity_matrix = miss_mat,
        conditions       = adata.obs['CONDITION'].tolist(),
        feature_ids      = adata.var_names.tolist(),
    )
    out.uns["missingness"] = missing_df
    out.uns["missingness_source"] = miss_source
    out.uns["missingness_rule"] = "nan-is-missing"

    out.uns["has_covariate"] = False

    return out

@log_time("Clustering")
def clustering_pipeline(adata: ad.AnnData) -> ad.AnnData:
    adata = run_clustering(adata, n_pcs=adata.X.shape[0]-1)
    adata = run_clustering_missingness(adata)

    return adata

@log_time("Running Covariate branch")
def run_limma_pipeline_covariate(adata: ad.AnnData, config: dict, pilot_mode: bool) -> ad.AnnData:
    """
    Two-stage residualization (no interaction):
      Stage 1:  Y ~ 1 + C         (per-feature OLS with global row-centering of C)
      Stage 2:  R ~ 0 + Group     (limma on residuals, with eBayes; same contrasts as raw)
    The decomposition 'raw ≈ adjusted + covariate_part' is preserved.

    All outputs/keys match the current implementation.
    """

    # ----------------------------
    # Small helpers (pure functions)
    # ----------------------------
    def _get_levels_and_base(obs: pd.DataFrame, cfg: dict) -> Tuple[list, str]:
        levels = sorted(obs["CONDITION"].unique())
        if len(levels) < 2:
            raise ValueError(f"Need ≥2 conditions for contrasts; found {levels}")
        base = (cfg or {}).get("analysis", {}).get("only_against", None)
        if base is None:
            base = levels[0]
        if base not in levels:
            raise ValueError(f"only_against='{base}' not in Condition {levels}")
        return levels, base

    def _find_covariate_layer(adata: ad.AnnData) -> str:
        for k in ("covariate", "imputed_covariate", "cov", "ft"):
            if k in adata.layers:
                return k
        raise ValueError("Covariate layer not found in AnnData.layers (tried: covariate, imputed_covariate, cov, ft)")

    def _genes_by_samples_df(X_like: np.ndarray, obs_names, var_names) -> pd.DataFrame:
        """Return (G × N) as DataFrame indexed by features with sample columns."""
        return pd.DataFrame(X_like, index=obs_names, columns=var_names).T

    def _build_group_design_no_intercept(obs: pd.DataFrame, levels: list) -> patsy.DesignMatrix:
        tmp = obs.copy()
        for lvl in levels:
            tmp[lvl] = (tmp["CONDITION"] == lvl).astype(int)
        return patsy.dmatrix("0 + " + " + ".join(levels), tmp)

    def _make_contrasts(levels: list, design_dm, cfg: dict) -> pd.DataFrame:
        base = (cfg or {}).get("analysis", {}).get("only_against", None)
        if base is not None:
            if base not in levels:
                raise ValueError(f"only_against='{base}' not found in Condition levels {levels}")
            defs = [f"{c} - {base}" for c in levels if c != base]
        else:
            defs = [f"{a} - {b}" for a, b in combinations(levels, 2)]
        contr = imo.makeContrasts(defs, levels=design_dm)
        contr.columns = [c.replace(" - ", "_vs_") for c in contr.columns]
        return contr

    def _stage1_ols_y_on_1_plus_c(Y: np.ndarray, C: np.ndarray, ridge_lambda: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Closed-form OLS for y ~ 1 + c per feature with global row-centering of C already applied.
        Returns:
          beta1 (G,), beta0 (G,), residuals R (G×N), se_beta1 (G,), t_beta1 (G,), p_beta1 (G,), q_beta1 (G,)
        """
        G, N = Y.shape
        df1 = max(N - 2, 1)

        sY  = Y.sum(axis=1)                     # (G,)
        sYC = (Y * C).sum(axis=1)               # (G,)
        sCC = (C * C).sum(axis=1)               # (G,)

        # tiny ridge based on global scale of sCC to avoid exploding slopes when C is ~flat
        # baseline = median(sCC[sCC>0]) (fallback 1.0)
        with np.errstate(invalid="ignore"):
            _pos = sCC[sCC > 0]
        ridge_baseline = float(np.median(_pos)) if _pos.size else 1.0
        ridge = ridge_lambda * ridge_baseline
        sCC_tilde = sCC + ridge

        with np.errstate(divide="ignore", invalid="ignore"):
            beta1 = np.where(sCC_tilde > 0, sYC / sCC_tilde, 1.0)  # slope wrt covariate
        beta0 = sY / N                                  # intercept

        # --- NEW: smart floor for near-flat covariates (treat as no covariate effect) ---
        # threshold scales with typical covariate energy, but never below 1e-3
        tau = max(1e-2, ridge_baseline * 1e-6)
        # also require a minimum per-row SD to consider the covariate "non-flat"
        # (handles fully-imputed jitter where sCC is small but not tiny)
        sdC = np.sqrt(sCC / max(N - 1, 1))
        eps_sd = 0.10  # log2 scale; ~2x your jitter (0.05)
        flat = (sCC < tau) | (sdC < eps_sd)
        if np.any(flat):
            beta1[flat] = 0.0

        Yhat = beta0[:, None] + beta1[:, None] * C
        R    = Y - Yhat

        rss = np.sum(R * R, axis=1)
        s2  = rss / df1
        with np.errstate(divide="ignore", invalid="ignore"):
            se1 = np.sqrt(s2 / np.maximum(sCC_tilde, 1e-15))

        #t1 = np.where(se1 > 0, beta1 / se1, 0.0)
        #p1 = 2 * t_dist.sf(np.abs(t1), df=df1)
        t1 = np.where(se1 > 0, beta1 / se1, 0.0)
        p1 = 2 * t_dist.sf(np.abs(t1), df=df1)
        if np.any(flat):
            # for “flat” rows, explicitly null the test: t=0, p=1 (avoids spurious huge betas)
            t1[flat] = 0.0
            p1[flat] = 1.0
        q1 = multipletests(p1, method="fdr_bh")[1]

        return beta1, beta0, R, se1, t1, p1, q1

    def _fit_limma_with_contrasts(df_GxN: pd.DataFrame, design_dm, contrasts_df: pd.DataFrame):
        """lmFit + contrasts + eBayes; also returns raw (pre-moderation) stats."""
        fit = imo.lmFit(df_GxN, design=design_dm)
        fit = imo.contrasts_fit(fit, contrasts=contrasts_df)

        # raw stats
        coefs  = fit.coefficients.values
        stdu   = fit.stdev_unscaled.values
        sigma  = fit.sigma.to_numpy()
        df_res = fit.df_residual
        se_raw = stdu * sigma[:, None]
        t_raw  = coefs / se_raw
        p_raw  = 2 * t_dist.sf(np.abs(t_raw), df=df_res[:, None])
        q_raw  = np.vstack([multipletests(p_raw[:, j], method="fdr_bh")[1] for j in range(p_raw.shape[1])]).T

        # eBayes
        if not pilot_mode:
            with np.errstate(divide='ignore', invalid='ignore'):
                fit = imo.eBayes(fit)
            s2post     = fit.s2_post.to_numpy()
            se_ebayes  = stdu * np.sqrt(s2post[:, None])
            t_ebayes   = fit.t.values
            p_ebayes   = fit.p_value.values
            q_ebayes   = np.vstack([multipletests(p_ebayes[:, j], method="fdr_bh")[1] for j in range(p_ebayes.shape[1])]).T
        else:
            se_ebayes = t_ebayes = p_ebayes = q_ebayes = None

        return {
            "coefs": coefs,
            "se_raw": se_raw,
            "t_raw": t_raw,
            "p_raw": p_raw,
            "q_raw": q_raw,
            "se_ebayes": se_ebayes,
            "t_ebayes": t_ebayes,
            "p_ebayes": p_ebayes,
            "q_ebayes": q_ebayes,
            "df_res": df_res,
        }

    def _covariate_part_per_contrast(beta1: np.ndarray, C_centered_df: pd.DataFrame,
                                     obs: pd.DataFrame, levels: list, contrast_names: list) -> np.ndarray:
        """
        Compute covariate contribution per contrast: beta_c * (mean_c[A] - mean_c[B]).
        """
        cond = obs["CONDITION"].astype(str).values
        grp_masks = {g: (cond == g) for g in levels}
        C_means = np.stack([C_centered_df.to_numpy(dtype=float)[:, grp_masks[g]].mean(axis=1) for g in levels], axis=1)
        level_to_idx = {g: i for i, g in enumerate(levels)}

        cov_part = np.zeros((beta1.shape[0], len(contrast_names)), dtype=float)

        def _parse_AB(name: str):
            a, b = name.split("_vs_")
            return a, b

        for j, cname in enumerate(contrast_names):
            A, B = _parse_AB(cname)
            iA, iB = level_to_idx[A], level_to_idx[B]
            delta_c = C_means[:, iA] - C_means[:, iB]
            cov_part[:, j] = beta1 * delta_c

        return cov_part

    def _compute_missingness_like_before(adata_in: ad.AnnData) -> Tuple[pd.DataFrame, str]:
        if "qvalue" in adata_in.layers:
            miss_mat = adata_in.layers["qvalue"].T
            miss_source = "qvalue"
        elif "raw" in adata_in.layers:
            miss_mat = adata_in.layers["raw"].T
            miss_source = "raw"
        else:
            miss_mat = adata_in.X.T
            miss_source = "X"

        missing_df = StatisticalTester.compute_missingness(
            intensity_matrix = miss_mat,
            conditions       = adata_in.obs['CONDITION'].tolist(),
            feature_ids      = adata_in.var_names.tolist(),
        )
        return missing_df, miss_source

    # === ANCOVA options (new) ===
    ancova_cfg        = (config or {}).get("ancova", {})
    beta_mode         = ancova_cfg.get("beta_mode", "fixed1")      # "free" | "nonneg" | "fixed1"
    min_non_imputed   = int(ancova_cfg.get("min_non_imputed", 4))
    ridge_lambda_cfg  = float(ancova_cfg.get("ridge_lambda", 1e-6))  # forwarded into Stage-1 OLS

    # ----------------------------
    # 1) Inputs & preparation
    # ----------------------------
    obs = adata.obs.copy()
    levels, base = _get_levels_and_base(obs, config)

    #cov_layer = _find_covariate_layer(adata)
    cov_layer = "centered_covariate"

    # G×N matrices (DataFrames)
    Y_df = _genes_by_samples_df(adata.X, adata.obs_names, adata.var_names)
    C_df = _genes_by_samples_df(adata.layers[cov_layer], adata.obs_names, adata.var_names)
    if Y_df.shape != C_df.shape:
        raise ValueError(f"Covariate matrix shape {C_df.shape} mismatches expression {Y_df.shape}")

    # Ensure consistent sample order
    sample_names = obs.index.to_list()
    Y_df = Y_df.loc[:, sample_names]
    C_df = C_df.loc[:, sample_names]

    # Global row-centering of C (no scaling)
    C_centered_df = C_df.subtract(C_df.mean(axis=1), axis=0)

    # check imputation at covariate level
    raw_cov = adata.layers["raw_covariate"]
    raw_cov_arr = np.asarray(raw_cov)  # shape: (n_obs, n_vars)
    imputed_all_cov = np.all(np.isnan(raw_cov_arr), axis=0)  # (n_vars,)

    # ----------------------------
    # 2) Stage 1: OLS y ~ 1 + c
    # ----------------------------
    beta1, beta0, R, cov_se, cov_t, cov_p, cov_q = _stage1_ols_y_on_1_plus_c(
        Y_df.to_numpy(dtype=float),
        C_centered_df.to_numpy(dtype=float),
        ridge_lambda=ridge_lambda_cfg,
    )

    # ==== NEW: beta modes & low-info handling (LOD-aware) ====
    # Non-imputed sample counts per feature from your 'raw_covariate' (NaN == imputed)
    nonimp_counts = np.sum(~np.isnan(raw_cov_arr), axis=0)  # shape: (n_vars,)
    low_info = nonimp_counts < max(min_non_imputed, 2)

    # Apply the requested beta_mode
    if beta_mode == "fixed1":
        beta1[:] = 1.0
    elif beta_mode == "nonneg":
        beta1 = np.maximum(beta1, 0.0)
    elif beta_mode != "free":
        raise ValueError(f"Unknown ancova.beta_mode={beta_mode!r}")

    # Fully imputed FT rows were already flagged; combine with low-info unless fixed1
    if beta_mode != "fixed1":
        # Treat 'no/too-little FT information' as 'no adjustment' to avoid β explosions
        low_or_all = low_info | imputed_all_cov
        if np.any(low_or_all):
            beta1[low_or_all] = 0.0
            # neutralize Stage-1 test for those rows (diagnostics only)
            cov_t[low_or_all] = 0.0
            cov_p[low_or_all] = 1.0
            cov_q[low_or_all] = 1.0

    # Recompute residuals with possibly modified β
    Y_np  = Y_df.to_numpy(dtype=float)
    Cc_np = C_centered_df.to_numpy(dtype=float)
    R     = Y_np - (beta0[:, None] + beta1[:, None] * Cc_np)

    ## If a feature's FT row was fully imputed, treat covariate as “uninformative”
    ## → force β_c=0 and t=0, p=q=1 for that feature; also refresh residuals with β_c=0.
    #if imputed_all_cov is not None and np.any(imputed_all_cov):
    #    beta1[imputed_all_cov] = 0.0
    #    cov_t[imputed_all_cov] = 0.0
    #    cov_p[imputed_all_cov] = 1.0
    #    cov_q[imputed_all_cov] = 1.0
    #    Y_np = Y_df.to_numpy(dtype=float)
    #    Cc_np = C_centered_df.to_numpy(dtype=float)
    #    R[imputed_all_cov, :] = Y_np[imputed_all_cov, :] - (beta0[imputed_all_cov, None] + 0.0 * Cc_np[imputed_all_cov, :])

    R_df = pd.DataFrame(R, index=Y_df.index, columns=Y_df.columns)  # (G×N)

    # ----------------------------
    # 3) Stage 2: limma on residuals
    # ----------------------------
    design2 = _build_group_design_no_intercept(obs, levels)
    contr   = _make_contrasts(levels, design2, config)

    limma_resid = _fit_limma_with_contrasts(R_df, design2, contr)
    contrast_names = list(contr.columns)
    # --- Per-contrast mask: True where a contrast lacks FT info on at least one side ---

    # Align raw_covariate to the same sample order, then make a (G x N) non-imputed mask
    raw_cov_df = pd.DataFrame(adata.layers["raw_covariate"], index=adata.obs_names, columns=adata.var_names)
    raw_cov_df = raw_cov_df.loc[sample_names, :].T            # (G x N) to align with Y_df/C_df
    mask_nonimp = ~np.isnan(raw_cov_df.values)                # True = real FT

    cond = obs["CONDITION"].astype(str).values
    grp_masks = {g: (cond == g) for g in levels}

    contrast_noft = np.zeros((Y_df.shape[0], len(contrast_names)), dtype=bool)
    for j, cname in enumerate(contrast_names):
        A, B = cname.split("_vs_")
        cntA = mask_nonimp[:, grp_masks[A]].sum(axis=1)       # (G,)
        cntB = mask_nonimp[:, grp_masks[B]].sum(axis=1)       # (G,)
        # Mark this contrast "no FT info" if either side has zero real FT
        contrast_noft[:, j] = (cntA == 0) | (cntB == 0)


    # ----------------------------
    # 4) Also fit raw (Y) with same design/contrasts (for decomposition)
    # ----------------------------
    limma_raw = imo.lmFit(Y_df, design=design2)
    limma_raw = imo.contrasts_fit(limma_raw, contr)
    if not pilot_mode :
        with np.errstate(divide='ignore', invalid='ignore'):
            limma_raw = imo.eBayes(limma_raw)
            raw_p     = limma_raw.p_value.values
            raw_q     = np.vstack([multipletests(raw_p[:, j], method="fdr_bh")[1] for j in range(raw_p.shape[1])]).T
    else:
        raw_p = raw_q = None

    raw_coefs = limma_raw.coefficients.values

    # For "no-FT" contrasts, fall back to RAW phospho (coef + p/q) in the adjusted outputs
    if beta_mode == "fixed1":
        if not pilot_mode:
            limma_resid["coefs"][contrast_noft]    = raw_coefs[contrast_noft]
            limma_resid["p_ebayes"][contrast_noft] = raw_p[contrast_noft]
            limma_resid["q_ebayes"][contrast_noft] = raw_q[contrast_noft]
            # (Optionally also copy t/se from limma_raw if your UI shows them heavily.)
        else:
            limma_resid["coefs"] = limma_resid["p_ebayes"] = limma_resid["q_ebayes"] = None

    # --- NEW: limma on the covariate (FT) matrix itself (same design/contrasts) ---
    limma_cov = imo.lmFit(C_df, design=design2)
    limma_cov = imo.contrasts_fit(limma_cov, contr)
    if not pilot_mode:
        with np.errstate(divide='ignore', invalid='ignore'):
            limma_cov = imo.eBayes(limma_cov)
        cov_p     = limma_cov.p_value.values
        cov_q     = np.vstack([multipletests(cov_p[:, j], method="fdr_bh")[1] for j in range(cov_p.shape[1])]).T
    else:
        cov_p = cov_q = None

    cov_coefs = limma_cov.coefficients.values

    # ----------------------------
    # 5) Covariate part per contrast
    # ----------------------------
    cov_part = _covariate_part_per_contrast(
        beta1=beta1,
        C_centered_df=C_centered_df,
        obs=obs,
        levels=levels,
        contrast_names=contrast_names,
    )
    # For contrasts with no FT information, do not adjust: covariate part := 0
    if beta_mode == "fixed1":
        cov_part[contrast_noft] = 0.0


    # ----------------------------
    # 6) Missingness (unchanged)
    # ----------------------------
    missing_df, miss_source = _compute_missingness_like_before(adata)

    # ----------------------------
    # 7) Assemble outputs centrally
    # ----------------------------
    out = adata.copy()

    # Stage-2 (adjusted) statistics
    out.varm["log2fc"]     = limma_resid["coefs"]
    if not pilot_mode:
        out.varm["se_raw"]     = limma_resid["se_raw"]
        out.varm["t_raw"]      = limma_resid["t_raw"]
        out.varm["p_raw"]      = limma_resid["p_raw"]
        out.varm["q_raw"]      = limma_resid["q_raw"]
        out.varm["se_ebayes"]  = limma_resid["se_ebayes"]
        out.varm["t_ebayes"]   = limma_resid["t_ebayes"]
        out.varm["p_ebayes"]   = limma_resid["p_ebayes"]
        out.varm["q_ebayes"]   = limma_resid["q_ebayes"]

    # Raw model (same contrasts) — for decomposition display
    out.varm["raw_log2fc"]   = raw_coefs
    if not pilot_mode:
        out.varm["raw_q_ebayes"] = raw_q

    # Covariate decomposition piece
    out.varm["cov_part"] = cov_part

    # Stage-1 covariate stats (per feature)
    if not pilot_mode:
        out.varm["covariate_beta"] = beta1[:, None]
        out.varm["covariate_se"]   = cov_se[:, None]
        out.varm["covariate_t"]    = cov_t[:, None]
        out.varm["covariate_p"]    = cov_p[:, None]
        out.varm["covariate_q"]    = cov_q[:, None]

    # Residuals layer for diagnostics
    out.layers["residuals_covariate"] = R_df.T.values

    # Metadata
    out.uns["contrast_names"]   = contrast_names
    out.uns["model_type"]       = "residualization_no_interaction"
    out.uns["missingness"]      = missing_df
    out.uns["missingness_source"] = miss_source
    out.uns["missingness_rule"] = "nan-is-missing"

    out.uns["has_covariate"] = True
    out.uns["pilot_study_mode"] = bool(pilot_mode)

    # --- NEW: stash covariate (FT) limma results for the viewer ---
    out.varm["ft_log2fc"]     = cov_coefs
    out.varm["ft_q_ebayes"]   = cov_q
    # (optional) p-values too:
    out.varm["ft_p_ebayes"]   = cov_p

    # Describe decomposition rule once for the viewer
    out.uns["decomposition_rule"] = "raw_log2fc ≈ log2fc (adjusted) + cov_part"

    # Keep your quick diagnostics (you said you'll delete later)
    #_quick_print_two(out, ids=("RRPESAPAESSPSK|p5","RRPESAPAESSPSK|p11", "GILAADESTGSIAKR|p8", "AAVGQESPGGLEAGNAKAPK|p7", "SAGALEEGTSEGQLCGR|p1"))
    #check_ft_consistency(out, protein_id="Q9NW97")

    return out

def _quick_print_two(adata, ids=("RRPESAPAESSPSK|p5","RRPESAPAESSPSK|p11")):

    obs_names = list(adata.obs_names)
    var_names = list(adata.var.index)
    cond = adata.obs["CONDITION"].astype(str)
    groups = list(pd.Categorical(cond).categories)

    def _as_feature_df(arr_like):
        if arr_like is None:
            return None
        A = np.asarray(arr_like)
        # expect (features x samples)
        if A.shape == (len(obs_names), len(var_names)):
            A = A.T
        elif A.shape != (len(var_names), len(obs_names)):
            raise ValueError(f"Unexpected shape {A.shape}; expected "
                             f"({len(var_names)}, {len(obs_names)}) or "
                             f"({len(obs_names)}, {len(var_names)}).")
        return pd.DataFrame(A, index=var_names, columns=obs_names)

    Y = _as_feature_df(adata.X)
    R = _as_feature_df(adata.layers.get("residuals_covariate")) if ("residuals_covariate" in adata.layers) else None

    # pick whichever covariate layer exists
    C = None
    for k in ("centered_covariate","imputed_covariate","cov","ft"):
        if k in adata.layers:
            C = _as_feature_df(adata.layers[k])
            break

    # limma outputs (stage 2)
    contrast_names = adata.uns.get("contrast_names", [])
    lfc_arr = adata.varm.get("log2fc", None)
    if lfc_arr is None:
        LFC = pd.DataFrame(index=var_names, columns=contrast_names, dtype=float)
    else:
        LFC = pd.DataFrame(np.asarray(lfc_arr), index=var_names, columns=contrast_names)

    qE_arr = adata.varm.get("q_ebayes", None)
    if qE_arr is None:
        qE = pd.DataFrame(index=var_names, columns=contrast_names, dtype=float)
    else:
        qE = pd.DataFrame(np.asarray(qE_arr), index=var_names, columns=contrast_names)

    # stage 1 covariate stats (optional)
    cov_beta = np.ravel(adata.varm["covariate_beta"]) if "covariate_beta" in adata.varm else None
    cov_se   = np.ravel(adata.varm["covariate_se"])   if "covariate_se"   in adata.varm else None
    cov_p    = np.ravel(adata.varm["covariate_p"])    if "covariate_p"    in adata.varm else None
    cov_q    = np.ravel(adata.varm["covariate_q"])    if "covariate_q"    in adata.varm else None

    def _means_by_group(M, rid):
        if M is None: return {}
        out = {}
        for g in groups:
            mask = (cond == g).values
            out[g] = float(np.nanmean(M.loc[rid, mask]))
        return out

    print("\n=== QUICK CHECK (ANCOVA residualization; no interaction) ===")
    for rid in ids:
        if rid not in var_names:
            print(f"\n--- {rid} : NOT FOUND ---")
            continue

        print("\n" + "="*80)
        print(f"Feature: {rid}")

        print("\nPost-imputation Y means by condition:")
        print(_means_by_group(Y, rid))

        if C is not None:
            print("Covariate (globally centered) means by condition:")
            print(_means_by_group(C, rid))

        if cov_beta is not None:
            i = adata.var.index.get_loc(rid)
            b  = float(cov_beta[i])
            se = float(cov_se[i]) if cov_se is not None else np.nan
            p  = float(cov_p[i])  if cov_p  is not None else np.nan
            q  = float(cov_q[i])  if cov_q  is not None else np.nan
            print("\nStage-1 (phospho ~ 1 + covariate) per-gene slope:")
            print(f"  beta_c = {b:.4g}   SE = {se:.4g}   p = {p:.3g}   q = {q:.3g}")

        if R is not None:
            print("Residual means by condition (should be near 0 if well adjusted):")
            print(_means_by_group(R, rid))

        if not LFC.empty:
            print("\nAdjusted phospho contrasts (log2FC on residuals):")
            row = LFC.loc[rid]
            for cn in contrast_names:
                qe = (qE.loc[rid, cn] if (cn in qE.columns) else np.nan)
                print(f"  {cn:<12}  log2FC={row[cn]:>8.4f}   q={qe:>6.3g}")

    # also show raw and cov_part if present
    raw = adata.varm.get("raw_log2fc", None)
    covp = adata.varm.get("cov_part", None)
    if raw is not None or covp is not None:
        rawDF = (pd.DataFrame(np.asarray(raw), index=var_names, columns=contrast_names)
                 if raw is not None else None)
        covDF = (pd.DataFrame(np.asarray(covp), index=var_names, columns=contrast_names)
                 if covp is not None else None)

        print("\nDecomposition per contrast:")
        print("  Raw ≈ Adjusted + CovariatePart")
        for cn in contrast_names:
            l_adj = float(LFC.loc[rid, cn]) if cn in LFC.columns else np.nan
            l_raw = float(rawDF.loc[rid, cn]) if (rawDF is not None and cn in rawDF.columns) else np.nan
            l_cov = float(covDF.loc[rid, cn]) if (covDF is not None and cn in covDF.columns) else np.nan
            print(f"  {cn:<12}  Raw={l_raw:>8.4f}  Adj={l_adj:>8.4f}  CovPart={l_cov:>8.4f}")

def check_ft_consistency(adata, protein_id="Q8NEN9"):
    # Extract mapping
    var = adata.var
    if "PARENT_PROTEIN" not in var:
        raise ValueError("No PARENT_PROTEIN column in adata.var")

    sites = var.index[var["PARENT_PROTEIN"] == protein_id]
    print(f"Protein {protein_id} — {len(sites)} sites")

    # Extract the flowthrough (FT) data
    ft = pd.DataFrame(
        adata.layers["centered_covariate"],  # or "centered_covariate" if that’s your FT layer
        index=adata.obs_names,
        columns=adata.var_names
    ).T

    ft2 = pd.DataFrame(
        adata.layers["processed_covariate"],  # or "centered_covariate" if that’s your FT layer
        index=adata.obs_names,
        columns=adata.var_names
    ).T


    # Extract the limma results (raw flowthrough fit)
    ft_logfc = pd.DataFrame(
        np.asarray(adata.varm["ft_log2fc"]),
        index=adata.var_names,
        columns=adata.uns["contrast_names"]
    )
    ft_q = pd.DataFrame(
        np.asarray(adata.varm["ft_q_ebayes"]),
        index=adata.var_names,
        columns=adata.uns["contrast_names"]
    )

    # Print raw differences
    for c in adata.uns["contrast_names"]:
        vals = ft_logfc.loc[sites, c]
        qs = ft_q.loc[sites, c]
        print(f"\nContrast: {c}")
        print(vals)
        print(qs)
        print(f"Δ logFC range: {vals.max()-vals.min():.4g}")
        print(f"Δ q range: {qs.max()-qs.min():.4g}")

    # Also check the actual FT intensities for equality
    ft_equal = np.allclose(ft.loc[sites].to_numpy(), ft.loc[sites[0]].to_numpy())
    print(ft.loc[sites])
    print(ft2.loc[sites])
    print("\nFT values identical across sites:", ft_equal)

