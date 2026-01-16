"""Limma-based differential analysis (with optional covariate residualization).

This module provides:
  - `run_limma_pipeline`: standard limma path (with eBayes if feasible)
  - `run_limma_pipeline_covariate`: two-stage residualization (no interaction)
  - `clustering_pipeline`: helper to run clustering + missingness clustering
"""

import anndata as ad
import pandas as pd
import numpy as np
import patsy
from typing import Tuple, Optional
import inmoose.limma as imo
from itertools import combinations
from proteoflux.utils.utils import log_time, log_warning, log_info
from proteoflux.analysis.clustering import run_clustering, run_clustering_missingness
from proteoflux.analysis.stats_ops import raw_stats_from_fit, bh_qvalues, two_sided_t_pvalue
from proteoflux.analysis.missingness import compute_missingness
from proteoflux.analysis.adata_schema import (
    # .uns
    UNS_CONTRAST_NAMES,
    UNS_PILOT_MODE,
    UNS_HAS_COVARIATE,
    UNS_MISSINGNESS,
    UNS_MISSINGNESS_SOURCE,
    UNS_MISSINGNESS_RULE,
    UNS_N_FULLY_IMPUTED_CELLS,
    UNS_RESIDUAL_VARIANCE,
    # .varm
    VARM_LOG2FC,
    VARM_SE_RAW,
    VARM_T_RAW,
    VARM_P_RAW,
    VARM_Q_RAW,
    VARM_SE_EBAYES,
    VARM_T_EBAYES,
    VARM_P_EBAYES,
    VARM_Q_EBAYES,
    VARM_RAW_LOG2FC,
    VARM_RAW_Q_EBAYES,
    VARM_COV_PART,
    VARM_COVARIATE_BETA,
    VARM_COVARIATE_SE,
    VARM_COVARIATE_T,
    VARM_COVARIATE_P,
    VARM_COVARIATE_Q,
    VARM_FT_LOG2FC,
    VARM_FT_P_EBAYES,
    VARM_FT_Q_EBAYES,
)


def _genes_by_samples_df(X_like: np.ndarray, obs_names, var_names) -> pd.DataFrame:
    """Return (G × N) as DataFrame indexed by features with sample columns."""
    return pd.DataFrame(X_like, index=obs_names, columns=var_names).T


def _design_no_intercept(obs: pd.DataFrame, levels: list[str]) -> patsy.DesignMatrix:
    """Patsy design matrix: 0 + levelA + levelB + ... (mechanical shared helper)."""
    tmp = obs.copy()
    for lvl in levels:
        tmp[lvl] = (tmp["CONDITION"] == lvl).astype(int)
    formula = "0 + " + " + ".join(levels)
    return patsy.dmatrix(formula, tmp)


def _fit_limma_with_contrasts(
    *,
    df_GxN: pd.DataFrame,
    design_dm,
    contrasts_df: pd.DataFrame,
    pilot_mode: bool,
    fully_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Shared limma fit path used by standard and covariate branches.

    Guarantees:
      - raw stats computed via shared primitive
      - BH q-values computed after applying fully_mask to p-values (if provided)
      - eBayes stats (if not pilot) likewise masked before BH
    """
    fit = imo.lmFit(df_GxN, design=design_dm)
    fit = imo.contrasts_fit(fit, contrasts=contrasts_df)

    coefs  = fit.coefficients.values
    stdu   = fit.stdev_unscaled.values
    sigma  = fit.sigma.to_numpy()
    df_res = fit.df_residual

    se_raw, t_raw, p_raw = raw_stats_from_fit(coefs=coefs, stdu=stdu, sigma=sigma, df_res=df_res)
    if fully_mask is not None and np.any(fully_mask):
        p_raw = np.asarray(p_raw, dtype=float)
        p_raw[fully_mask] = 1.0
    q_raw = bh_qvalues(p_raw)

    if not pilot_mode:
        with np.errstate(divide="ignore", invalid="ignore"):
            fit = imo.eBayes(fit)
        s2post     = fit.s2_post.to_numpy()
        se_ebayes  = stdu * np.sqrt(s2post[:, None])
        t_ebayes   = fit.t.values
        p_ebayes   = fit.p_value.values
        if fully_mask is not None and np.any(fully_mask):
            p_ebayes = np.asarray(p_ebayes, dtype=float)
            p_ebayes[fully_mask] = 1.0
        q_ebayes = bh_qvalues(p_ebayes)
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

def _contrast_defs_from_cfg(levels: list[str], cfg: dict) -> list[str]:
    """Return list of limma contrast definitions like ['A - B', ...] (mechanical extraction)."""
    analysis_cfg = (cfg or {}).get("analysis", {}) or {}
    only_list = analysis_cfg.get("only_contrasts") or []
    base = analysis_cfg.get("only_against", None)

    if isinstance(only_list, str):
        only_list = [only_list]

    def _parse_only_list(items: list[str], valid_levels: list[str]) -> list[str]:
        out: list[str] = []
        for it in items:
            s = str(it)
            if "_vs_" in s:
                A, B = s.split("_vs_", 1)
            elif "_v_" in s:
                A, B = s.split("_v_", 1)
            else:
                raise ValueError(f"only_contrasts item '{s}' must use _v_ or _vs_ as separator")
            A, B = A.strip(), B.strip()
            if A not in valid_levels or B not in valid_levels:
                raise ValueError(f"only_contrasts '{s}': conditions must be in {valid_levels}")
            out.append(f"{A} - {B}")
        return out

    if len(only_list) > 0:
        log_info("Analysis only on selected contrasts")
        return _parse_only_list(only_list, levels)
    if base is not None:
        if base not in levels:
            raise ValueError(f"only_against='{base}' not found in Condition levels {levels}")
        log_info("Analysis only on against selected condition")
        return [f"{c} - {base}" for c in levels if c != base]

    return [f"{a} - {b}" for a, b in combinations(levels, 2)]

def _compute_missingness_payload(adata_in: ad.AnnData) -> tuple[pd.DataFrame, str]:
    """Compute missingness over the full feature set (downstream depends on full-length alignment)."""
    res = compute_missingness(adata_in, max_features=None, random_seed=0)
    return res.df, res.source

def _make_contrasts(levels: list[str], design_dm, cfg: dict) -> pd.DataFrame:
    """Build limma contrasts matrix (mechanical extraction)."""
    defs = _contrast_defs_from_cfg(levels, cfg)
    contrast_df = imo.makeContrasts(defs, levels=design_dm)
    contrast_df.columns = [c.replace(" - ", "_vs_") for c in contrast_df.columns]
    return contrast_df


def _fully_imputed_mask_from_layer(
    adata: ad.AnnData,
    layer_name: str,
    conditions: np.ndarray,
    contrast_names: list[str],
) -> np.ndarray:
    """Return (n_vars × n_contrasts) mask: True where BOTH sides are fully missing in raw."""
    if layer_name not in adata.layers:
        raise ValueError(
            f"Cannot detect fully-imputed contrasts: missing adata.layers[{layer_name!r}] "
            "(pre-imputation matrix required, with NAs)."
        )
    raw = np.asarray(adata.layers[layer_name])
    if raw.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(f"Unexpected {layer_name!r} layer shape {raw.shape}, expected {(adata.n_obs, adata.n_vars)}")
    raw_T = raw.T  # (n_vars × n_obs)

    fully = np.zeros((adata.n_vars, len(contrast_names)), dtype=bool)
    for j, cname in enumerate(contrast_names):
        if "_vs_" not in cname:
            raise ValueError(f"Contrast name {cname!r} does not match expected '<A>_vs_<B>'")
        A, B = cname.split("_vs_", 1)
        maskA = conditions == A
        maskB = conditions == B
        if not maskA.any() or not maskB.any():
            raise ValueError(f"Contrast {cname!r}: missing samples for A={A!r} or B={B!r}")
        fully[:, j] = np.all(np.isnan(raw_T[:, maskA]), axis=1) & np.all(np.isnan(raw_T[:, maskB]), axis=1)
    return fully

def _neutralize_fully_imputed_contrasts(out: ad.AnnData, fully: np.ndarray) -> None:
    """Overwrite outputs for fully-imputed (feature × contrast) cells (post-fit, explicit)."""
    if fully.shape != out.varm[VARM_LOG2FC].shape:
        raise ValueError(
            f"Fully-imputed mask shape {fully.shape} does not match log2fc shape {out.varm[VARM_LOG2FC].shape}"
        )
    if not np.any(fully):
        return

    out.varm[VARM_LOG2FC][fully] = 0.0
    for key in (VARM_P_RAW, VARM_Q_RAW, VARM_P_EBAYES, VARM_Q_EBAYES):
        if key in out.varm:
            out.varm[key][fully] = 1.0
    for key in (VARM_T_RAW, VARM_T_EBAYES):
        if key in out.varm:
            out.varm[key][fully] = 0.0
    for key in (VARM_SE_RAW, VARM_SE_EBAYES):
        if key in out.varm:
            out.varm[key][fully] = np.inf

@log_time("Analysis pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Standard limma workflow on `adata.X` with optional eBayes.

    If dataset config contains injected covariate runs, the covariate branch is used.
    """

    injected_runs = (config or {}).get("dataset", {}).get("inject_runs") or {}
    has_covariate_cfg = any(bool(run.get("is_covariate")) for _, run in injected_runs.items())

    # Read design & detect pilot mode (≤1 replicate for a condition)
    obs = adata.obs.copy()

    repl_counts = obs["CONDITION"].value_counts()
    levels = sorted(obs["CONDITION"].unique())
    pilot_mode   = (repl_counts.min() <= 1) or (len(levels) < 2)

    # Early-out when there is only one condition: mark pilot mode and skip stats cleanly
    if len(levels) < 2:
        log_warning("Pilot study mode: only 1 condition detected; skipping statistical analysis.")

        missing_df, miss_source = _compute_missingness_payload(adata)

        out = adata.copy()
        out.uns[UNS_CONTRAST_NAMES] = []
        out.uns[UNS_PILOT_MODE] = True
        out.uns[UNS_MISSINGNESS] = missing_df
        out.uns[UNS_MISSINGNESS_SOURCE] = miss_source
        out.uns[UNS_MISSINGNESS_RULE] = "nan-is-missing"
        out.uns[UNS_HAS_COVARIATE] = False
        return out

    # Pilot mode due to singleton replicates (keep existing behavior/logging)
    if repl_counts.min() <= 1:
        log_warning(
            "Pilot study mode: at least one condition has only 1 replicate, no statistical analysis done. "
        )

    if has_covariate_cfg:
        return run_limma_pipeline_covariate(adata, config, pilot_mode)

    if len(levels) < 2:
        raise ValueError(f"Need ≥2 conditions for contrasts; found {levels}")

    design_dm = _design_no_intercept(obs, levels)

    # Expression: genes × samples
    df_X = _genes_by_samples_df(adata.X, adata.obs_names, adata.var_names)

    contrast_df = _make_contrasts(levels, design_dm, config)
    contrast_names = list(contrast_df.columns)

    # Fully-imputed detection (contrast-local) — used for pre-BH masking in shared fit helper.
    cond_arr = adata.obs["CONDITION"].astype(str).to_numpy()
    fully = _fully_imputed_mask_from_layer(adata=adata, layer_name="raw", conditions=cond_arr, contrast_names=contrast_names)

    limma_res = _fit_limma_with_contrasts(
        df_GxN=df_X,
        design_dm=design_dm,
        contrasts_df=contrast_df,
        pilot_mode=pilot_mode,
        fully_mask=fully,
    )

    # Residual variance (from the raw lmFit, matching prior behavior)
    fit_for_var = imo.lmFit(df_X, design=design_dm)
    resid_var = np.asarray(fit_for_var.sigma, dtype=np.float32) ** 2
    adata.uns[UNS_RESIDUAL_VARIANCE] = resid_var

    # Assemble into AnnData
    out = adata.copy()

    out.varm[VARM_LOG2FC] = limma_res["coefs"]

    if not pilot_mode:
        out.varm[VARM_SE_RAW] = limma_res["se_raw"]
        out.varm[VARM_T_RAW] = limma_res["t_raw"]
        out.varm[VARM_P_RAW] = limma_res["p_raw"]
        out.varm[VARM_Q_RAW] = limma_res["q_raw"]

        # moderated statistics
        out.varm[VARM_SE_EBAYES] = limma_res["se_ebayes"]
        out.varm[VARM_T_EBAYES] = limma_res["t_ebayes"]
        out.varm[VARM_P_EBAYES] = limma_res["p_ebayes"]
        out.varm[VARM_Q_EBAYES] = limma_res["q_ebayes"]

    # metadata

    out.uns[UNS_CONTRAST_NAMES] = contrast_names
    out.uns[UNS_PILOT_MODE] = bool(pilot_mode)

    if not pilot_mode:
        _neutralize_fully_imputed_contrasts(out=out, fully=fully)
        out.uns[UNS_N_FULLY_IMPUTED_CELLS] = int(np.count_nonzero(fully))


    # Missingness
    missing_df, miss_source = _compute_missingness_payload(adata)

    out.uns[UNS_MISSINGNESS] = missing_df
    out.uns[UNS_MISSINGNESS_SOURCE] = miss_source
    out.uns[UNS_MISSINGNESS_RULE] = "nan-is-missing"
    out.uns[UNS_HAS_COVARIATE] = False

    return out

@log_time("Clustering")
def clustering_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Run feature clustering and missingness clustering (unchanged behavior)."""
    analysis_cfg = (config or {}).get("analysis", {}) or {}
    max_features = int(analysis_cfg.get("clustering_max", 8000))
    adata = run_clustering(adata,
                           n_pcs=adata.X.shape[0]-1,
                           max_features=max_features,
                           layer="normalized")
    adata = run_clustering_missingness(adata, max_features=max_features)

    return adata

@log_time("Running Covariate branch")
def run_limma_pipeline_covariate(adata: ad.AnnData, config: dict, pilot_mode: bool) -> ad.AnnData:
    """Two-stage residualization (no interaction).

    Stage 1:  Y ~ 1 + C         (per-feature OLS with global row-centering of C)
    Stage 2:  R ~ 0 + Group     (limma on residuals; same contrasts as raw)

    Outputs/keys match the existing implementation; the decomposition
    rule 'raw ≈ adjusted + covariate_part' is preserved.


    All outputs/keys match the current implementation.
    """

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

    def _stage1_ols_y_on_1_plus_c(Y: np.ndarray, C: np.ndarray, ridge_lambda: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Closed-form OLS for y ~ 1 + c per feature with global row-centering of C already applied.
        Returns:
          beta1 (G,), beta0 (G,), residuals R (G×N), se_beta1 (G,), t_beta1 (G,), p_beta1 (G,), q_beta1 (G,)
        """
        G, N = Y.shape
        df1 = max(N - 2, 1)

        sY  = Y.sum(axis=1)        # (G,)
        sYC = (Y * C).sum(axis=1)  # (G,)
        sCC = (C * C).sum(axis=1)  # (G,)

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

        # Smart floor for near-flat covariates (treat as no covariate effect)
        # threshold scales with typical covariate energy, but never below 1e-3
        tau = max(1e-2, ridge_baseline * 1e-6)
        # also require a minimum per-row SD to consider the covariate "non-flat"
        # (handles fully-imputed jitter where sCC is small but not tiny)
        sdC = np.sqrt(sCC / max(N - 1, 1))
        eps_sd = 0.10  # log2 scale; ~2x jitter (0.05)
        flat = (sCC < tau) | (sdC < eps_sd)
        if np.any(flat):
            beta1[flat] = 0.0

        Yhat = beta0[:, None] + beta1[:, None] * C
        R    = Y - Yhat

        rss = np.sum(R * R, axis=1)
        s2  = rss / df1
        with np.errstate(divide="ignore", invalid="ignore"):
            se1 = np.sqrt(s2 / np.maximum(sCC_tilde, 1e-15))

        # t-statistics and FDR for covariate slope
        t1 = np.where(se1 > 0, beta1 / se1, 0.0)
        p1 = two_sided_t_pvalue(t1, df=df1)
        if np.any(flat):
            # for “flat” rows, explicitly null the test: t=0, p=1 (avoids huge betas)
            t1[flat] = 0.0
            p1[flat] = 1.0
        q1 = bh_qvalues(p1[:, None])[:, 0]

        return beta1, beta0, R, se1, t1, p1, q1

    def _covariate_part_per_contrast(beta1: np.ndarray, C_centered_df: pd.DataFrame,
                                     obs: pd.DataFrame, levels: list, contrast_names: list) -> np.ndarray:
        """Compute covariate contribution per contrast.

        cov_part = beta_c * (mean_c[A] - mean_c[B])  for each contrast A_vs_B.
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

    # === ANCOVA options ===
    ancova_cfg        = (config or {}).get("ancova", {})
    # Different beta modes for testing. Now hard fix to fixed1, because occam.
    beta_mode         = ancova_cfg.get("beta_mode", "fixed1")      # "free" | "nonneg" | "fixed1"
    ridge_lambda_cfg  = float(ancova_cfg.get("ridge_lambda", 1e-6))  # forwarded into Stage-1 OLS

    # Inputs & preparation
    obs = adata.obs.copy()
    levels, base = _get_levels_and_base(obs, config)

    # Option for testing, now hard fixed
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

    # Stage 1: OLS y ~ 1 + c
    beta1, beta0, R, cov_se, cov_t, cov_p, cov_q = _stage1_ols_y_on_1_plus_c(
        Y_df.to_numpy(dtype=float),
        C_centered_df.to_numpy(dtype=float),
        ridge_lambda=ridge_lambda_cfg,
    )

    # Apply the requested beta_mode
    if beta_mode == "fixed1":
        beta1[:] = 1.0
    elif beta_mode == "nonneg":
        beta1 = np.maximum(beta1, 0.0)
    elif beta_mode != "free":
        raise ValueError(f"Unknown ancova.beta_mode={beta_mode!r}")

    # Fully imputed FT rows were already flagged; combine with low-info unless fixed1
    if beta_mode != "fixed1":
        # Per-contrast: if one condition has zero non-imputed FT samples, do not adjust.
        no_adjust = contrast_noft | imputed_all_cov[:, None]
        if np.any(no_adjust):
            beta1[no_adjust] = 0.0
            cov_t[no_adjust] = 0.0
            cov_p[no_adjust] = 1.0
            cov_q[no_adjust] = 1.0


    # Recompute residuals with possibly modified betas
    Y_np  = Y_df.to_numpy(dtype=float)
    Cc_np = C_centered_df.to_numpy(dtype=float)
    R     = Y_np - (beta0[:, None] + beta1[:, None] * Cc_np)

    R_df = pd.DataFrame(R, index=Y_df.index, columns=Y_df.columns)  # (G×N)

    # Stage 2: limma on residuals
    design2 = _design_no_intercept(obs, levels)
    contr   = _make_contrasts(levels, design2, config)

    contrast_names = list(contr.columns)

    cond_arr = obs["CONDITION"].astype(str).to_numpy()
    fully = _fully_imputed_mask_from_layer(
        adata=adata,
        layer_name="raw",
        conditions=cond_arr,
        contrast_names=contrast_names,
    )

    limma_resid = _fit_limma_with_contrasts(
        df_GxN=R_df,
        design_dm=design2,
        contrasts_df=contr,
        pilot_mode=pilot_mode,
        fully_mask=fully,
    )

    # Per-contrast mask: True where a contrast lacks FT info on at least one side
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

    limma_raw = _fit_limma_with_contrasts(
        df_GxN=Y_df,
        design_dm=design2,
        contrasts_df=contr,
        pilot_mode=pilot_mode,
        fully_mask=fully,
    )
    raw_coefs = limma_raw["coefs"]
    raw_p = limma_raw["p_ebayes"] if not pilot_mode else None
    raw_q = limma_raw["q_ebayes"] if not pilot_mode else None

    # For "no-FT" contrasts, fall back to RAW phospho (coef + p/q) in the adjusted outputs
    if beta_mode == "fixed1":
        if not pilot_mode:
            limma_resid["coefs"][contrast_noft]    = raw_coefs[contrast_noft]
            limma_resid["p_ebayes"][contrast_noft] = raw_p[contrast_noft]
            limma_resid["q_ebayes"][contrast_noft] = raw_q[contrast_noft]
        else:
            limma_resid["p_ebayes"] = limma_resid["q_ebayes"] = None
            limma_resid["coefs"][contrast_noft] = raw_coefs[contrast_noft]

    # limma on the covariate (FT) matrix itself (same design/contrasts)
    fully_ft = None
    if not pilot_mode:
        fully_ft = _fully_imputed_mask_from_layer(
            adata=adata,
            layer_name="raw_covariate",
            conditions=cond_arr,
            contrast_names=contrast_names,
        )
    limma_cov = _fit_limma_with_contrasts(
        df_GxN=C_df,
        design_dm=design2,
        contrasts_df=contr,
        pilot_mode=pilot_mode,
        fully_mask=fully_ft,
    )
    cov_coefs = limma_cov["coefs"]
    cov_p = limma_cov["p_ebayes"] if not pilot_mode else None
    cov_q = limma_cov["q_ebayes"] if not pilot_mode else None

    # Covariate part per contrast
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


    # Missingness
    missing_df, miss_source = _compute_missingness_payload(adata)

    # Assemble outputs centrally
    out = adata.copy()

    # Stage-2 (adjusted) statistics
    out.varm[VARM_LOG2FC] = limma_resid["coefs"]

    if not pilot_mode:
        out.varm[VARM_SE_RAW] = limma_resid["se_raw"]
        out.varm[VARM_T_RAW] = limma_resid["t_raw"]
        out.varm[VARM_P_RAW] = limma_resid["p_raw"]
        out.varm[VARM_Q_RAW] = limma_resid["q_raw"]
        out.varm[VARM_SE_EBAYES] = limma_resid["se_ebayes"]
        out.varm[VARM_T_EBAYES] = limma_resid["t_ebayes"]
        out.varm[VARM_P_EBAYES] = limma_resid["p_ebayes"]
        out.varm[VARM_Q_EBAYES] = limma_resid["q_ebayes"]

        # Residual variance from stage-2 limma (adjusted model)
        fit_resid = imo.lmFit(R_df, design=design2)
        resid_var = np.asarray(fit_resid.sigma, dtype=np.float32) ** 2
        out.uns[UNS_RESIDUAL_VARIANCE] = resid_var

    # Raw model (same contrasts) - for decomposition display
    out.varm[VARM_RAW_LOG2FC] = raw_coefs
    if not pilot_mode:
        out.varm[VARM_RAW_Q_EBAYES] = raw_q

    # Covariate decomposition piece
    out.varm[VARM_COV_PART] = cov_part

    # Stage-1 covariate stats (per feature)
    if not pilot_mode:
        out.varm[VARM_COVARIATE_BETA] = beta1[:, None]
        out.varm[VARM_COVARIATE_SE] = cov_se[:, None]
        out.varm[VARM_COVARIATE_T] = cov_t[:, None]
        out.varm[VARM_COVARIATE_P] = cov_p[:, None]
        out.varm[VARM_COVARIATE_Q] = cov_q[:, None]

    # Residuals layer for diagnostics
    out.layers["residuals_covariate"] = R_df.T.values

    # Metadata
    out.uns[UNS_CONTRAST_NAMES] = contrast_names
    out.uns["model_type"] = "residualization_no_interaction"

    out.uns[UNS_MISSINGNESS] = missing_df
    out.uns[UNS_MISSINGNESS_SOURCE] = miss_source
    out.uns[UNS_MISSINGNESS_RULE] = "nan-is-missing"

    out.uns[UNS_HAS_COVARIATE] = True
    out.uns[UNS_PILOT_MODE] = bool(pilot_mode)

    if (not pilot_mode) and (fully is not None):
        _neutralize_fully_imputed_contrasts(out=out, fully=fully)
        out.uns[UNS_N_FULLY_IMPUTED_CELLS] = int(np.count_nonzero(fully))

    # FT limma results (already fully-masked pre-BH in shared fit helper)
    out.varm[VARM_FT_LOG2FC] = cov_coefs

    if not pilot_mode:
        out.varm[VARM_FT_Q_EBAYES] = cov_q
        out.varm[VARM_FT_P_EBAYES] = cov_p

    # Describe decomposition rule once for the viewer
    out.uns["decomposition_rule"] = "raw_log2fc ≈ log2fc (adjusted) + cov_part"

    return out
