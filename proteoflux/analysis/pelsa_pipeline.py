"""PELSA curve-fitting workflow."""

import numpy as np
import pandas as pd
import anndata as ad

from proteoflux.utils.utils import log_time, log_info
from proteoflux.analysis.pelsa_torch import fit_4pl_torch_from_ratio_df


def _log2_mean(values: np.ndarray) -> float:
    """Mean on linear scale, returned on log2 scale."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.log2(np.mean(np.exp2(values))))


def _build_pelsa_ratio_table(
    adata: ad.AnnData,
    *,
    concentration_col: str = "Concentration",
    layer: str = "normalized",
    control_concentration: float = 0.0,
) -> pd.DataFrame:
    """Build long peptide × sample ratio table for PELSA curve fitting.

    Input matrix is expected on log2 scale.
    Ratios are computed against the mean control abundance per peptide.
    """

    if concentration_col not in adata.obs.columns:
        raise ValueError(
            f"PELSA requires adata.obs[{concentration_col!r}]. "
            "This should have been created from the annotation file."
        )

    if layer not in adata.layers:
        raise ValueError(f"PELSA requires adata.layers[{layer!r}].")

    conc = pd.to_numeric(adata.obs[concentration_col], errors="raise").astype(float)
    control_mask = conc.to_numpy() == float(control_concentration)

    if not np.any(control_mask):
        raise ValueError(
            f"PELSA requires at least one control sample with "
            f"{concentration_col} == {control_concentration}."
        )

    X_log2 = np.asarray(adata.layers[layer], dtype=float)
    if X_log2.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"Unexpected PELSA matrix shape for layer {layer!r}: "
            f"{X_log2.shape}, expected {(adata.n_obs, adata.n_vars)}."
        )

    control_ref = np.apply_along_axis(_log2_mean, 0, X_log2[control_mask, :])
    log2_ratio = X_log2 - control_ref[None, :]

    ratio = np.exp2(log2_ratio)
    ratio[~np.isfinite(ratio)] = np.nan

    sample_names = np.asarray(adata.obs_names.astype(str))
    peptide_ids = np.asarray(adata.var_names.astype(str))

    sample_idx, peptide_idx = np.where(np.isfinite(ratio))

    out = pd.DataFrame(
        {
            "peptide_id": peptide_ids[peptide_idx],
            "sample": sample_names[sample_idx],
            "concentration": conc.to_numpy()[sample_idx],
            "log10_concentration": np.nan,
            "log2_intensity": X_log2[sample_idx, peptide_idx],
            "log2_ratio": log2_ratio[sample_idx, peptide_idx],
            "ratio": ratio[sample_idx, peptide_idx],
        }
    )

    nonzero = out["concentration"].to_numpy(dtype=float) > 0
    out.loc[nonzero, "log10_concentration"] = np.log10(
        out.loc[nonzero, "concentration"].to_numpy(dtype=float)
    )

    if "REPLICATE" in adata.obs.columns:
        repl = adata.obs["REPLICATE"].astype(str)
        out["replicate"] = repl.to_numpy()[sample_idx]

    return out


def _summarize_ratio_input(
    ratio_df: pd.DataFrame,
    *,
    control_concentration: float = 0.0,
) -> dict:
    conc = ratio_df["concentration"].to_numpy(dtype=float)

    return {
        "n_ratio_points": int(len(ratio_df)),
        "n_peptides_with_any_ratio": int(ratio_df["peptide_id"].nunique()),
        "n_concentrations": int(pd.Series(conc).nunique()),
        "n_control_points": int(np.sum(conc == control_concentration)),
        "n_nonzero_points": int(np.sum(conc > 0)),
        "control_concentration": float(control_concentration),
    }


def _fit_all_4pl_curves(ratio_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return fit_4pl_torch_from_ratio_df(ratio_df, config)


@log_time("PELSA pipeline")
def run_pelsa_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    adata = adata.copy()

    pelsa_cfg = (
        ((config or {}).get("analysis", {}) or {})
        .get("pelsa", {})
        or {}
    )

    concentration_col = pelsa_cfg.get("concentration_column", "Concentration")
    layer = pelsa_cfg.get("layer", "normalized")
    control_concentration = float(pelsa_cfg.get("control_concentration", 0.0))

    ratio_df = _build_pelsa_ratio_table(
        adata,
        concentration_col=concentration_col,
        layer=layer,
        control_concentration=control_concentration,
    )

    curve_results = _fit_all_4pl_curves(ratio_df, config)

    n_success = int(curve_results["fit_success"].sum())
    log_info(
        f"PELSA 4PL fitting done: {n_success}/{len(curve_results)} successful fits."
    )

    summary = _summarize_ratio_input(
        ratio_df,
        control_concentration=control_concentration,
    )

    nonzero_conc = np.sort(
        ratio_df.loc[
            ratio_df["concentration"].to_numpy(dtype=float) > 0,
            "concentration",
        ].dropna().unique().astype(float)
    )

    log_info(
        "PELSA ratio table ready: "
        f"{summary['n_peptides_with_any_ratio']} peptides, "
        f"{summary['n_ratio_points']} finite ratio points, "
        f"{summary['n_concentrations']} concentration levels."
    )

    adata.uns.setdefault("analysis", {})
    adata.uns["analysis"]["analysis_type"] = "pelsa"
    adata.uns["analysis"]["analysis_method"] = "pelsa_curve_fit"

    adata.uns["pelsa"] = {
        "status": "curves_fitted",
        "concentration_column": concentration_col,
        "layer": layer,
        "control_concentration": control_concentration,
        "control_reference": "mean_linear",
        "curve_model": {
            "name": "4pl_log_logistic",
            "x_scale": "log10_concentration",
            "y_scale": "ratio_to_control",
            "formula": "back + (front - back) / (1 + 10 ** (slope * (x + pec50)))",
        },
        "concentrations": np.r_[control_concentration, nonzero_conc],
        "summary": summary,
        "curve_points": ratio_df,
        "curve_results": curve_results,
    }

    return adata
