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


def _control_log10_position(nonzero_concentrations: np.ndarray) -> float:
    """Return a finite plotting/fitting x-position for vehicle controls.

    A zero concentration cannot be represented on a log10 axis. For 4PL fitting
    and viewer display, controls are placed one typical log-dose step left of
    the lowest non-zero concentration.
    """
    nonzero = np.asarray(nonzero_concentrations, dtype=float)
    nonzero = np.sort(np.unique(nonzero[np.isfinite(nonzero) & (nonzero > 0)]))
    if nonzero.size == 0:
        raise ValueError("PELSA requires at least one non-zero concentration.")

    log_nonzero = np.log10(nonzero)
    if log_nonzero.size == 1:
        step = 1.0
    else:
        diffs = np.diff(log_nonzero)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        step = float(np.median(diffs)) if diffs.size else 1.0

    return float(log_nonzero[0] - step)


def _build_pelsa_ratio_table(
    adata: ad.AnnData,
    *,
    concentration_col: str = "Concentration",
    layer: str = "normalized",
    control_concentration: float = 0.0,
) -> pd.DataFrame:
    """Build long peptide × sample log2FC table for PELSA curve fitting.

    Input matrix is expected on log2 scale. The fitted response is log2 fold
    change to the mean linear-scale control abundance per peptide. The
    ratio-to-control column is retained only for display/export convenience.
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

    conc_values = conc.to_numpy(dtype=float)
    control_log10 = _control_log10_position(conc_values[conc_values > 0])

    control_ref = np.apply_along_axis(_log2_mean, 0, X_log2[control_mask, :])
    log2_ratio = X_log2 - control_ref[None, :]

    ratio = np.exp2(log2_ratio)
    ratio[~np.isfinite(ratio)] = np.nan

    sample_names = np.asarray(adata.obs_names.astype(str))
    peptide_ids = np.asarray(adata.var_names.astype(str))

    sample_idx, peptide_idx = np.where(np.isfinite(log2_ratio))

    out = pd.DataFrame(
        {
            "peptide_id": peptide_ids[peptide_idx],
            "sample": sample_names[sample_idx],
            "concentration": conc_values[sample_idx],
            "log10_concentration": np.nan,
            "log10_concentration_raw": np.nan,
            "is_control": control_mask[sample_idx],
            "log2_intensity": X_log2[sample_idx, peptide_idx],
            "log2_ratio": log2_ratio[sample_idx, peptide_idx],
            "ratio": ratio[sample_idx, peptide_idx],
        }
    )

    nonzero = out["concentration"].to_numpy(dtype=float) > 0
    log10_nonzero = np.log10(out.loc[nonzero, "concentration"].to_numpy(dtype=float))
    out.loc[nonzero, "log10_concentration"] = log10_nonzero
    out.loc[nonzero, "log10_concentration_raw"] = log10_nonzero
    out.loc[out["is_control"].to_numpy(dtype=bool), "log10_concentration"] = control_log10

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


def _build_pelsa_localization_metadata(adata: ad.AnnData) -> dict:
    var = adata.var

    length_metric = None
    length_values = None

    for col, metric in (
        ("PROTEIN_LENGTH", "length"),
        ("Protein.Length", "length"),
        ("PROTEIN_WEIGHT", "weight"),
        ("PG.MolecularWeight", "weight"),
    ):
        if col in var.columns:
            length_metric = metric
            length_values = pd.to_numeric(var[col], errors="coerce").to_numpy(dtype=float)
            break

    peptide_start = None
    peptide_end = None
    for col in ("PEPTIDE_START", "EG.Start", "PEP.Start", "Peptide.Start"):
        if col in var.columns:
            peptide_start = pd.to_numeric(var[col], errors="coerce").to_numpy(dtype=float)
            break
    for col in ("PEPTIDE_END", "EG.End", "PEP.End", "Peptide.End"):
        if col in var.columns:
            peptide_end = pd.to_numeric(var[col], errors="coerce").to_numpy(dtype=float)
            break

    peptide_length = np.array([len(str(x)) for x in adata.var_names], dtype=int)

    return {
        "feature_ids": np.asarray(adata.var_names.astype(str)),
        "peptide_length": peptide_length,
        "peptide_start": peptide_start,
        "peptide_end": peptide_end,
        "protein_size_metric": length_metric or "",
        "protein_size_values": length_values,
    }

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
        "PELSA log2FC table ready: "
        f"{summary['n_peptides_with_any_ratio']} peptides, "
        f"{summary['n_ratio_points']} finite log2FC points, "
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
        "control_reference": "mean_linear_on_log2_input",
        "control_log10_concentration": float(
            ratio_df.loc[
                ratio_df["is_control"].to_numpy(dtype=bool),
                "log10_concentration",
            ].iloc[0]
        ),
        "fit_includes_control": True,
        "curve_model": {
            "name": "4pl_log_logistic",
            "x_scale": "log10_concentration_with_pseudo_control",
            "y_scale": "log2_ratio_to_control",
            "display_y_scale": "ratio_to_control",
            "formula": "back + (front - back) / (1 + 10 ** (slope * (x + pec50)))",
        },
        "concentrations": np.r_[control_concentration, nonzero_conc],
        "summary": summary,
        "localization": _build_pelsa_localization_metadata(adata),
        "curve_points": ratio_df,
        "curve_results": curve_results,
    }

    return adata
