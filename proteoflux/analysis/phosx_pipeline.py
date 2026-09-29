"""PhosX kinase-activity integration for phosphoproteomics contrasts."""

from __future__ import annotations

import re
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from proteoflux.analysis.kinase_sequences import (
    ACCESSION_SITES_COLUMN,
    TENMER_COLUMN,
    configured_kinase_method,
    kinase_activity_config,
    parse_accession_sites,
)
from proteoflux.analysis.phosx_adapter import (
    REQUIRED_PHOSX_VERSION,
    PhosXSession,
    create_phosx_session,
    run_phosx_in_memory,
    validate_phosx_installation,
    validate_phosx_resources,
)
from proteoflux.utils.kinases_mapping import (
    KINASE_IDENTIFIER_SOURCE,
    KINASE_IDENTIFIER_SOURCE_SHA256,
    PHOSX_KINASE_IDENTIFIERS,
)
from proteoflux.utils.utils import log_info, log_time, log_warning

_SEQUENCE_RE = re.compile(r"^[GPAVLIMCFYWHKRQNEDSTsty_]{10}$")

_RESULT_COLUMNS = [
    "contrast",
    "method",
    "kinase_id",
    "kinase_model_id",
    "kinase",
    "kinase_gene",
    "kinase_uniprot",
    "n_substrates",
    "kinase_mean_log2fc",
    "global_mean_log2fc",
    "global_sd_log2fc",
    "effect",
    "activity_score",
    "z_score",
    "pvalue",
    "qvalue",
    "tested",
    "reason",
    "ks",
    "legacy_activity_score",
]
_SUBSTRATE_COLUMNS = [
    "contrast",
    "kinase_id",
    "kinase_model_id",
    "kinase_gene",
    "kinase_uniprot",
    "phosphosite_id",
    "matched_accession",
    "database_source",
]
_CONDITION_KINASE_COLUMNS = [
    "condition",
    "kinase_id",
    "kinase_model_id",
    "kinase_gene",
    "kinase_uniprot",
    "n_substrates",
]


def _kinase_identifiers(model_id: Any) -> tuple[str, str, str]:
    """Return the PhosX model ID, canonical HGNC symbol, and UniProt accession."""
    model = str(model_id)
    canonical = PHOSX_KINASE_IDENTIFIERS.get(model)
    if canonical is None:
        raise RuntimeError(
            f"PhosX returned unmapped PSSM model identifier {model!r}. "
            "Update PHOSX_KINASE_IDENTIFIERS for custom PSSM resources; "
            "ProteoFlux deliberately performs no runtime network lookup."
        )
    gene, uniprot = canonical
    return model, gene, uniprot


def _integer(value: Any, name: str, default: int, minimum: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(
            f"analysis.kinase_activity.{name} must be an integer >= {minimum}."
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"analysis.kinase_activity.{name} must be an integer >= {minimum}."
        ) from exc
    if result < minimum:
        raise ValueError(f"analysis.kinase_activity.{name} must be >= {minimum}.")
    return result


def _probability(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"analysis.kinase_activity.{name} must be between 0 and 1."
        ) from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"analysis.kinase_activity.{name} must be between 0 and 1.")
    return result


def _parse_config(config: dict) -> dict[str, Any] | None:
    if configured_kinase_method(config) != "phosx":
        return None
    cfg = kinase_activity_config(config)

    min_substrates = cfg.get("min_substrates", cfg.get("min_n_hits"))
    resource_overrides = cfg.get("phosx_resources") or {}
    if not isinstance(resource_overrides, dict):
        raise TypeError("analysis.kinase_activity.phosx_resources must be a mapping.")

    parsed = {
        "n_permutations": _integer(
            cfg.get("n_permutations", cfg.get("n_perm")),
            "n_permutations",
            20000,
        ),
        "random_seed": _integer(
            cfg.get("random_seed", cfg.get("seed")), "random_seed", 1, 0
        ),
        "n_processes": _integer(
            cfg.get("n_processes", cfg.get("n_proc")), "n_processes", 1
        ),
        "ser_thr_only": bool(cfg.get("ser_thr_only", False)),
        "tyr_only": bool(cfg.get("tyr_only", False)),
        "s_t_n_top_kinases": _integer(
            cfg.get("s_t_n_top_kinases"), "s_t_n_top_kinases", 5
        ),
        "y_n_top_kinases": _integer(cfg.get("y_n_top_kinases"), "y_n_top_kinases", 5),
        "min_n_hits": _integer(min_substrates, "min_substrates", 4),
        "s_t_min_quantile": _probability(
            cfg.get("s_t_min_quantile"), "s_t_min_quantile", 0.95
        ),
        "y_min_quantile": _probability(
            cfg.get("y_min_quantile"), "y_min_quantile", 0.90
        ),
        "no_upstream_activation_evidence": bool(
            cfg.get("no_upstream_activation_evidence", False)
        ),
        "a_loop_s_t_quantile_threshold": _probability(
            cfg.get("a_loop_s_t_quantile_threshold"),
            "a_loop_s_t_quantile_threshold",
            0.95,
        ),
        "a_loop_y_quantile_threshold": _probability(
            cfg.get("a_loop_y_quantile_threshold"),
            "a_loop_y_quantile_threshold",
            0.95,
        ),
        "upreg_redundancy_threshold": _probability(
            cfg.get("upreg_redundancy_threshold"),
            "upreg_redundancy_threshold",
            0.5,
        ),
        "downreg_redundancy_threshold": _probability(
            cfg.get("downreg_redundancy_threshold"),
            "downreg_redundancy_threshold",
            0.5,
        ),
        "decay_factor": float(cfg.get("decay_factor", 256.0)),
        "resource_overrides": resource_overrides,
    }
    if parsed["ser_thr_only"] and parsed["tyr_only"]:
        raise ValueError(
            "analysis.kinase_activity.ser_thr_only and tyr_only cannot both be true."
        )
    if not np.isfinite(parsed["decay_factor"]) or parsed["decay_factor"] <= 0:
        raise ValueError("analysis.kinase_activity.decay_factor must be > 0.")
    return parsed


def validate_phosx_config(config: dict) -> None:
    parsed = _parse_config(config)
    if parsed is None:
        return
    validate_phosx_installation()
    validate_phosx_resources(parsed["resource_overrides"])
    log_info(
        "PhosX validation passed: "
        f"version={REQUIRED_PHOSX_VERSION}, "
        f"permutations={parsed['n_permutations']:,}, "
        f"seed={parsed['random_seed']}."
    )


def _select_log2fc(adata: ad.AnnData) -> tuple[np.ndarray, str, str]:
    if "log2fc" in adata.varm:
        matrix = np.asarray(adata.varm["log2fc"], dtype=float)
        source = (
            "flowthrough_adjusted_with_raw_fallback"
            if bool(adata.uns.get("has_covariate", False))
            else "raw"
        )
        key = "log2fc"
    elif "raw_log2fc" in adata.varm:
        matrix = np.asarray(adata.varm["raw_log2fc"], dtype=float)
        source = "raw"
        key = "raw_log2fc"
        log_warning(
            "PhosX could not find varm['log2fc']; falling back to varm['raw_log2fc']."
        )
    else:
        raise ValueError(
            "PhosX requires adata.varm['log2fc'] or adata.varm['raw_log2fc']."
        )
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix, source, key


def _fully_imputed_mask(adata: ad.AnnData, contrasts: list[str]) -> np.ndarray:
    result = np.zeros((adata.n_vars, len(contrasts)), dtype=bool)
    if "raw" not in adata.layers:
        log_warning(
            "PhosX cannot exclude fully imputed sites because "
            "adata.layers['raw'] is missing."
        )
        return result
    raw_layer = adata.layers["raw"]
    raw = (
        raw_layer.toarray() if hasattr(raw_layer, "toarray") else np.asarray(raw_layer)
    )
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"PhosX expected adata.layers['raw'] shape {(adata.n_obs, adata.n_vars)}, "
            f"received {raw.shape}."
        )
    conditions = adata.obs["CONDITION"].astype(str).to_numpy()
    for contrast_index, contrast in enumerate(contrasts):
        if "_vs_" not in contrast:
            raise ValueError(
                f"PhosX contrast {contrast!r} does not match '<A>_vs_<B>'."
            )
        condition_a, condition_b = contrast.split("_vs_", 1)
        mask_a = conditions == condition_a
        mask_b = conditions == condition_b
        if not mask_a.any() or not mask_b.any():
            raise ValueError(
                f"PhosX contrast {contrast!r} has no samples for one condition."
            )
        result[:, contrast_index] = np.all(np.isnan(raw[mask_a, :]), axis=0) & np.all(
            np.isnan(raw[mask_b, :]), axis=0
        )
    return result


def _empty_condition_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        columns=[
            "condition",
            "observed_phosphosites",
            "database_matched_sites",
            "matched_kinases",
            "eligible_kinases",
        ]
    )
    summary.index = pd.Index([], dtype=str, name="condition_id")
    memberships = pd.DataFrame(columns=_CONDITION_KINASE_COLUMNS)
    memberships.index = pd.Index([], dtype=str, name="condition_kinase_id")
    return summary, memberships


def _condition_kinase_data(
    adata: ad.AnnData,
    sequences: pd.Series,
    sequence_valid: np.ndarray,
    session: PhosXSession,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Viewer membership tables using sites observed in each condition."""
    if "raw" not in adata.layers:
        log_warning(
            "PhosX condition-level kinase counts are unavailable because "
            "adata.layers['raw'] is missing."
        )
        return _empty_condition_data()

    raw_layer = adata.layers["raw"]
    raw = (
        raw_layer.toarray() if hasattr(raw_layer, "toarray") else np.asarray(raw_layer)
    )
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            "adata.layers['raw'] must have shape "
            f"{(adata.n_obs, adata.n_vars)}, got {raw.shape}."
        )

    central_residues = sequences.str[5].to_numpy()
    considered = sequence_valid.copy()
    if parameters["ser_thr_only"]:
        considered &= np.isin(central_residues, ["S", "T"])
    elif parameters["tyr_only"]:
        considered &= central_residues == "Y"

    family_data: list[tuple[np.ndarray, pd.DataFrame]] = []
    if not parameters["tyr_only"]:
        indices = np.flatnonzero(considered & np.isin(central_residues, ["S", "T"]))
        assignments, _ = session.assignments(
            sequences.iloc[indices].tolist(),
            family_name="S/T",
            n_top_kinases=parameters["s_t_n_top_kinases"],
            min_quantile=parameters["s_t_min_quantile"],
        )
        family_data.append((indices, assignments))
    if not parameters["ser_thr_only"]:
        indices = np.flatnonzero(considered & (central_residues == "Y"))
        assignments, _ = session.assignments(
            sequences.iloc[indices].tolist(),
            family_name="Y",
            n_top_kinases=parameters["y_n_top_kinases"],
            min_quantile=parameters["y_min_quantile"],
        )
        family_data.append((indices, assignments))

    condition_values = adata.obs["CONDITION"].astype(str).to_numpy()
    conditions = list(dict.fromkeys(condition_values.tolist()))
    summary_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []

    for condition in conditions:
        sample_mask = condition_values == condition
        observed = considered & np.any(np.isfinite(raw[sample_mask, :]), axis=0)
        kinase_counts: dict[str, int] = {}
        matched_sites = 0

        for feature_indices, assignments in family_data:
            selected = observed[feature_indices]
            if not selected.any() or assignments.empty:
                continue
            selected_assignments = assignments.to_numpy(dtype=np.uint8, copy=False)[
                selected
            ]
            matched_sites += int(
                np.count_nonzero(np.any(selected_assignments != 0, axis=1))
            )
            counts = selected_assignments.sum(axis=0, dtype=np.int64)
            for kinase, count in zip(assignments.columns, counts):
                if count:
                    kinase_counts[str(kinase)] = kinase_counts.get(
                        str(kinase), 0
                    ) + int(count)

        eligible = {
            kinase: count
            for kinase, count in kinase_counts.items()
            if count >= parameters["min_n_hits"]
        }
        summary_rows.append(
            {
                "condition": condition,
                "observed_phosphosites": int(observed.sum()),
                # Keep the established KSEA schema name consumed by ProteoViewer.
                "database_matched_sites": matched_sites,
                "matched_kinases": len(kinase_counts),
                "eligible_kinases": len(eligible),
            }
        )
        for kinase, count in eligible.items():
            model_id, gene, uniprot = _kinase_identifiers(kinase)
            membership_rows.append(
                {
                    "condition": condition,
                    # kinase_id remains the stable, unique PSSM model key.  Use the
                    # canonical fields for joins to other kinase-analysis methods.
                    "kinase_id": model_id,
                    "kinase_model_id": model_id,
                    "kinase_gene": gene,
                    "kinase_uniprot": uniprot,
                    "n_substrates": count,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.index = pd.Index(summary["condition"].astype(str), name="condition_id")
    memberships = pd.DataFrame(membership_rows, columns=_CONDITION_KINASE_COLUMNS)
    if memberships.empty:
        memberships.index = pd.Index([], dtype=str, name="condition_kinase_id")
    else:
        memberships.index = pd.Index(
            memberships["condition"].astype(str)
            + "|"
            + memberships["kinase_id"].astype(str),
            name="condition_kinase_id",
        )
    return summary, memberships


def _ranked_family(input_frame: pd.DataFrame, family: str) -> pd.DataFrame:
    ranked = input_frame.loc[
        input_frame["Score"].ne(0)
        & input_frame[["Sequence", "Score"]].notna().all(axis=1)
    ].copy()
    if family == "st":
        ranked = ranked.loc[ranked["Sequence"].str[5].isin(["S", "T"])]
    else:
        ranked = ranked.loc[ranked["Sequence"].str[5].eq("Y")]
    # Mirror PhosX 0.23.0 exactly so assignment row numbers map back to features.
    ranked.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)
    return ranked


def _accessions(compact: str) -> str:
    return ";".join(
        dict.fromkeys(accession for accession, _, _ in parse_accession_sites(compact))
    )


def _assignment_rows(
    assignments: pd.DataFrame,
    ranked: pd.DataFrame,
    contrast: str,
) -> list[dict[str, str]]:
    if assignments.empty:
        return []
    if len(assignments) != len(ranked):
        raise RuntimeError(
            "PhosX assignment rows cannot be aligned to the ranked phosphosites: "
            f"assignments={len(assignments)}, ranked={len(ranked)}."
        )
    rows: list[dict[str, str]] = []
    for kinase in assignments.columns:
        model_id, gene, uniprot = _kinase_identifiers(kinase)
        hit_positions = np.flatnonzero(
            pd.to_numeric(assignments[kinase], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
            > 0
        )
        for position in hit_positions:
            item = ranked.iloc[int(position)]
            rows.append(
                {
                    "contrast": contrast,
                    "kinase_id": model_id,
                    "kinase_model_id": model_id,
                    "kinase_gene": gene,
                    "kinase_uniprot": uniprot,
                    "phosphosite_id": str(item["phosphosite_id"]),
                    "matched_accession": _accessions(str(item[ACCESSION_SITES_COLUMN])),
                    "database_source": f"PhosX {REQUIRED_PHOSX_VERSION} bundled PSSM",
                }
            )
    return rows


def _result_rows(
    activity: pd.DataFrame,
    substrate_counts: pd.Series,
    substrate_means: pd.Series,
    contrast: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kinase, values in activity.iterrows():
        model_id, gene, uniprot = _kinase_identifiers(kinase)
        pvalue = pd.to_numeric(
            pd.Series([values.get("p value")]), errors="coerce"
        ).iloc[0]
        qvalue = pd.to_numeric(
            pd.Series([values.get("FDR q value")]), errors="coerce"
        ).iloc[0]
        ks = pd.to_numeric(pd.Series([values.get("KS")]), errors="coerce").iloc[0]
        activity_score = pd.to_numeric(
            pd.Series([values.get("Activity Score")]), errors="coerce"
        ).iloc[0]
        legacy = pd.to_numeric(
            pd.Series([values.get("Legacy Activity Score")]), errors="coerce"
        ).iloc[0]
        kinase_mean = pd.to_numeric(
            pd.Series([substrate_means.get(model_id)]), errors="coerce"
        ).iloc[0]
        tested = bool(np.isfinite(pvalue))
        rows.append(
            {
                "contrast": contrast,
                "method": "phosx",
                "kinase_id": model_id,
                "kinase_model_id": model_id,
                "kinase": gene,
                "kinase_gene": gene,
                "kinase_uniprot": uniprot,
                "n_substrates": int(substrate_counts.get(model_id, 0)),
                "kinase_mean_log2fc": (
                    float(kinase_mean) if np.isfinite(kinase_mean) else np.nan
                ),
                "global_mean_log2fc": np.nan,
                "global_sd_log2fc": np.nan,
                "effect": float(ks) if np.isfinite(ks) else np.nan,
                "activity_score": (
                    float(activity_score) if np.isfinite(activity_score) else np.nan
                ),
                "z_score": np.nan,
                "pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "qvalue": float(qvalue) if np.isfinite(qvalue) else np.nan,
                "tested": tested,
                "reason": "" if tested else "insufficient_substrates",
                "ks": float(ks) if np.isfinite(ks) else np.nan,
                "legacy_activity_score": (
                    float(legacy) if np.isfinite(legacy) else np.nan
                ),
            }
        )
    return rows


@log_time("PhosX kinase activity analysis")
def run_phosx_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    parsed = _parse_config(config)
    if parsed is None:
        return adata
    validate_phosx_installation()

    analysis_type = str(
        (adata.uns.get("preprocessing", {}) or {}).get("analysis_type", "phospho")
    ).lower()
    if analysis_type != "phospho":
        raise ValueError("PhosX is only supported for analysis_type='phospho'.")
    if TENMER_COLUMN not in adata.var.columns:
        raise ValueError(
            "PhosX requires prepared kinase sequence windows in "
            "adata.var['KINASE_10MER']."
        )

    contrasts = [str(value) for value in adata.uns.get("contrast_names", [])]
    if not contrasts:
        log_warning(
            "PhosX skipped because no differential-analysis contrasts are available."
        )
        return adata
    log2fc, log2fc_source, log2fc_key = _select_log2fc(adata)
    if log2fc.shape != (adata.n_vars, len(contrasts)):
        raise ValueError(
            f"PhosX expected {log2fc_key!r} shape {(adata.n_vars, len(contrasts))}, "
            f"received {log2fc.shape}."
        )

    sequences = adata.var[TENMER_COLUMN].astype("string").fillna("").astype(str)
    sequence_valid = sequences.map(
        lambda value: (
            bool(_SEQUENCE_RE.fullmatch(value)) and value[5] in {"S", "T", "Y"}
        )
    ).to_numpy(dtype=bool)
    compact_sites = (
        adata.var[ACCESSION_SITES_COLUMN]
        .astype("string")
        .fillna("")
        .astype(str)
        .to_numpy()
        if ACCESSION_SITES_COLUMN in adata.var.columns
        else np.repeat("", adata.n_vars)
    )
    fully_imputed = _fully_imputed_mask(adata, contrasts)
    phosx_session = create_phosx_session(parsed["resource_overrides"])
    condition_summary, condition_kinases = _condition_kinase_data(
        adata,
        sequences,
        sequence_valid,
        phosx_session,
        parsed,
    )

    all_results: list[dict[str, Any]] = []
    all_substrates: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    contrast_progress = tqdm(
        enumerate(contrasts),
        total=len(contrasts),
        desc=" PhosX contrasts",
        ncols=80,
        file=sys.stdout,
    )
    for contrast_index, contrast in contrast_progress:
        scores = log2fc[:, contrast_index]
        finite = np.isfinite(scores)
        eligible = sequence_valid & finite & ~fully_imputed[:, contrast_index]
        input_frame = pd.DataFrame(
            {
                "Sequence": sequences.to_numpy()[eligible],
                "Score": scores[eligible],
                "phosphosite_id": adata.var_names.astype(str).to_numpy()[eligible],
                ACCESSION_SITES_COLUMN: compact_sites[eligible],
            }
        )
        input_frame = input_frame.loc[input_frame["Score"].ne(0)].reset_index(drop=True)
        if input_frame.empty:
            raise ValueError(
                f"PhosX contrast {contrast!r} has no finite, non-zero, "
                "sequence-eligible sites."
            )
        duplicate_sites = input_frame["phosphosite_id"].duplicated(keep=False)
        if duplicate_sites.any():
            examples = input_frame.loc[duplicate_sites, "phosphosite_id"].unique()[:5]
            raise ValueError(
                "PhosX requires unique adata.var_names so assigned-site means are "
                f"unambiguous; duplicate examples: {examples.tolist()}."
            )

        contrast_seed = int(
            np.random.SeedSequence(
                [parsed["random_seed"], contrast_index]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        run = run_phosx_in_memory(
            input_frame[["Sequence", "Score"]],
            n_permutations=parsed["n_permutations"],
            random_seed=contrast_seed,
            n_processes=parsed["n_processes"],
            ser_thr_only=parsed["ser_thr_only"],
            tyr_only=parsed["tyr_only"],
            s_t_n_top_kinases=parsed["s_t_n_top_kinases"],
            y_n_top_kinases=parsed["y_n_top_kinases"],
            min_n_hits=parsed["min_n_hits"],
            s_t_min_quantile=parsed["s_t_min_quantile"],
            y_min_quantile=parsed["y_min_quantile"],
            no_upstream_activation_evidence=parsed["no_upstream_activation_evidence"],
            a_loop_s_t_quantile_threshold=parsed["a_loop_s_t_quantile_threshold"],
            a_loop_y_quantile_threshold=parsed["a_loop_y_quantile_threshold"],
            upreg_redundancy_threshold=parsed["upreg_redundancy_threshold"],
            downreg_redundancy_threshold=parsed["downreg_redundancy_threshold"],
            decay_factor=parsed["decay_factor"],
            resource_overrides=parsed["resource_overrides"],
            session=phosx_session,
        )

        ranked_st = (
            input_frame.iloc[0:0].copy()
            if parsed["tyr_only"]
            else _ranked_family(input_frame, "st")
        )
        ranked_y = (
            input_frame.iloc[0:0].copy()
            if parsed["ser_thr_only"]
            else _ranked_family(input_frame, "y")
        )
        substrate_rows = _assignment_rows(
            run.ser_thr_assignments, ranked_st, contrast
        ) + _assignment_rows(run.tyr_assignments, ranked_y, contrast)
        # A kinase/site is counted once even if future PSSM families overlap.
        substrate_frame = pd.DataFrame(substrate_rows, columns=_SUBSTRATE_COLUMNS)
        if not substrate_frame.empty:
            substrate_frame = substrate_frame.drop_duplicates(
                ["contrast", "kinase_id", "phosphosite_id"], ignore_index=True
            )
            substrate_counts = substrate_frame.groupby("kinase_id")[
                "phosphosite_id"
            ].nunique()
            score_by_site = input_frame.set_index("phosphosite_id")["Score"]
            assigned_scores = substrate_frame["phosphosite_id"].map(score_by_site)
            if assigned_scores.isna().any():
                raise RuntimeError(
                    "PhosX substrate assignments could not be mapped back to all "
                    "contrast log2FC values."
                )
            substrate_means = assigned_scores.groupby(
                substrate_frame["kinase_id"], sort=False
            ).mean()
            all_substrates.extend(substrate_frame.to_dict("records"))
        else:
            substrate_counts = pd.Series(dtype=int)
            substrate_means = pd.Series(dtype=float)

        all_results.extend(
            _result_rows(run.activity, substrate_counts, substrate_means, contrast)
        )
        summaries.append(
            {
                "contrast": contrast,
                "valid_sequence_sites": int(sequence_valid.sum()),
                "nonfinite_removed": int(np.count_nonzero(sequence_valid & ~finite)),
                "fully_imputed_removed": int(
                    np.count_nonzero(
                        sequence_valid & finite & fully_imputed[:, contrast_index]
                    )
                ),
                "zero_log2fc_removed": int(np.count_nonzero(eligible & (scores == 0))),
                "input_sites": len(input_frame),
                "ser_thr_sites": len(ranked_st),
                "tyr_sites": len(ranked_y),
                "assigned_sites": int(
                    substrate_frame["phosphosite_id"].nunique()
                    if not substrate_frame.empty
                    else 0
                ),
                "tested_kinases": int(run.activity["p value"].notna().sum()),
            }
        )
        contrast_progress.set_postfix_str(
            f"{contrast}: {summaries[-1]['tested_kinases']:,} kinases"
        )

    results = pd.DataFrame(all_results, columns=_RESULT_COLUMNS)
    substrates = pd.DataFrame(all_substrates, columns=_SUBSTRATE_COLUMNS)
    contrast_summary = pd.DataFrame(summaries)

    if not results.empty:
        results.index = pd.Index(
            results["contrast"].astype(str) + "|" + results["kinase_id"].astype(str),
            name="kinase_result_id",
        )
    else:
        results.index = pd.Index([], dtype=str, name="kinase_result_id")
    if not substrates.empty:
        substrates.index = pd.Index(
            substrates["contrast"].astype(str)
            + "|"
            + substrates["kinase_id"].astype(str)
            + "|"
            + substrates["phosphosite_id"].astype(str),
            name="kinase_substrate_id",
        )
    else:
        substrates.index = pd.Index([], dtype=str, name="kinase_substrate_id")
    contrast_summary.index = pd.Index(
        contrast_summary["contrast"].astype(str), name="contrast_id"
    )

    output = adata.copy()
    output.uns["kinase_activity"] = {
        "schema_version": 1,
        "method": "phosx",
        "phosx_version": REQUIRED_PHOSX_VERSION,
        "kinase_identifier_source": KINASE_IDENTIFIER_SOURCE,
        "kinase_identifier_source_sha256": KINASE_IDENTIFIER_SOURCE_SHA256,
        "kinase_id_semantics": (
            "PhosX PSSM model identifier; use kinase_uniprot (preferred) or "
            "kinase_gene for cross-method joins"
        ),
        "min_substrates": parsed["min_n_hits"],
        "log2fc_varm_key": log2fc_key,
        "log2fc_source": log2fc_source,
        "background_rule": (
            "valid 10-mer, finite non-zero contrast log2FC; fully imputed "
            "contrast cells excluded"
        ),
        "pvalue_rule": "PhosX one-sided empirical KS permutation p value",
        "fdr_scope": "PhosX BH within each contrast and kinase family",
        "kinase_mean_log2fc_rule": (
            "arithmetic mean contrast log2FC across unique quantified "
            "phosphosites assigned to the PhosX PSSM model"
        ),
        "parameters": {
            "n_permutations": parsed["n_permutations"],
            "random_seed": parsed["random_seed"],
            "n_processes": parsed["n_processes"],
            "s_t_n_top_kinases": parsed["s_t_n_top_kinases"],
            "y_n_top_kinases": parsed["y_n_top_kinases"],
            "s_t_min_quantile": parsed["s_t_min_quantile"],
            "y_min_quantile": parsed["y_min_quantile"],
            "upstream_activation_evidence": not parsed[
                "no_upstream_activation_evidence"
            ],
        },
        "condition_summary": condition_summary,
        "condition_kinases": condition_kinases,
        "contrast_summary": contrast_summary,
        "results": results,
        "substrates": substrates,
    }
    return output
