"""Kinase-substrate enrichment analysis for phosphoproteomics contrasts.

The implementation follows the canonical KSEA z-score.  Kinase-substrate
relationships come from a local, versioned table; no network service is used.
All statistics and multiple-testing correction are computed independently for
each contrast.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import norm

from proteoflux.analysis.stats_ops import bh_qvalues
from proteoflux.utils.utils import log_info, log_time, log_warning


# Centralized here so a future database format needs changes in one place only.
DB_KINASE_NAME = "KINASE"
DB_KINASE_ACCESSION = "KIN_ACC_ID"
DB_KINASE_GENE = "GENE"
DB_KINASE_ORGANISM = "KIN_ORGANISM"
DB_SUBSTRATE_ACCESSION = "SUB_ACC_ID"
DB_SUBSTRATE_ORGANISM = "SUB_ORGANISM"
DB_SUBSTRATE_SITE = "SUB_MOD_RSD"
DB_SOURCE = "Source"

_REQUIRED_DATABASE_COLUMNS = {
    DB_KINASE_NAME,
    DB_KINASE_ORGANISM,
    DB_SUBSTRATE_ACCESSION,
    DB_SUBSTRATE_ORGANISM,
    DB_SUBSTRATE_SITE,
}
_SITE_RE = re.compile(r"^[STY]\d+$", flags=re.IGNORECASE)

_RESULT_COLUMNS = [
    "contrast",
    "method",
    "kinase_id",
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
]

_SUBSTRATE_COLUMNS = [
    "contrast",
    "kinase_id",
    "phosphosite_id",
    "matched_accession",
    "database_source",
]


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().astype(object)


def _first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def _join_unique(values: pd.Series) -> str:
    unique = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(unique)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_config(config: dict) -> dict[str, Any] | None:
    analysis_cfg = (config or {}).get("analysis", {}) or {}
    kinase_cfg = analysis_cfg.get("kinase_activity") or {}
    method = kinase_cfg.get("method")

    if method is None or str(method).strip() == "":
        return None

    method = str(method).strip().lower()
    if method != "ksea":
        raise ValueError(
            "analysis.kinase_activity.method currently supports only 'ksea'; "
            f"received {method!r}."
        )

    organism = str(kinase_cfg.get("organism") or "").strip()
    if not organism:
        raise ValueError("analysis.kinase_activity.organism must not be null or empty.")

    database_value = kinase_cfg.get("substrate_database")
    if database_value is None or str(database_value).strip() == "":
        raise ValueError(
            "analysis.kinase_activity.substrate_database must point to a local "
            "kinase-substrate TSV file."
        )

    min_substrates = kinase_cfg.get("min_substrates", 5)
    if isinstance(min_substrates, bool):
        raise ValueError("analysis.kinase_activity.min_substrates must be an integer >= 1.")
    if isinstance(min_substrates, float) and not min_substrates.is_integer():
        raise ValueError("analysis.kinase_activity.min_substrates must be an integer >= 1.")
    try:
        min_substrates = int(min_substrates)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "analysis.kinase_activity.min_substrates must be an integer >= 1."
        ) from exc
    if min_substrates < 1:
        raise ValueError("analysis.kinase_activity.min_substrates must be >= 1.")

    return {
        "method": method,
        "organism": organism,
        "substrate_database": Path(str(database_value)).expanduser(),
        "min_substrates": min_substrates,
    }


def _load_substrate_database(
    database_path: Path,
    organism: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not database_path.is_file():
        raise FileNotFoundError(
            f"KSEA substrate database does not exist or is not a file: {database_path}"
        )

    database = pd.read_csv(
        database_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    database.columns = [str(column).strip() for column in database.columns]
    duplicated_columns = (
        pd.Index(database.columns)[pd.Index(database.columns).duplicated()]
        .unique()
        .tolist()
    )
    if duplicated_columns:
        raise ValueError(
            f"KSEA substrate database contains duplicate columns: {duplicated_columns!r}."
        )

    missing_columns = sorted(_REQUIRED_DATABASE_COLUMNS.difference(database.columns))
    if missing_columns:
        raise ValueError(
            "KSEA substrate database is missing required columns: "
            f"{missing_columns!r}. Available columns: {list(database.columns)!r}."
        )

    loaded_rows = int(len(database))
    organism_key = organism.casefold()
    kinase_organism = _clean_text(database[DB_KINASE_ORGANISM]).str.casefold()
    substrate_organism = _clean_text(database[DB_SUBSTRATE_ORGANISM]).str.casefold()
    database = database.loc[
        (kinase_organism == organism_key) & (substrate_organism == organism_key)
    ].copy()
    organism_rows = int(len(database))

    if database.empty:
        raise ValueError(
            f"KSEA substrate database contains no kinase-substrate rows for "
            f"organism {organism!r}."
        )

    database["kinase"] = _clean_text(database[DB_KINASE_NAME])
    database["kinase_uniprot"] = (
        _clean_text(database[DB_KINASE_ACCESSION])
        if DB_KINASE_ACCESSION in database.columns
        else ""
    )
    database["kinase_gene"] = (
        _clean_text(database[DB_KINASE_GENE])
        if DB_KINASE_GENE in database.columns
        else ""
    )
    database["substrate_accession"] = (
        _clean_text(database[DB_SUBSTRATE_ACCESSION]).str.upper()
    )
    database["site"] = _clean_text(database[DB_SUBSTRATE_SITE]).str.upper()
    database["database_source"] = (
        _clean_text(database[DB_SOURCE]) if DB_SOURCE in database.columns else ""
    )

    database["kinase_id"] = database["kinase_uniprot"]
    missing_id = database["kinase_id"].eq("")
    database.loc[missing_id, "kinase_id"] = database.loc[missing_id, "kinase_gene"]
    missing_id = database["kinase_id"].eq("")
    database.loc[missing_id, "kinase_id"] = database.loc[missing_id, "kinase"]

    valid = (
        database["kinase_id"].ne("")
        & database["substrate_accession"].ne("")
        & database["site"].str.fullmatch(_SITE_RE)
    )
    invalid_rows = int((~valid).sum())
    database = database.loc[valid].copy()
    if database.empty:
        raise ValueError(
            f"KSEA substrate database has no valid {organism!r} relationships after "
            "requiring a kinase identifier, UniProt substrate accession, and S/T/Y site."
        )

    source_counts = database.loc[
        database["database_source"].ne(""), "database_source"
    ].value_counts()

    relationships = (
        database.groupby(
            ["kinase_id", "substrate_accession", "site"],
            as_index=False,
            sort=False,
        )
        .agg(
            kinase=("kinase", _first_nonempty),
            kinase_gene=("kinase_gene", _first_nonempty),
            kinase_uniprot=("kinase_uniprot", _first_nonempty),
            database_source=("database_source", _join_unique),
        )
    )

    duplicate_relationships = int(len(database) - len(relationships))
    metadata = {
        "filename": database_path.name,
        "sha256": _sha256(database_path),
        "loaded_rows": loaded_rows,
        "organism_rows": organism_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_relationships_removed": duplicate_relationships,
        "unique_relationships": int(len(relationships)),
        "source_names": source_counts.index.astype(str).tolist(),
        "source_rows": source_counts.astype(int).tolist(),
    }

    log_info(
        "KSEA substrate database loaded: "
        f"file={database_path.name!r}, rows={loaded_rows}, organism={organism!r}, "
        f"organism_rows={organism_rows}, invalid_removed={invalid_rows}, "
        f"duplicate_relationships_removed={duplicate_relationships}, "
        f"unique_relationships={len(relationships)}."
    )
    return relationships, metadata


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _build_feature_site_table(adata: ad.AnnData) -> tuple[pd.DataFrame, int]:
    parent_column = next(
        (column for column in ("PARENT_PROTEIN", "UNIPROT") if column in adata.var.columns),
        None,
    )

    rows: list[dict[str, Any]] = []
    invalid_features = 0
    for feature_index, feature_name in enumerate(adata.var_names.astype(str)):
        phosphosite_id = str(feature_name)
        if "|" not in phosphosite_id:
            invalid_features += 1
            continue

        index_parent, site = phosphosite_id.rsplit("|", 1)
        site = site.strip().upper()
        if _SITE_RE.fullmatch(site) is None:
            invalid_features += 1
            continue

        parent_text = ""
        if parent_column is not None:
            parent_text = _scalar_text(adata.var.iloc[feature_index][parent_column])
        if not parent_text:
            parent_text = index_parent.strip()

        accessions = sorted(
            {
                accession.strip().upper()
                for accession in parent_text.split(";")
                if accession.strip()
            }
        )
        if not accessions:
            invalid_features += 1
            continue

        for accession in accessions:
            rows.append(
                {
                    "feature_index": feature_index,
                    "phosphosite_id": phosphosite_id,
                    "substrate_accession": accession,
                    "site": site,
                }
            )

    feature_sites = pd.DataFrame(
        rows,
        columns=[
            "feature_index",
            "phosphosite_id",
            "substrate_accession",
            "site",
        ],
    )
    return feature_sites, invalid_features


def _match_substrates(
    feature_sites: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    if feature_sites.empty:
        return pd.DataFrame(
            columns=[
                "kinase_id",
                "kinase",
                "kinase_gene",
                "kinase_uniprot",
                "feature_index",
                "phosphosite_id",
                "matched_accession",
                "database_source",
            ]
        )

    matched = feature_sites.merge(
        relationships,
        on=["substrate_accession", "site"],
        how="inner",
        validate="many_to_many",
    )
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "kinase_id",
                "kinase",
                "kinase_gene",
                "kinase_uniprot",
                "feature_index",
                "phosphosite_id",
                "matched_accession",
                "database_source",
            ]
        )

    # A phosphosite from a protein group may match more than one accession for
    # the same kinase.  It still counts once in n_substrates.
    matched = (
        matched.groupby(
            ["kinase_id", "feature_index", "phosphosite_id"],
            as_index=False,
            sort=False,
        )
        .agg(
            kinase=("kinase", _first_nonempty),
            kinase_gene=("kinase_gene", _first_nonempty),
            kinase_uniprot=("kinase_uniprot", _first_nonempty),
            matched_accession=("substrate_accession", _join_unique),
            database_source=("database_source", _join_unique),
        )
    )
    return matched


def _select_log2fc(adata: ad.AnnData) -> tuple[np.ndarray, str, str]:
    has_covariate = bool(adata.uns.get("has_covariate", False))

    if "log2fc" in adata.varm:
        matrix = np.asarray(adata.varm["log2fc"], dtype=float)
        source = (
            "flowthrough_adjusted_with_raw_fallback"
            if has_covariate
            else "raw"
        )
        key = "log2fc"
    elif "raw_log2fc" in adata.varm:
        matrix = np.asarray(adata.varm["raw_log2fc"], dtype=float)
        source = "raw"
        key = "raw_log2fc"
        log_warning(
            "KSEA could not find varm['log2fc']; falling back to varm['raw_log2fc']."
        )
    else:
        raise ValueError(
            "KSEA requires differential fold changes in adata.varm['log2fc'] "
            "or adata.varm['raw_log2fc']."
        )

    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix, source, key


def _fully_imputed_mask(
    adata: ad.AnnData,
    contrast_names: list[str],
) -> np.ndarray:
    fully = np.zeros((adata.n_vars, len(contrast_names)), dtype=bool)
    if "raw" not in adata.layers:
        log_warning(
            "KSEA cannot exclude fully imputed phosphosites because adata.layers['raw'] "
            "is unavailable."
        )
        return fully

    raw_layer = adata.layers["raw"]
    raw = raw_layer.toarray() if hasattr(raw_layer, "toarray") else np.asarray(raw_layer)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"KSEA expected adata.layers['raw'] shape {(adata.n_obs, adata.n_vars)}, "
            f"received {raw.shape}."
        )

    conditions = adata.obs["CONDITION"].astype(str).to_numpy()
    raw_t = raw.T
    for contrast_index, contrast in enumerate(contrast_names):
        if "_vs_" not in contrast:
            raise ValueError(
                f"KSEA contrast {contrast!r} does not match expected '<A>_vs_<B>'."
            )
        condition_a, condition_b = contrast.split("_vs_", 1)
        mask_a = conditions == condition_a
        mask_b = conditions == condition_b
        if not mask_a.any() or not mask_b.any():
            raise ValueError(
                f"KSEA contrast {contrast!r} has no samples for {condition_a!r} "
                f"or {condition_b!r}."
            )
        fully[:, contrast_index] = (
            np.all(np.isnan(raw_t[:, mask_a]), axis=1)
            & np.all(np.isnan(raw_t[:, mask_b]), axis=1)
        )
    return fully


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=_RESULT_COLUMNS)


def _empty_substrates() -> pd.DataFrame:
    return pd.DataFrame(columns=_SUBSTRATE_COLUMNS)


def _compute_contrast(
    *,
    contrast: str,
    log2fc: np.ndarray,
    valid_feature_mask: np.ndarray,
    fully_imputed: np.ndarray,
    assignments: pd.DataFrame,
    min_substrates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    finite = np.isfinite(log2fc)
    nonfinite_removed = int(np.count_nonzero(valid_feature_mask & ~finite))
    fully_imputed_removed = int(
        np.count_nonzero(valid_feature_mask & finite & fully_imputed)
    )
    background_mask = valid_feature_mask & finite & ~fully_imputed
    background = log2fc[background_mask]

    global_mean = float(np.mean(background)) if background.size else np.nan
    global_sd = float(np.std(background, ddof=1)) if background.size >= 2 else np.nan

    if assignments.empty:
        matched = assignments.copy()
    else:
        assignment_indices = assignments["feature_index"].to_numpy(dtype=int)
        matched = assignments.loc[background_mask[assignment_indices]].copy()
        matched["_log2fc"] = log2fc[
            matched["feature_index"].to_numpy(dtype=int)
        ]

    result_rows: list[dict[str, Any]] = []
    if not matched.empty:
        for kinase_id, kinase_sites in matched.groupby("kinase_id", sort=False):
            n_substrates = int(kinase_sites["feature_index"].nunique())
            kinase_mean = float(kinase_sites["_log2fc"].mean())
            effect = kinase_mean - global_mean

            if n_substrates < min_substrates:
                tested = False
                reason = "insufficient_substrates"
                z_score = np.nan
                pvalue = np.nan
            elif not np.isfinite(global_sd) or global_sd <= 0.0:
                tested = False
                reason = "invalid_background_variance"
                z_score = np.nan
                pvalue = np.nan
            else:
                tested = True
                reason = ""
                z_score = float(effect * np.sqrt(n_substrates) / global_sd)
                pvalue = float(2.0 * norm.sf(abs(z_score)))

            first = kinase_sites.iloc[0]
            result_rows.append(
                {
                    "contrast": contrast,
                    "method": "ksea",
                    "kinase_id": str(kinase_id),
                    "kinase": str(first["kinase"]),
                    "kinase_gene": str(first["kinase_gene"]),
                    "kinase_uniprot": str(first["kinase_uniprot"]),
                    "n_substrates": n_substrates,
                    "kinase_mean_log2fc": kinase_mean,
                    "global_mean_log2fc": global_mean,
                    "global_sd_log2fc": global_sd,
                    "effect": effect,
                    "activity_score": z_score,
                    "z_score": z_score,
                    "pvalue": pvalue,
                    "qvalue": np.nan,
                    "tested": tested,
                    "reason": reason,
                }
            )

    results = pd.DataFrame(result_rows, columns=_RESULT_COLUMNS)
    if not results.empty:
        tested_mask = results["tested"].to_numpy(dtype=bool)
        if tested_mask.any():
            tested_pvalues = results.loc[tested_mask, "pvalue"].to_numpy(dtype=float)
            tested_qvalues = bh_qvalues(tested_pvalues[:, None])[:, 0]
            results.loc[tested_mask, "qvalue"] = tested_qvalues

    if matched.empty:
        substrates = _empty_substrates()
    else:
        substrates = matched[
            [
                "kinase_id",
                "phosphosite_id",
                "matched_accession",
                "database_source",
            ]
        ].copy()
        substrates.insert(0, "contrast", contrast)
        substrates = substrates[_SUBSTRATE_COLUMNS]

    matched_sites = int(matched["feature_index"].nunique()) if not matched.empty else 0
    matched_kinases = int(matched["kinase_id"].nunique()) if not matched.empty else 0
    tested_kinases = int(results["tested"].sum()) if not results.empty else 0
    below_minimum_kinases = (
        int(results["reason"].eq("insufficient_substrates").sum())
        if not results.empty
        else 0
    )
    summary = {
        "contrast": contrast,
        "valid_phosphosites": int(np.count_nonzero(valid_feature_mask)),
        "nonfinite_removed": nonfinite_removed,
        "fully_imputed_removed": fully_imputed_removed,
        "background_sites": int(background.size),
        "database_matched_sites": matched_sites,
        "database_unmatched_sites": int(background.size - matched_sites),
        "matched_kinases": matched_kinases,
        "tested_kinases": tested_kinases,
        "below_minimum_kinases": below_minimum_kinases,
        "untested_kinases": int(matched_kinases - tested_kinases),
        "global_mean_log2fc": global_mean,
        "global_sd_log2fc": global_sd,
    }
    return results, substrates, summary


def _set_string_index(frame: pd.DataFrame, values: pd.Series, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.Index(values.astype(str), name=name)
    return frame


@log_time("Kinase activity analysis")
def run_ksea_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Run local KSEA after differential analysis and store compact Viewer data."""
    parsed_config = _parse_config(config)
    if parsed_config is None:
        return adata

    analysis_type = str(
        (adata.uns.get("preprocessing", {}) or {}).get("analysis_type", "phospho")
    ).lower()
    if analysis_type != "phospho":
        raise ValueError(
            "analysis.kinase_activity is only supported for analysis_type='phospho'; "
            f"received {analysis_type!r}."
        )

    contrast_names = [str(name) for name in adata.uns.get("contrast_names", [])]
    if not contrast_names:
        log_warning("KSEA skipped because no differential-analysis contrasts are available.")
        return adata

    log2fc, log2fc_source, log2fc_key = _select_log2fc(adata)
    expected_shape = (adata.n_vars, len(contrast_names))
    if log2fc.shape != expected_shape:
        raise ValueError(
            f"KSEA expected {log2fc_key!r} shape {expected_shape}, received "
            f"{log2fc.shape}."
        )
    log_info(
        "KSEA fold-change input: "
        f"varm={log2fc_key!r}, source={log2fc_source}, "
        f"features={adata.n_vars}, contrasts={len(contrast_names)}."
    )

    relationships, database_metadata = _load_substrate_database(
        parsed_config["substrate_database"],
        parsed_config["organism"],
    )
    feature_sites, invalid_features = _build_feature_site_table(adata)
    if feature_sites.empty:
        raise ValueError(
            "KSEA could not parse any phosphosite identifiers. Expected feature IDs "
            "such as 'P05198|S52'."
        )

    unique_feature_sites = feature_sites.drop_duplicates("feature_index")
    valid_feature_mask = np.zeros(adata.n_vars, dtype=bool)
    valid_feature_mask[
        unique_feature_sites["feature_index"].to_numpy(dtype=int)
    ] = True
    assignments = _match_substrates(feature_sites, relationships)

    log_info(
        "KSEA phosphosite matching: "
        f"features={adata.n_vars}, valid_site_ids={len(unique_feature_sites)}, "
        f"invalid_site_ids_removed={invalid_features}, "
        f"database_matched_sites={assignments['feature_index'].nunique() if not assignments.empty else 0}."
    )

    fully_imputed = _fully_imputed_mask(adata, contrast_names)
    result_frames: list[pd.DataFrame] = []
    substrate_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for contrast_index, contrast in enumerate(contrast_names):
        results, substrates, summary = _compute_contrast(
            contrast=contrast,
            log2fc=log2fc[:, contrast_index],
            valid_feature_mask=valid_feature_mask,
            fully_imputed=fully_imputed[:, contrast_index],
            assignments=assignments,
            min_substrates=parsed_config["min_substrates"],
        )
        result_frames.append(results)
        substrate_frames.append(substrates)
        summaries.append(summary)

        log_info(
            f"KSEA contrast {contrast!r}: "
            f"valid_sites={summary['valid_phosphosites']}, "
            f"nonfinite_removed={summary['nonfinite_removed']}, "
            f"fully_imputed_removed={summary['fully_imputed_removed']}, "
            f"background={summary['background_sites']}, "
            f"database_matched={summary['database_matched_sites']}, "
            f"database_unmatched={summary['database_unmatched_sites']}, "
            f"kinases={summary['matched_kinases']}, "
            f"tested={summary['tested_kinases']}, "
            f"below_minimum={summary['below_minimum_kinases']}, "
            f"not_tested={summary['untested_kinases']}."
        )

    results_df = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else _empty_results()
    )
    substrates_df = (
        pd.concat(substrate_frames, ignore_index=True)
        if substrate_frames
        else _empty_substrates()
    )
    summary_df = pd.DataFrame(summaries)

    if not results_df.empty:
        result_ids = (
            results_df["contrast"].astype(str)
            + "|"
            + results_df["kinase_id"].astype(str)
        )
        results_df = _set_string_index(results_df, result_ids, "kinase_result_id")
    else:
        results_df.index = pd.Index([], dtype=str, name="kinase_result_id")

    if not substrates_df.empty:
        substrate_ids = (
            substrates_df["contrast"].astype(str)
            + "|"
            + substrates_df["kinase_id"].astype(str)
            + "|"
            + substrates_df["phosphosite_id"].astype(str)
        )
        substrates_df = _set_string_index(
            substrates_df,
            substrate_ids,
            "kinase_substrate_id",
        )
    else:
        substrates_df.index = pd.Index([], dtype=str, name="kinase_substrate_id")

    summary_df = _set_string_index(
        summary_df,
        summary_df["contrast"],
        "contrast_id",
    )

    output = adata.copy()
    output.uns["kinase_activity"] = {
        "schema_version": 1,
        "method": "ksea",
        "organism": parsed_config["organism"],
        "min_substrates": parsed_config["min_substrates"],
        "log2fc_varm_key": log2fc_key,
        "log2fc_source": log2fc_source,
        "background_rule": (
            "all valid localized phosphosites with finite contrast log2FC; "
            "fully imputed contrast cells excluded"
        ),
        "pvalue_rule": "two-sided standard-normal z-test",
        "fdr_scope": "BH within each contrast across tested kinases",
        "database": database_metadata,
        "contrast_summary": summary_df,
        "results": results_df,
        "substrates": substrates_df,
    }
    return output
