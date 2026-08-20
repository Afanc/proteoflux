"""Kinase-substrate enrichment analysis for phosphoproteomics contrasts.

The implementation follows the canonical KSEA z-score.  Kinase-substrate
relationships come from a local, versioned table; no network service is used.
All statistics and multiple-testing correction are computed independently for
each contrast.
"""

from __future__ import annotations

import copy
import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from proteoflux.analysis.stats_ops import bh_qvalues
from proteoflux.utils.utils import log_info, log_time, log_warning
from scipy.stats import norm

# The config defines the input schema explicitly. One kinase identifier plus
# substrate site and at least one substrate identifier (UniProt or gene) are
# mandatory; the remaining concepts are optional metadata.
_DATABASE_COLUMN_KEYS = (
    "kinase",
    "kinase_uniprot",
    "kinase_gene",
    "substrate_uniprot",
    "substrate_gene",
    "substrate_site",
    "source",
)

_SITE_RE = re.compile(r"^[STY]\d+$", flags=re.IGNORECASE)

_MISSING_IDENTIFIER_TOKENS = {
    "",
    "?",
    "NA",
    "N/A",
    "NAN",
    "NONE",
    "NULL",
    "-",
}

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

_CONDITION_KINASE_COLUMNS = [
    "condition",
    "kinase_id",
    "n_substrates",
]


def _pl_clean_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .fill_null("")
        .str.strip_chars()
    )


def _pl_clean_identifier(column: str) -> pl.Expr:
    """Normalize identifiers and convert common missing-value tokens to empty."""
    cleaned = _pl_clean_text(column).str.to_uppercase()
    return (
        pl.when(cleaned.is_in(sorted(_MISSING_IDENTIFIER_TOKENS)))
        .then(pl.lit(""))
        .otherwise(cleaned)
    )


def _pl_first_nonempty(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .filter(pl.col(column).ne(""))
        .first()
        .fill_null("")
        .alias(column)
    )


def _pl_join_unique_list(column: str) -> pl.Expr:
    """Collect sorted unique semicolon-delimited values as a string list."""
    return (
        pl.col(column)
        .str.split(";")
        .flatten()
        .unique()
        .sort()
        .alias(column)
    )


def _pl_join_unique_lists(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return frame.with_columns([
        pl.col(column)
        .list.filter(pl.element().ne(""))
        .list.join(";")
        .alias(column)
        for column in columns
    ])


def _join_semicolon_values(values: list[str]) -> str:
    return ";".join(
        sorted(
            {
                item.strip()
                for value in values
                for item in str(value).split(";")
                if item.strip()
            }
        )
    )


def _resolve_database_column(
    columns: list[str],
    concept: str,
    configured: Any,
) -> str | None:
    configured_name = "" if configured is None else str(configured).strip()
    if not configured_name:
        return None
    if configured_name not in columns:
        raise ValueError(
            f"KSEA database_columns.{concept} refers to missing column "
            f"{configured_name!r}. Available columns: {columns!r}."
        )
    return configured_name


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

    database_value = kinase_cfg.get("substrate_database")
    if isinstance(database_value, (str, Path)):
        database_values = [database_value]
    elif isinstance(database_value, list):
        database_values = database_value
    else:
        raise ValueError(
            "analysis.kinase_activity.substrate_database must be a local path "
            "or a non-empty list of local paths."
        )

    database_paths: list[Path] = []
    for index, value in enumerate(database_values):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(
                "analysis.kinase_activity.substrate_database contains an "
                f"invalid path at index {index}: {value!r}."
            )
        database_paths.append(Path(str(value)).expanduser())

    if not database_paths:
        raise ValueError(
            "analysis.kinase_activity.substrate_database must contain at "
            "least one local kinase-substrate database path."
        )
    normalized_paths = [str(path.resolve(strict=False)) for path in database_paths]
    duplicate_paths = sorted(
        {path for path in normalized_paths if normalized_paths.count(path) > 1}
    )
    if duplicate_paths:
        raise ValueError(
            "analysis.kinase_activity.substrate_database contains duplicate "
            f"paths: {duplicate_paths!r}."
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

    database_columns = kinase_cfg.get("database_columns") or {}
    if not isinstance(database_columns, dict):
        raise ValueError(
            "analysis.kinase_activity.database_columns must be a mapping."
        )
    unknown_columns = sorted(
        set(database_columns).difference(_DATABASE_COLUMN_KEYS)
    )
    if unknown_columns:
        raise ValueError(
            "analysis.kinase_activity.database_columns contains unknown keys: "
            f"{unknown_columns!r}. Supported keys: "
            f"{list(_DATABASE_COLUMN_KEYS)!r}."
        )

    return {
        "method": method,
        "substrate_databases": database_paths,
        "min_substrates": min_substrates,
        "database_columns": database_columns,
    }


def _load_substrate_database(
    database_path: Path,
    configured_columns: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if not database_path.is_file():
        raise FileNotFoundError(
            f"KSEA substrate database does not exist or is not a file: {database_path}"
        )

    with database_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), None)
    if not header:
        raise ValueError(
            f"KSEA substrate database contains no header: {database_path}"
        )

    columns = [str(column).lstrip("\ufeff").strip() for column in header]
    duplicated_columns = sorted(
        {column for column in columns if columns.count(column) > 1}
    )
    if duplicated_columns:
        raise ValueError(
            "KSEA substrate database contains duplicate columns: "
            f"{duplicated_columns!r}."
        )

    resolved = {
        concept: _resolve_database_column(
            columns,
            concept,
            configured_columns.get(concept),
        )
        for concept in _DATABASE_COLUMN_KEYS
    }
    kinase_identity_columns = [
        resolved[concept]
        for concept in ("kinase_uniprot", "kinase_gene", "kinase")
        if resolved[concept] is not None
    ]
    missing_concepts = []
    if resolved["substrate_site"] is None:
        missing_concepts.append("substrate_site")
    if (
        resolved["substrate_uniprot"] is None
        and resolved["substrate_gene"] is None
    ):
        missing_concepts.append(
            "substrate identifier (substrate_uniprot or substrate_gene)"
        )
    if not kinase_identity_columns:
        missing_concepts.insert(0, "kinase identifier")
    if missing_concepts:
        raise ValueError(
            "KSEA substrate database could not resolve required concepts: "
            f"{missing_concepts!r}. Available columns: {columns!r}. Set the "
            "corresponding analysis.kinase_activity.database_columns mappings."
        )

    def optional_text(concept: str) -> pl.Expr:
        column = resolved[concept]
        if column is None:
            return pl.lit("", dtype=pl.String)
        return _pl_clean_text(column)

    def optional_identifier(concept: str) -> pl.Expr:
        column = resolved[concept]
        if column is None:
            return pl.lit("", dtype=pl.String)
        return _pl_clean_identifier(column)

    # Projection pushdown means the large input files parse only the configured
    # identifier/site/source columns, and infer_schema=False keeps every value
    # as text without a separate schema-inference pass.
    database = (
        pl.scan_csv(
            database_path,
            separator="\t",
            infer_schema=False,
            missing_utf8_is_empty_string=True,
            low_memory=False,
            rechunk=False,
            with_column_names=lambda names: [
                str(name).lstrip("\ufeff").strip() for name in names
            ],
        )
        .select(
            optional_text("kinase").alias("kinase"),
            optional_identifier("kinase_uniprot").alias("kinase_uniprot"),
            optional_identifier("kinase_gene").alias("kinase_gene"),
            optional_identifier("substrate_uniprot").alias(
                "substrate_accession"
            ),
            optional_identifier("substrate_gene").alias("substrate_gene"),
            optional_text("substrate_site").str.to_uppercase().alias("site"),
            optional_text("source").alias("database_source"),
        )
        .collect()
    )
    if database.is_empty():
        raise ValueError(
            f"KSEA substrate database contains no rows: {database_path}"
        )

    kinase_isoform_rows_normalized = int(
        database.select(
            pl.col("kinase_uniprot").str.contains(r"-\d+$").sum()
        ).item()
    )
    database = database.with_columns(
        pl.when(pl.col("database_source").eq(""))
        .then(pl.lit(database_path.stem))
        .otherwise(pl.col("database_source"))
        .alias("database_source"),
        # KSEA reports activity for the kinase gene/protein, not separately for
        # UniProt isoforms. Keep substrate accessions untouched because their
        # residue numbering can be isoform-specific.
        pl.col("kinase_uniprot")
        .str.replace(r"-\d+$", "")
        .alias("kinase_uniprot"),
        _pl_clean_identifier("kinase").alias("_kinase_name_id"),
    )

    # Prefer UniProt for each database row. Fall back to the substrate gene
    # only when the UniProt accession is genuinely unavailable.
    database = database.with_columns(
        pl.when(pl.col("substrate_accession").ne(""))
        .then(pl.lit("uniprot"))
        .otherwise(pl.lit("gene"))
        .alias("substrate_match_type"),
        pl.when(pl.col("substrate_accession").ne(""))
        .then(pl.col("substrate_accession"))
        .otherwise(pl.col("substrate_gene"))
        .alias("substrate_id"),
        pl.when(pl.col("kinase_gene").ne(""))
        .then(pl.col("kinase_gene"))
        .when(pl.col("kinase").ne(""))
        .then(pl.col("kinase"))
        .otherwise(pl.col("kinase_uniprot"))
        .alias("kinase"),
        pl.when(pl.col("kinase_gene").ne(""))
        .then(pl.col("kinase_gene"))
        .when(pl.col("kinase_uniprot").ne(""))
        .then(pl.col("kinase_uniprot"))
        .otherwise(pl.col("_kinase_name_id"))
        .alias("kinase_id"),
    ).drop("_kinase_name_id")

    valid_site = pl.col("site").str.contains(r"^[STY]\d+$")
    if not bool(database.select(valid_site.any()).item()):
        site_examples = (
            database.filter(pl.col("site").ne(""))
            .select("site")
            .unique(maintain_order=True)
            .head(5)
            .get_column("site")
            .to_list()
        )
        raise ValueError(
            f"KSEA column {resolved['substrate_site']!r}, resolved as "
            f"substrate_site, contains no valid S/T/Y sites. Expected values "
            f"such as 'S52'; observed examples: {site_examples!r}. Check "
            "analysis.kinase_activity.database_columns.substrate_site."
        )

    valid = (
        pl.col("kinase_id").ne("")
        & pl.col("substrate_id").ne("")
        & valid_site
    )
    input_rows = database.height
    both_substrate_ids = int(
        database.select(
            (
                pl.col("substrate_accession").ne("")
                & pl.col("substrate_gene").ne("")
            ).sum()
        ).item()
    )
    database = database.filter(valid)
    invalid_rows = int(input_rows - database.height)
    if database.is_empty():
        raise ValueError(
            "KSEA substrate database has no valid relationships after "
            "requiring a kinase identifier, substrate UniProt accession or "
            f"gene symbol, and an S/T/Y site: {database_path}"
        )

    source_counts = {
        str(source): int(count)
        for source, count in database.group_by("database_source")
        .len()
        .sort("database_source")
        .iter_rows()
    }

    relationships = (
        database.group_by(
            [
                "kinase_id",
                "substrate_match_type",
                "substrate_id",
                "site",
            ],
        )
        .agg(
            _pl_first_nonempty("kinase"),
            _pl_first_nonempty("kinase_gene"),
            _pl_first_nonempty("kinase_uniprot"),
            _pl_first_nonempty("substrate_accession"),
            _pl_first_nonempty("substrate_gene"),
            _pl_join_unique_list("database_source"),
        )
    )
    relationships = _pl_join_unique_lists(relationships, ["database_source"])

    metadata = {
        "filename": database_path.name,
        "source": _join_semicolon_values(list(source_counts)),
        "input_rows": input_rows,
        "valid_rows": database.height,
        "invalid_rows": invalid_rows,
        "relationships": relationships.height,
        "duplicate_relationships_removed": int(
            database.height - relationships.height
        ),
        "uniprot_relationships": relationships.filter(
            pl.col("substrate_match_type").eq("uniprot")
        ).height,
        "gene_fallback_relationships": relationships.filter(
            pl.col("substrate_match_type").eq("gene")
        ).height,
        "rows_with_both_substrate_ids": both_substrate_ids,
        "kinase_isoform_rows_normalized": kinase_isoform_rows_normalized,
        "source_counts": source_counts,
        "column_mapping": {
            concept: column
            for concept, column in resolved.items()
            if column is not None
        },
    }

    return relationships, metadata


def _reconcile_kinase_identities(
    relationships: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Map accession-only kinases to unambiguous genes across databases."""
    pairs = (
        relationships.filter(
            pl.col("kinase_gene").ne("")
            & pl.col("kinase_uniprot").ne("")
        )
        .select("kinase_gene", "kinase_uniprot")
        .unique()
    )
    if pairs.is_empty():
        return relationships, {
            "accession_only_kinase_relationships_remapped": 0,
            "conflicting_kinase_accessions": 0,
            "conflicting_kinase_accession_examples": [],
        }

    accession_map = pairs.group_by("kinase_uniprot").agg(
        pl.col("kinase_gene").n_unique().alias("_n_genes"),
        pl.col("kinase_gene").first().alias("_canonical_kinase_gene"),
    )
    conflicts = accession_map.filter(pl.col("_n_genes").gt(1))
    unambiguous = accession_map.filter(pl.col("_n_genes").eq(1)).select(
        "kinase_uniprot",
        "_canonical_kinase_gene",
    )

    relationships = relationships.join(
        unambiguous,
        on="kinase_uniprot",
        how="left",
    )
    remap = (
        pl.col("kinase_gene").eq("")
        & pl.col("_canonical_kinase_gene").is_not_null()
    )
    remapped = int(relationships.select(remap.sum()).item())
    relationships = relationships.with_columns(
        pl.when(remap)
        .then(pl.col("_canonical_kinase_gene"))
        .otherwise(pl.col("kinase_id"))
        .alias("kinase_id"),
        pl.when(remap)
        .then(pl.col("_canonical_kinase_gene"))
        .otherwise(pl.col("kinase"))
        .alias("kinase"),
        pl.when(remap)
        .then(pl.col("_canonical_kinase_gene"))
        .otherwise(pl.col("kinase_gene"))
        .alias("kinase_gene"),
    ).drop("_canonical_kinase_gene")

    return relationships, {
        "accession_only_kinase_relationships_remapped": remapped,
        "conflicting_kinase_accessions": conflicts.height,
        "conflicting_kinase_accession_examples": conflicts.get_column(
            "kinase_uniprot"
        ).head(5).to_list(),
    }


def _load_substrate_databases_uncached(
    database_paths: list[Path],
    configured_columns: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    loaded: list[tuple[pl.DataFrame, dict[str, Any]]] = []
    for priority, path in enumerate(database_paths):
        frame, metadata = _load_substrate_database(
            path,
            configured_columns,
        )
        frame = frame.with_columns(
            pl.lit(priority, dtype=pl.Int32).alias("database_priority")
        )
        loaded.append((frame, metadata))

    relationships = pl.concat(
        [frame for frame, _metadata in loaded],
        how="vertical_relaxed",
        rechunk=False,
    )
    relationships, kinase_identity_metadata = _reconcile_kinase_identities(
        relationships
    )
    if kinase_identity_metadata["conflicting_kinase_accessions"]:
        log_warning(
            "KSEA found kinase UniProt accessions assigned to multiple gene "
            "symbols: accessions="
            f"{kinase_identity_metadata['conflicting_kinase_accessions']:,}, "
            "examples="
            f"{kinase_identity_metadata['conflicting_kinase_accession_examples']!r}. "
            "Gene symbols remain canonical, so these relationships stay "
            "separate; check the database annotations."
        )

    # Configured database order defines priority for exact duplicate
    # relationships. Keep the highest-priority metadata while retaining all
    # contributing source labels for provenance.
    relationships_before_merge = relationships.height
    relationships = relationships.sort(
        "database_priority",
        maintain_order=True,
    ).group_by(
        [
            "kinase_id",
            "substrate_match_type",
            "substrate_id",
            "site",
        ],
    ).agg(
        _pl_first_nonempty("kinase"),
        _pl_first_nonempty("kinase_gene"),
        _pl_first_nonempty("kinase_uniprot"),
        _pl_first_nonempty("substrate_accession"),
        _pl_first_nonempty("substrate_gene"),
        _pl_join_unique_list("database_source"),
        pl.col("database_priority").min(),
    )
    relationships = _pl_join_unique_lists(relationships, ["database_source"])

    metadata_rows = [metadata for _frame, metadata in loaded]

    new_relationship_counts = {
        int(priority): int(count)
        for priority, count in relationships.group_by(
            "database_priority"
        ).len().iter_rows()
    }
    for priority, item in enumerate(metadata_rows):
        item["new_relationships"] = new_relationship_counts.get(priority, 0)

    metadata = {
        "filename": ", ".join(
            str(item["filename"]) for item in metadata_rows
        ),
        "source": _join_semicolon_values(
            [str(item.get("source", "")) for item in metadata_rows]
        ),
        "input_rows": sum(int(item["input_rows"]) for item in metadata_rows),
        "valid_rows": sum(int(item["valid_rows"]) for item in metadata_rows),
        "invalid_rows": sum(int(item["invalid_rows"]) for item in metadata_rows),
        "relationships": relationships.height,
        "duplicate_relationships_removed": (
            sum(
                int(item["duplicate_relationships_removed"])
                for item in metadata_rows
            )
            + int(relationships_before_merge - relationships.height)
        ),
        "kinase_isoform_rows_normalized": sum(
            int(item["kinase_isoform_rows_normalized"])
            for item in metadata_rows
        ),
        "accession_only_kinase_relationships_remapped": (
            kinase_identity_metadata[
                "accession_only_kinase_relationships_remapped"
            ]
        ),
        "conflicting_kinase_accessions": kinase_identity_metadata[
            "conflicting_kinase_accessions"
        ],
        "identifier_policy": (
            "kinase gene first with UniProt/name fallback and kinase isoforms "
            "collapsed; substrate UniProt first with gene fallback only when "
            "UniProt is absent"
        ),
        "column_mapping": {
            key: str(value).strip()
            for key, value in configured_columns.items()
            if value is not None and str(value).strip()
        },
        "_file_summaries": metadata_rows,
    }

    return relationships, metadata


@lru_cache(maxsize=2)
def _load_substrate_databases_cached(
    database_signatures: tuple[tuple[str, int, int], ...],
    configured_columns: tuple[tuple[str, str], ...],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = [Path(path) for path, _size, _mtime_ns in database_signatures]
    columns = {
        key: value if value else None
        for key, value in configured_columns
    }
    return _load_substrate_databases_uncached(paths, columns)


def _load_substrate_databases(
    database_paths: list[Path],
    configured_columns: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    signatures: list[tuple[str, int, int]] = []
    for path in database_paths:
        if not path.is_file():
            raise FileNotFoundError(
                "KSEA substrate database does not exist or is not a file: "
                f"{path}"
            )
        resolved_path = path.resolve()
        stat = resolved_path.stat()
        signatures.append(
            (str(resolved_path), int(stat.st_size), int(stat.st_mtime_ns))
        )

    column_signature = tuple(
        (
            key,
            "" if configured_columns.get(key) is None
            else str(configured_columns.get(key)).strip(),
        )
        for key in _DATABASE_COLUMN_KEYS
    )
    relationships, metadata = _load_substrate_databases_cached(
        tuple(signatures),
        column_signature,
    )
    # Polars clones are cheap and prevent callers from mutating the cached handle.
    return relationships.clone(), copy.deepcopy(metadata)


def validate_ksea_database(config: dict) -> None:
    """Validate the configured KSEA database without running KSEA."""
    parsed_config = _parse_config(config)
    if parsed_config is None:
        return

    relationships, metadata = _load_substrate_databases(
        parsed_config["substrate_databases"],
        parsed_config["database_columns"],
    )

    log_info(
        "KSEA database validation passed: "
        f"file={metadata['filename']!r}, "
        f"relationships={len(relationships):,}, "
        f"invalid_rows={metadata['invalid_rows']:,}, "
        "duplicate_relationships_removed="
        f"{metadata['duplicate_relationships_removed']:,}."
    )


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
    gene_column = "GENE_NAMES" if "GENE_NAMES" in adata.var.columns else None

    feature_names = adata.var_names.astype(str).tolist()
    n_features = len(feature_names)
    columns = [
        "feature_index",
        "phosphosite_id",
        "substrate_match_type",
        "substrate_id",
        "matched_accession",
        "site",
    ]
    if n_features == 0:
        return pd.DataFrame(columns=columns), 0

    parent_values = (
        [_scalar_text(value) for value in adata.var[parent_column].tolist()]
        if parent_column is not None
        else [""] * n_features
    )
    gene_values = (
        [_scalar_text(value) for value in adata.var[gene_column].tolist()]
        if gene_column is not None
        else [""] * n_features
    )

    def identifier_list(column: str) -> pl.Expr:
        return (
            pl.col(column)
            .str.split(";")
            .list.eval(pl.element().str.strip_chars().str.to_uppercase())
            .list.filter(
                pl.element().ne("")
                & ~pl.element().is_in(sorted(_MISSING_IDENTIFIER_TOKENS))
            )
            .list.unique()
            .list.sort()
        )

    features = pl.DataFrame(
        {
            "feature_index": np.arange(n_features, dtype=np.int64),
            "phosphosite_id": feature_names,
            "_parent_value": parent_values,
            "_gene_value": gene_values,
        }
    ).with_columns(
        pl.col("phosphosite_id")
        .str.extract(r"^(.*)\|([^|]+)$", group_index=1)
        .fill_null("")
        .str.strip_chars()
        .alias("_index_parent"),
        pl.col("phosphosite_id")
        .str.extract(r"^(.*)\|([^|]+)$", group_index=2)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
        .alias("site"),
    )
    features = features.with_columns(
        pl.when(pl.col("_parent_value").ne(""))
        .then(pl.col("_parent_value"))
        .otherwise(pl.col("_index_parent"))
        .alias("_parent_value"),
    ).with_columns(
        identifier_list("_parent_value").alias("_accessions"),
        identifier_list("_gene_value").alias("_genes"),
    ).with_columns(
        pl.col("_accessions").list.join(";").alias("_parent_accessions")
    )

    valid = (
        pl.col("site").str.contains(r"^[STY]\d+$")
        & (
            pl.col("_accessions").list.len().gt(0)
            | pl.col("_genes").list.len().gt(0)
        )
    )
    valid_features = features.filter(valid)
    invalid_features = int(n_features - valid_features.height)

    accessions = (
        valid_features.select(
            "feature_index",
            "phosphosite_id",
            "site",
            pl.col("_accessions").alias("substrate_id"),
        )
        .explode("substrate_id")
        .filter(pl.col("substrate_id").is_not_null())
        .with_columns(
            pl.lit("uniprot").alias("substrate_match_type"),
            pl.col("substrate_id").alias("matched_accession"),
        )
    )
    genes = (
        valid_features.select(
            "feature_index",
            "phosphosite_id",
            "site",
            "_parent_accessions",
            pl.col("_genes").alias("substrate_id"),
        )
        .explode("substrate_id")
        .filter(pl.col("substrate_id").is_not_null())
        .with_columns(
            pl.lit("gene").alias("substrate_match_type"),
            pl.col("_parent_accessions").alias("matched_accession"),
        )
        .drop("_parent_accessions")
    )

    feature_sites = pl.concat(
        [accessions.select(columns), genes.select(columns)],
        how="vertical_relaxed",
        rechunk=False,
    )
    return (
        pd.DataFrame(feature_sites.to_dict(as_series=False), columns=columns),
        invalid_features,
    )


def _match_substrates(
    feature_sites: pd.DataFrame,
    relationships: pl.DataFrame | pd.DataFrame,
) -> pd.DataFrame:
    empty_columns = [
        "kinase_id",
        "kinase",
        "kinase_gene",
        "kinase_uniprot",
        "feature_index",
        "phosphosite_id",
        "matched_accession",
        "database_source",
        "match_types",
        "database_priority",
    ]
    relationships_empty = (
        relationships.is_empty()
        if isinstance(relationships, pl.DataFrame)
        else relationships.empty
    )
    if feature_sites.empty or relationships_empty:
        return pd.DataFrame(columns=empty_columns)

    feature_sites_pl = pl.from_pandas(feature_sites, include_index=False)
    relationships_for_match = (
        relationships.clone()
        if isinstance(relationships, pl.DataFrame)
        else pl.from_pandas(relationships, include_index=False)
    )
    if "database_priority" not in relationships_for_match.columns:
        relationships_for_match = relationships_for_match.with_columns(
            pl.lit(0, dtype=pl.Int32).alias("database_priority")
        )

    matched = feature_sites_pl.join(
        relationships_for_match.select(
            [
                "kinase_id",
                "kinase",
                "kinase_gene",
                "kinase_uniprot",
                "substrate_match_type",
                "substrate_id",
                "site",
                "database_source",
                "database_priority",
            ]
        ),
        on=["substrate_match_type", "substrate_id", "site"],
        how="inner",
    )
    if matched.is_empty():
        return pd.DataFrame(columns=empty_columns)

    # Count a kinase/phosphosite once even if it was reached through several
    # identifiers or databases. Database order chooses display metadata, while
    # every contributing source and match type remains visible for provenance.
    matched = matched.sort(
        "database_priority",
        maintain_order=True,
    ).group_by(
        ["kinase_id", "feature_index", "phosphosite_id"],
    ).agg(
        _pl_first_nonempty("kinase"),
        _pl_first_nonempty("kinase_gene"),
        _pl_first_nonempty("kinase_uniprot"),
        _pl_join_unique_list("matched_accession"),
        _pl_join_unique_list("database_source"),
        _pl_join_unique_list("substrate_match_type").alias("match_types"),
        pl.col("database_priority").min(),
    )
    matched = _pl_join_unique_lists(
        matched,
        ["matched_accession", "database_source", "match_types"],
    )
    # The matched table is small relative to the relationship database; this
    # conversion avoids adding a pyarrow requirement to the KSEA path.
    return pd.DataFrame(matched.to_dict(as_series=False))


def _database_contribution_summary(
    file_summaries: list[dict[str, Any]],
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each database's incremental contribution in config order."""
    columns = [
        "filename",
        "relationships",
        "new_relationships",
        "new_matched_relationships",
        "new_matched_phosphosites",
        "new_matched_kinases",
    ]

    relationship_counts: dict[int, int] = {}
    phosphosite_counts: dict[int, int] = {}
    kinase_counts: dict[int, int] = {}
    if not assignments.empty:
        work = assignments.copy()
        work["database_priority"] = pd.to_numeric(
            work["database_priority"], errors="coerce"
        )
        work = work.dropna(subset=["database_priority"])
        work["database_priority"] = work["database_priority"].astype(int)

        relationship_counts = {
            int(priority): int(count)
            for priority, count in work["database_priority"]
            .value_counts()
            .items()
        }
        phosphosite_counts = {
            int(priority): int(count)
            for priority, count in work.groupby("feature_index")[
                "database_priority"
            ]
            .min()
            .value_counts()
            .items()
        }
        kinase_counts = {
            int(priority): int(count)
            for priority, count in work.groupby("kinase_id")[
                "database_priority"
            ]
            .min()
            .value_counts()
            .items()
        }

    rows = [
        {
            "filename": str(item["filename"]),
            "relationships": int(item["relationships"]),
            "new_relationships": int(item.get("new_relationships", 0)),
            "new_matched_relationships": relationship_counts.get(priority, 0),
            "new_matched_phosphosites": phosphosite_counts.get(priority, 0),
            "new_matched_kinases": kinase_counts.get(priority, 0),
        }
        for priority, item in enumerate(file_summaries)
    ]
    summary = pd.DataFrame(rows, columns=columns)
    summary.index = pd.Index(
        [str(priority) for priority in range(len(summary))],
        dtype=str,
        name="database_priority",
    )
    return summary


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

def _condition_kinase_data(
    adata: ad.AnnData,
    assignments: pd.DataFrame,
    valid_feature_mask: np.ndarray,
    min_substrates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "condition",
        "observed_phosphosites",
        "database_matched_sites",
        "matched_kinases",
        "eligible_kinases",
    ]

    if "raw" not in adata.layers:
        log_warning(
            "KSEA condition-level kinase counts are unavailable because "
            "adata.layers['raw'] is missing."
        )
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=_CONDITION_KINASE_COLUMNS),
        )

    raw_layer = adata.layers["raw"]
    raw = raw_layer.toarray() if hasattr(raw_layer, "toarray") else np.asarray(raw_layer)
    if raw.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            "adata.layers['raw'] must have shape "
            f"{(adata.n_obs, adata.n_vars)}, got {raw.shape}."
        )

    condition_values = adata.obs["CONDITION"].astype(str).to_numpy()
    conditions = list(dict.fromkeys(condition_values.tolist()))
    assignment_indices = assignments["feature_index"].to_numpy(dtype=int)

    summary_rows: list[dict[str, object]] = []
    kinase_rows: list[dict[str, object]] = []
    for condition in conditions:
        sample_mask = condition_values == condition
        observed = (
            valid_feature_mask
            & np.any(np.isfinite(raw[sample_mask, :]), axis=0)
        )

        matched = assignments.loc[observed[assignment_indices]]
        if matched.empty:
            substrate_counts = pd.Series(dtype=int)
        else:
            substrate_counts = matched.groupby("kinase_id")[
                "feature_index"
            ].nunique()

        eligible_counts = substrate_counts.loc[
            substrate_counts.ge(min_substrates)
        ]
        summary_rows.append(
            {
                "condition": condition,
                "observed_phosphosites": int(observed.sum()),
                "database_matched_sites": int(
                    matched["feature_index"].nunique()
                ),
                "matched_kinases": int(substrate_counts.size),
                "eligible_kinases": int(eligible_counts.size),
            }
        )
        kinase_rows.extend(
            {
                "condition": condition,
                "kinase_id": str(kinase_id),
                "n_substrates": int(n_substrates),
            }
            for kinase_id, n_substrates in eligible_counts.items()
        )

    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    summary.index = pd.Index(
        summary["condition"].astype(str),
        name="condition_id",
    )

    condition_kinases = pd.DataFrame(
        kinase_rows,
        columns=_CONDITION_KINASE_COLUMNS,
    )
    if condition_kinases.empty:
        condition_kinases.index = pd.Index(
            [], dtype=str, name="condition_kinase_id"
        )
    else:
        membership_ids = (
            condition_kinases["condition"].astype(str)
            + "|"
            + condition_kinases["kinase_id"].astype(str)
        )
        condition_kinases.index = pd.Index(
            membership_ids,
            name="condition_kinase_id",
        )
    return summary, condition_kinases


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

    relationships, database_metadata = _load_substrate_databases(
        parsed_config["substrate_databases"],
        parsed_config["database_columns"],
    )
    for file_summary in database_metadata.get("_file_summaries", []):
        log_info(
            "KSEA database loaded: "
            f"file={file_summary['filename']!r}, "
            f"rows={file_summary['input_rows']:,}, "
            f"relationships={file_summary['relationships']:,}, "
            "uniprot_relationships="
            f"{file_summary['uniprot_relationships']:,}, "
            "gene_fallback_relationships="
            f"{file_summary['gene_fallback_relationships']:,}, "
            f"invalid_rows={file_summary['invalid_rows']:,}, "
            "duplicate_relationships_removed="
            f"{file_summary['duplicate_relationships_removed']:,}."
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

    if assignments.empty:
        uniprot_assignments = 0
        gene_only_assignments = 0
        mixed_identifier_assignments = 0
    else:
        match_type_sets = assignments["match_types"].map(
            lambda value: set(str(value).split(";"))
        )
        has_uniprot = match_type_sets.map(lambda values: "uniprot" in values)
        has_gene = match_type_sets.map(lambda values: "gene" in values)
        uniprot_assignments = int(has_uniprot.sum())
        gene_only_assignments = int((has_gene & ~has_uniprot).sum())
        mixed_identifier_assignments = int((has_gene & has_uniprot).sum())

    log_info(
        "KSEA database matching: "
        f"relationships={len(relationships):,}; "
        f"kinase-site assignments={len(assignments):,}; "
        f"matched phosphosites="
        f"{assignments['feature_index'].nunique() if not assignments.empty else 0:,}; "
        f"matched kinases="
        f"{assignments['kinase_id'].nunique() if not assignments.empty else 0:,}; "
        f"uniprot assignments={uniprot_assignments:,}; "
        f"gene-only assignments={gene_only_assignments:,}; "
        f"mixed-source assignments={mixed_identifier_assignments:,}; "
        f"invalid dataset features={invalid_features:,}."
    )

    condition_summary_df, condition_kinases_df = _condition_kinase_data(
        adata=adata,
        assignments=assignments,
        valid_feature_mask=valid_feature_mask,
        min_substrates=parsed_config["min_substrates"],
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

    matched_phosphosites = (
        int(assignments["feature_index"].nunique())
        if not assignments.empty
        else 0
    )
    unmatched_phosphosites = int(adata.n_vars - matched_phosphosites)

    matched_kinases = (
        int(results_df["kinase_id"].nunique())
        if not results_df.empty
        else 0
    )
    tested_kinases = (
        int(
            results_df.loc[
                results_df["tested"].eq(True),
                "kinase_id",
            ].nunique()
        )
        if not results_df.empty
        else 0
    )

    log_info(
        "KSEA summary: "
        f"phosphosites matched={matched_phosphosites:,}/{adata.n_vars:,}, "
        f"unmatched={unmatched_phosphosites:,}; "
        f"kinases tested in >=1 contrast="
        f"{tested_kinases:,}/{matched_kinases:,}; "
    )

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
    stored_database_metadata = {
        key: value
        for key, value in database_metadata.items()
        if not key.startswith("_")
    }
    stored_database_metadata.update(
        {
            "matched_relationships": int(len(assignments)),
            "matched_phosphosites": matched_phosphosites,
            "matched_kinases": int(assignments["kinase_id"].nunique())
            if not assignments.empty
            else 0,
        }
    )
    database_summary_df = _database_contribution_summary(
        database_metadata.get("_file_summaries", []),
        assignments,
    )

    output = adata.copy()
    output.uns["kinase_activity"] = {
        "schema_version": 1,
        "method": "ksea",
        "min_substrates": parsed_config["min_substrates"],
        "log2fc_varm_key": log2fc_key,
        "log2fc_source": log2fc_source,
        "background_rule": (
            "all valid localized phosphosites with finite contrast log2FC; "
            "fully imputed contrast cells excluded"
        ),
        "pvalue_rule": "two-sided standard-normal z-test",
        "fdr_scope": "BH within each contrast across tested kinases",
        "database": stored_database_metadata,
        "database_summary": database_summary_df,
        "condition_summary": condition_summary_df,
        "condition_kinases": condition_kinases_df,
        "contrast_summary": summary_df,
        "results": results_df,
        "substrates": substrates_df,
    }
    return output
