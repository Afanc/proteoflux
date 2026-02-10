"""Preprocessing pipeline for Proteoflux.

This module performs:
1) Filtering (contaminants, q-value, PEP, min. precursors)
2) Metadata extraction (protein meta, condition map)
3) Pivoting to wide matrices (intensity, qvalue, PEP, precursors, etc.)
4) Normalization (log, median, quantile, regression-based, tag-based)
5) Imputation (main and optional covariate block)
6) Optional phospho localization filtering

All steps record intermediate artifacts to `IntermediateResults`, which are then
assembled into a `PreprocessResults` container consumed by downstream code.
"""

import time
import sys
import re
from typing import Optional, List, Tuple, Dict

import numpy as np
import polars as pl
import pandas as pd

import sklearn.impute
import sklearn.preprocessing
import sklearn.ensemble
from skmisc.loess import loess

from proteoflux.workflow.normalizers.regression_normalization import regression_normalization
from proteoflux.workflow.imputer_factory import get_imputer
from proteoflux.dataset.preprocessresults import PreprocessResults
from proteoflux.dataset.intermediateresults import IntermediateResults
from proteoflux.utils.utils import (
    load_contaminant_accessions,
    log_time,
    logger,
    log_info,
    log_indent,
)
from proteoflux.utils.sequence_ops import (
    expr_peptide_index_seq,
    expr_strip_bracket_mods,
    expr_strip_underscores,
)
from proteoflux.workflow.peptide_tables import build_peptide_tables
from proteoflux.utils.analysis_type import normalize_analysis_type

from proteoflux.utils.directlfq import estimate_protein_intensities
import proteoflux.utils.directlfq_config as dlcfg
from enum import Enum


class Block(str, Enum):
    MAIN = "main"
    COV = "covariate"

def _missing_columns(df: pl.DataFrame, required: set[str]) -> list[str]:
    """Return sorted list of missing columns (strict, deterministic order)."""
    return sorted(required - set(df.columns))


def _require_columns(df: pl.DataFrame, required: set[str], *, context: str) -> None:
    """Fail-fast required-column guard with consistent error formatting."""
    missing = _missing_columns(df, required)
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing!r}")


class Preprocessor:
    """Handles filtering, pivoting, normalization, and imputation for proteomics data."""

    available_normalization = ["zscore", "vst", "linear"]
    available_imputation = ["mean", "median", "knn", "randomforest"]

    def __init__(self, config: Optional[dict] = None):
        """Initialize from a config dict mirroring `config.yaml` sections."""

        self.intermediate_results = IntermediateResults()
        self.analysis_type = normalize_analysis_type((config or {}).get("analysis_type"))

        config = config or {}

        # Filtering
        self.filtering = config.get("filtering") or {}
        self.remove_contaminants = self.filtering.get("contaminants_files", [])
        self.filter_qvalue = self.filtering.get("qvalue", 0.01)
        self.filter_pep = self.filtering.get("pep", 0.2)
        self.filter_num_precursors = self.filtering.get("min_precursors", 1)
        self.min_linear_intensity = self.filtering.get("min_linear_intensity", None)
        self.pep_direction = (
            self.filtering.get("pep_direction", "lower") or "lower"
        ).lower()
        if self.pep_direction not in {"lower", "higher"}:
            raise ValueError(
                f"Invalid pep_direction='{self.pep_direction}'. Use 'lower' or 'higher'."
            )

        # Pivoting / rollups
        # Canonical key: protein_rollup_method
        # Legacy alias: pivot_signal_method
        protein_rollup_method = config.get("protein_rollup_method", None)
        pivot_signal_method = config.get("pivot_signal_method", None)

        if (protein_rollup_method is not None) and (pivot_signal_method is not None):
            raise ValueError(
                "Config error: both 'protein_rollup_method' and legacy "
                "'pivot_signal_method' are set. Use only 'protein_rollup_method'."
            )
        allowed_protein_rollup_methods = ["sum", "mean", "median", "top3", "directlfq", "min", "max", "count", None]
        if protein_rollup_method not in allowed_protein_rollup_methods:
            raise ValueError(
                "Invalid protein_rollup_method "
                f"'{protein_rollup_method}'. "
                f"Allowed values: {allowed_protein_rollup_methods}"
            )


        if protein_rollup_method is None:
            protein_rollup_method = pivot_signal_method

        self.protein_rollup_method = (protein_rollup_method or "sum")

        # Explicit peptide rollup (precursor→peptide, peptide tables)
        # Default remains 'sum' to preserve current behavior.
        self.peptide_rollup_method = (
            config.get("peptide_rollup_method", "sum") or "sum"
        ).lower()

        if self.peptide_rollup_method not in {"sum", "mean", "median"}:
            raise ValueError(
                "Invalid peptide_rollup_method "
                f"'{self.peptide_rollup_method}'. "
                "Allowed values: 'sum', 'mean', 'median'."
            )

        # Peptide identity normalization 
        self.collapse_met_oxidation = bool(config.get("collapse_met_oxidation", True))
        self.collapse_all_ptms = bool(config.get("collapse_all_ptms", False))

        self.directlfq_cores = config.get("directlfq_cores", 4)
        self.directlfq_min_nonan = config.get("directlfq_min_nonan", 1)

        # Normalization
        self.normalization = config.get("normalization") or {}

        # Imputation
        self.imputation = config.get("imputation") or {}

        self.exports = config.get("exports") or {}

        # Phospho block
        phospho_cfg = config.get("phospho") or {}

        _mode = (
            phospho_cfg.get("localization_filter_mode", "soft") or "soft"
        ).lower()
        if _mode not in {"soft", "strict"}:
            raise ValueError(
                "preprocessing.phospho.localization_mode must be 'soft' or 'strict'"
            )
        self.phospho_loc_mode = _mode
        self.phospho_loc_thr = float(
            phospho_cfg.get("localization_filter_threshold", 0.75)
        )
        self.covariate_protein_rollup_method = phospho_cfg.get(
            "covariate_protein_rollup_method", "directlfq"
        ).lower()

        # Covariates: list of assay labels to extract & center after imputation
        cov_cfg = config.get("covariates") or {}
        self.covariate_assays: List[str] = [
            str(a).lower() for a in (cov_cfg.get("assays") or [])
        ]
        self.covariates_enabled = bool(cov_cfg.get("enabled", False))

        self.cov_align_on = "parent_protein"

    def _split_main_covariate_rows(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """
        Split a long table into main vs covariate rows.
        Mechanical extraction of the previous inlined logic:
          - Prefer IS_COVARIATE if present (Null treated as False)
          - Else, if ASSAY present and covariate assays configured, split by lowercased ASSAY membership
          - Else: no covariate block (df_cov=None)
        NOTE: This function does NOT enforce presence of covariate rows. That check
        remains at the call site (same as before).
        """
        if not self.covariates_enabled:
            return df, None

        if "IS_COVARIATE" in df.columns:
            # treat nulls as False (exact legacy semantics)
            is_cov = pl.coalesce([pl.col("IS_COVARIATE"), pl.lit(False)])
            return df.filter(~is_cov), df.filter(is_cov)

        # Fallback: infer covariate rows from ASSAY matching configured assays (case-insensitive)
        has_assay = "ASSAY" in df.columns
        assay_lc = set(self.covariate_assays or [])
        if has_assay and assay_lc:
            return (
                df.filter(~pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))),
                df.filter(pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))),
            )

        return df, None

    def fit_transform(self, df: pl.DataFrame) -> PreprocessResults:
        """Run the full preprocessing pipeline and return a `PreprocessResults` bundle."""

        # Step 1: Filtering
        self._filter(df)

        # Step 2: Metadata
        self._get_protein_metadata()
        self._get_condition_map()

        # Step 3: Pivoting
        self._pivot_data()

        # Step 4: Normalization
        self._normalize()

        # Step 5: Imputation
        self._impute()

        # Final step: returning the structured data class
        ir = self.intermediate_results
        meta_filtering = ir.metadata.get("filtering", {})

        return PreprocessResults(
            filtered=ir.dfs.get("raw_df"),
            lognormalized=ir.dfs.get("postlog"),
            normalized=ir.dfs.get("normalized"),
            processed=ir.dfs.get("imputed"),
            qvalues=ir.dfs.get("qvalues"),
            pep=ir.dfs.get("pep"),
            locprob=ir.dfs.get("locprob"),
            spectral_counts=ir.dfs.get("spectral_counts"),
            ibaq=ir.dfs.get("ibaq"),
            condition_pivot=ir.dfs.get("condition_pivot"),
            protein_meta=ir.dfs.get("protein_metadata"),
            peptides_wide=ir.dfs.get("peptides_wide"),
            peptides_centered=ir.dfs.get("peptides_centered"),
            meta_cont=meta_filtering.get("meta_cont"),
            meta_qvalue=meta_filtering.get("meta_qvalue"),
            meta_pep=meta_filtering.get("meta_pep"),
            meta_prec=meta_filtering.get("meta_prec"),
            meta_censor=meta_filtering.get("meta_censor"),
            meta_loc=meta_filtering.get("meta_loc"),
            meta_quant=ir.metadata.get("quantification"),
            raw_covariate=ir.dfs.get("raw_df_covariate"),
            lognormalized_covariate=ir.dfs.get("postlog_covariate"),
            normalized_covariate=ir.dfs.get("normalized_covariate"),
            processed_covariate=ir.dfs.get("imputed_covariate"),
            centered_covariate=ir.dfs.get("centered_covariate"),
            qvalues_covariate=ir.dfs.get("qvalues_covariate"),
            pep_covariate=ir.dfs.get("pep_covariate"),
            spectral_counts_covariate=ir.dfs.get("spectral_counts_covariate"),
            ibaq_covariate=ir.dfs.get("ibaq_covariate"),
        )

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    @log_time("Computing Missed Cleavage rates")
    def _compute_missed_cleavages_per_sample(
        self,
        df_in: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute missed-cleavage fraction per run, deduplicated at the peptide sequence level.

        Output:
          Sample | MISSED_CLEAVAGE_FRACTION

        Missed cleavage definition (tryptic):
          any K/R not followed by P, anywhere except the C-terminal residue.

        IMPORTANT:
          - This intentionally does NOT gate on SIGNAL > 0, to match workflow-side
            deduplication on peptide identifiers per run.
          - This should be called on the peptide-level long table (pre-protein rollup).
        """
        if "FILENAME" not in df_in.columns:
            raise ValueError(
                "Cannot compute missed cleavages: missing required column 'FILENAME'."
            )
        if "PEPTIDE_LSEQ" not in df_in.columns:
            raise ValueError(
                "Cannot compute missed cleavages: missing required column 'PEPTIDE_LSEQ'."
            )

        if "SIGNAL" not in df_in.columns:
            raise ValueError(
                "Cannot compute missed cleavages: missing required column 'SIGNAL'."
            )

        # In peptidomics wide inputs (e.g. FragPipe), missing values are often encoded as 0.0
        # after melting to long format. If we don't drop those rows, every sample can end up
        # with the same peptide set -> identical missed-cleavage fractions.
        # For DIA-style proteomics/phospho we keep the workflow-like behavior (no SIGNAL gating).
        at = self.analysis_type

        signal_present_filter = pl.lit(True)

        if at == "peptidomics":
            signal_present_filter = (
                pl.col("SIGNAL").is_not_null()
                & ~pl.col("SIGNAL").is_nan()
                & (pl.col("SIGNAL") > 0)
            )

        pep_id_expr = (
            expr_strip_underscores("PEPTIDE_LSEQ")
            .cast(pl.Utf8, strict=False)
            .alias("PEP_ID")
        )
        seq_expr = (
            expr_strip_bracket_mods("PEPTIDE_LSEQ")
            .cast(pl.Utf8, strict=False)
            .alias("SEQ_RAW")
        )

        tbl = (
            df_in.select(
                pl.col("FILENAME").cast(pl.Utf8).alias("Sample"),
                pep_id_expr,
                seq_expr,
                pl.col("SIGNAL").cast(pl.Float64, strict=False).alias("SIGNAL"),
            )
            .drop_nulls(["Sample", "PEP_ID"])
            .filter(signal_present_filter)
            # normalize sequence for cleavage test: keep letters only
            .with_columns(
                pl.col("SEQ_RAW")
                .cast(pl.Utf8, strict=False)
                .str.to_uppercase()
                .str.replace_all(r"[^A-Z]", "")
                .alias("SEQ")
            )
            .filter(pl.col("SEQ").str.len_chars() >= 2)
            # Sum intensity per (Sample, peptide-id) then deduplicate.
            .group_by(["Sample", "PEP_ID"], maintain_order=True)
            .agg(
                [
                    pl.first("SEQ").alias("SEQ"),
                    pl.col("SIGNAL").sum().alias("SIGNAL"),
                ]
            )
            .with_columns(
                pl.col("SEQ").str.split("").list.slice(1).alias("_chars")
            )
            .with_columns(
                pl.col("_chars").list.slice(0, pl.col("_chars").list.len() - 1).alias("_cur"),
                pl.col("_chars").list.slice(1).alias("_nxt"),
            )
            .with_columns(
                pl.struct(["_cur", "_nxt"])
                .map_elements(
                    lambda s: any(
                        (c in ("K", "R")) and (n != "P")
                        for c, n in zip(s["_cur"], s["_nxt"])
                    ),
                    return_dtype=pl.Boolean,
                )
                .alias("MISSED_CLEAVAGE")
            )
            .drop(["SEQ_RAW", "_chars", "_cur", "_nxt"], strict=False)
            .group_by("Sample")
            .agg(
                [
                    pl.col("MISSED_CLEAVAGE").cast(pl.Float64).mean().alias(
                        "MISSED_CLEAVAGE_FRACTION"
                    ),
                    (
                        (pl.col("MISSED_CLEAVAGE").cast(pl.Float64) * pl.col("SIGNAL"))
                        .sum()
                        / pl.col("SIGNAL").sum()
                    ).alias("MISSED_CLEAVAGE_FRACTION_WEIGHTED"),
                ]
            )
            .sort("Sample")
        )

        return tbl

    def _filter_block(self, df: pl.DataFrame, tag: str) -> pl.DataFrame:
        """Apply filtering steps in the same order as the original implementation."""
        x = self._filter_contaminants(df)
        self.intermediate_results.add_df(f"filtered_contaminants/{tag}", x)

        x = self._filter_by_stat(x, "QVALUE")
        self.intermediate_results.add_df(f"filtered_QVALUE/{tag}", x)

        x = self._filter_by_stat(x, "PEP")
        self.intermediate_results.add_df(f"filtered_PEP/{tag}", x)

        x = self._filter_by_num_precursors(x)
        self.intermediate_results.add_df(f"filtered_PREC/{tag}", x)

        x = self._censor_low_val(x)
        self.intermediate_results.add_df(f"filtered_final_censored/{tag}", x)

        return x

    @log_time("Filtering")
    def _filter(self, df: pl.DataFrame) -> None:
        """Apply first-pass filtering to main and (optionally) covariate blocks."""

        df_main, df_cov = self._split_main_covariate_rows(df)

        log_info("Filtering (main)")
        with log_indent():
            main_f = self._filter_block(df_main, "main")

        if df_cov is not None and len(df_cov):
            log_info("Filtering (covariate)")
            with log_indent():
                cov_f = self._filter_block(df_cov, "covariate")
        else:
            cov_f = None

        # concatenate back (same schema by construction)
        out = (
            main_f
            if cov_f is None
            else pl.concat([main_f, cov_f], how="vertical_relaxed", rechunk=True)
        )

        self.intermediate_results.add_df("filtered_final_censored", out)

    def _filter_contaminants(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.remove_contaminants or "INDEX" not in df.columns:
            return df

        all_contaminants = set()
        for path in self.remove_contaminants:
            all_contaminants |= load_contaminant_accessions(path)

        contaminants_list = list(all_contaminants)

        mask_is_contam = (
            pl.col("INDEX")
            .cast(pl.Utf8, strict=False)
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().is_in(contaminants_list))
            .list.any()
        )

        mask_keep = ~mask_is_contam

        df_kept = df.filter(mask_keep)
        df_dropped = df.filter(~mask_keep)
        mask_bools = (
            df.select(mask_keep.alias("keep")).get_column("keep").to_list()
        )

        dropped_dict = {
            "files": self.remove_contaminants,
            "values": mask_bools,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }

        self.intermediate_results.add_metadata(
            "filtering", "meta_cont", dropped_dict
        )

        return df_kept

    def _filter_by_stat(self, df: pl.DataFrame, stat: str) -> pl.DataFrame:
        """Filter by QVALUE or PEP; records both pass/fail counts and raw values."""
        if stat not in ["PEP", "QVALUE"]:
            raise ValueError(f"Invalid filtering stat '{stat}'")

        if stat not in df.columns:
            skipped = {
                "skipped": True,
                "reason": f"'{stat}' column not present",
                "threshold": None,
                "direction": "lower_or_equal"
                if stat == "QVALUE"
                else self.pep_direction.replace("higher", "greater_or_equal"),
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata(
                "filtering", f"meta_{stat.lower()}", skipped
            )
            log_info(f"{stat} filtering: skipped (column not present).")
            return df

        if stat == "QVALUE":
            threshold = self.filter_qvalue
            keep_mask = pl.col(stat) <= threshold
            direction_note = "lower_or_equal"
        else:
            threshold = self.filter_pep
            if self.pep_direction == "lower":
                keep_mask = pl.col(stat) <= threshold
                direction_note = "lower_or_equal"
            else:
                keep_mask = pl.col(stat) >= threshold
                direction_note = "greater_or_equal"

        values = df[stat].to_numpy()
        self.intermediate_results.add_array(f"{stat.lower()}_array", values)

        df_kept = df.filter(keep_mask)
        df_dropped = df.filter(~keep_mask)

        dropped_dict = {
            "threshold": threshold,
            "direction": direction_note,
            "raw_values": values,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }
        self.intermediate_results.add_metadata(
            "filtering", f"meta_{stat.lower()}", dropped_dict
        )
        log_info(
            f"{stat} filtering: kept={len(df_kept)} dropped={len(df_dropped)} "
            f"(direction={direction_note}, thr={threshold})."
        )

        return df_kept

    def _filter_by_num_precursors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out proteins with insufficient run evidence."""

        if self.filter_num_precursors is None or self.filter_num_precursors <= 0:
            skipped = {
                "skipped": True,
                "reason": "min_precursors <= 0 or None",
                "threshold": self.filter_num_precursors,
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata(
                "filtering", "meta_prec", skipped
            )
            log_info("Run-evidence filtering: skipped (threshold disabled).")
            return df

        if "PRECURSORS_EXP" not in df.columns:
            skipped = {
                "skipped": True,
                "reason": "'PRECURSORS_EXP' column not present",
                "threshold": None,
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata(
                "filtering", "meta_prec", skipped
            )
            log_info("Run-evidence filtering: skipped (column not present).")
            return df

        values = df["PRECURSORS_EXP"].to_numpy()

        # all-null column → skip to avoid dropping the entire block
        if df.select(pl.col("PRECURSORS_EXP").is_null().all()).item():
            skipped = {
                "skipped": True,
                "reason": "'PRECURSORS_EXP' all null",
                "threshold": self.filter_num_precursors,
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata(
                "filtering", "meta_prec", skipped
            )
            log_info("Run-evidence filtering: skipped (all-null PRECURSORS_EXP).")
            return df

        self.intermediate_results.add_array("num_precursors_array", values)

        df_kept = df.filter(
            pl.col("PRECURSORS_EXP") >= self.filter_num_precursors
        )
        df_dropped = df.filter(
            pl.col("PRECURSORS_EXP") < self.filter_num_precursors
        )
        dropped_dict = {
            "threshold": self.filter_num_precursors,
            "raw_values": values,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }

        self.intermediate_results.add_metadata(
            "filtering", "meta_prec", dropped_dict
        )
        log_info(
            f"Precursors filtering: kept={len(df_kept)} dropped={len(df_dropped)} "
            f"(thr={self.filter_num_precursors})."
        )

        return df_kept

    def _censor_low_val(self, df: pl.DataFrame) -> pl.DataFrame:
        """Set very low intensity values to NA and log the counts."""
        if self.min_linear_intensity is None:
            return df

        thr = self.min_linear_intensity

        mask_censor = (pl.col("SIGNAL") < thr) & pl.col("SIGNAL").is_not_null()
        mask_count = mask_censor

        # In peptidomics wide inputs (e.g. FragPipe), missing can be encoded as 0.0 after melt.
        # Those zeros are “missing”, not “low-but-present”, so exclude them from censor *counts* only.
        if self.analysis_type == "peptidomics" and "INPUT_LAYOUT" in df.columns:
            is_wide = (
                df.select(pl.col("INPUT_LAYOUT").cast(pl.Utf8, strict=False).first())
                .item()
                == "wide"
            )
            if is_wide:
                mask_count = mask_count & (pl.col("SIGNAL") != 0)

        raw_vals_np = df["SIGNAL"].to_numpy()
        will_censor = int(df.select(mask_count.sum()).item())
        df_kept = len(df) - will_censor

        df = df.with_columns(
            pl.when(mask_censor).then(None).otherwise(pl.col("SIGNAL")).alias("SIGNAL")
        )

        log_info(
            f"Low-intensity censoring: kept={df_kept}, dropped={will_censor} "
            f"(thr={thr})."
        )
        self.intermediate_results.add_metadata(
            "filtering",
            "meta_censor",
            {
                "threshold": thr,
                "number_dropped": will_censor,
                "number_kept": df_kept,
                "raw_values": raw_vals_np,
            },
        )

        return df

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def _get_protein_metadata(self) -> None:
        df_full = self.intermediate_results.dfs[
            "filtered_final_censored"
        ]  # long format, post filter

        if "INDEX" not in df_full.columns:
            raise ValueError(
                "Protein metadata extraction requires column 'INDEX'. "
                "Your input did not produce it during harmonization. "
                "For wide proteomics inputs, set dataset.index_column or ensure UNIPROT is available for fallback."
            )

        keep_cols = {
            "INDEX",
            "FASTA_HEADERS",
            "GENE_NAMES",
            "PROTEIN_DESCRIPTIONS",
            "PROTEIN_WEIGHT",
            "IBAQ",
            "PRECURSORS_EXP",
        }
        if self.analysis_type == "phospho":
            keep_cols.update({"PARENT_PEPTIDE_ID", "PARENT_PROTEIN"})
        if self.analysis_type != "proteomics":
            keep_cols.update({"UNIPROT"})

        existing = [c for c in df_full.columns if c in keep_cols]

        base_meta = (
            df_full.select(existing)
            .group_by("INDEX")
            .agg([pl.first(c).alias(c) for c in existing if c != "INDEX"])
            .sort("INDEX")
        )

        # Fill missing/blank gene names with UniProt ID (INDEX)
        if "GENE_NAMES" in base_meta.columns:
            base_meta = base_meta.with_columns(
                pl.coalesce(
                    [
                        pl.col("GENE_NAMES")
                        .cast(pl.Utf8, strict=False)
                        .str.strip_chars()
                        .replace(["", "NA", "NaN", "nan"], None),
                        pl.col("INDEX").cast(pl.Utf8),
                    ]
                ).alias("GENE_NAMES")
            )

        # Number of precursors used (unique sequence+charge)
        if {"PEPTIDE_LSEQ", "CHARGE"}.issubset(df_full.columns):
            prec_used = (
                df_full.select(["INDEX", "PEPTIDE_LSEQ", "CHARGE"])
                .drop_nulls(["PEPTIDE_LSEQ", "CHARGE"])
                .with_columns(
                    (
                        pl.col("PEPTIDE_LSEQ").cast(pl.Utf8, strict=False)
                        + pl.lit("/")
                        + pl.col("CHARGE").cast(pl.Utf8)
                    ).alias("PREC_KEY")
                )
                .unique(subset=["INDEX", "PREC_KEY"], maintain_order=True)
                .group_by("INDEX")
                .agg(pl.len().alias("PRECURSORS_USED"))
            )
            base_meta = base_meta.join(prec_used, on="INDEX", how="left")

        # IBAQ mean across runs
        if "IBAQ" in df_full.columns:
            ibaq_mean = (
                df_full.select(["INDEX", "IBAQ"])
                .with_columns(
                    pl.col("IBAQ")
                    .cast(pl.Utf8, strict=False)
                    .str.split(";")
                    .list.eval(pl.element().cast(pl.Float64, strict=False))
                    .list.mean()
                    .alias("IBAQ")
                )
                .group_by("INDEX")
                .agg(pl.col("IBAQ").mean().alias("IBAQ"))
            )
            base_meta = (
                base_meta.drop("IBAQ", strict=False)
                .join(ibaq_mean, on="INDEX", how="left")
            )

        casts = []
        if "IBAQ" in base_meta.columns:
            casts.append(pl.col("IBAQ").cast(pl.Float64, strict=False))
        if "PROTEIN_WEIGHT" in base_meta.columns:
            casts.append(
                pl.col("PROTEIN_WEIGHT").cast(pl.Float64, strict=False)
            )
        if "PRECURSORS_EXP" in base_meta.columns:
            casts.append(
                pl.col("PRECURSORS_EXP").cast(pl.Int64, strict=False)
            )
        if "PRECURSORS_USED" in base_meta.columns:
            casts.append(
                pl.col("PRECURSORS_USED").cast(pl.Int64, strict=False)
            )

        if casts:
            base_meta = base_meta.with_columns(casts)

        self.intermediate_results.add_df("protein_metadata", base_meta)

    def _get_condition_map(self):
        """Build sample→condition/replicate map, with sanitized condition and ALIGN_KEY."""

        df = self.intermediate_results.dfs["filtered_final_censored"]
        base_cols = ["FILENAME", "CONDITION", "REPLICATE"]
        if "ASSAY" in df.columns:
            base_cols.append("ASSAY")
        if "IS_COVARIATE" in df.columns:
            base_cols.append("IS_COVARIATE")

        condition_mapping = (
            df.select(base_cols)
            .unique()
            .rename({"FILENAME": "Sample"})
            .sort("Sample")
        )

        # aggregate IS_COVARIATE per Sample if present
        if "IS_COVARIATE" in condition_mapping.columns:
            condition_mapping = (
                condition_mapping.group_by(
                    "Sample", maintain_order=True
                ).agg(
                    [
                        pl.first("CONDITION"),
                        pl.first("REPLICATE"),
                        pl.first("ASSAY").alias("ASSAY")
                        if "ASSAY" in condition_mapping.columns
                        else pl.lit(None).alias("ASSAY"),
                        pl.col("IS_COVARIATE").max().alias("IS_COVARIATE"),
                    ]
                )
            )

        def _sanitize_condition(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
            if re.match(r"^[0-9]", s or ""):
                s = "C_" + s
            return s or "C_UNLABELED"

        condition_mapping = condition_mapping.with_columns(
            pl.col("CONDITION").alias("CONDITION_ORIG")
        )
        condition_mapping = condition_mapping.with_columns(
            pl.col("CONDITION")
            .cast(pl.Utf8, strict=False)
            .map_elements(_sanitize_condition, return_dtype=pl.Utf8)
            .alias("CONDITION")
        )

        mapping_tmp = condition_mapping
        using_covariate = "IS_COVARIATE" in mapping_tmp.columns and mapping_tmp[
            "IS_COVARIATE"
        ].any()

        if using_covariate:
            if "REPLICATE" not in mapping_tmp.columns:
                raise ValueError(
                    "Covariate alignment requires a REPLICATE column in both main and covariate runs."
                )
            if mapping_tmp.select(
                pl.col("REPLICATE").is_null().any()
            ).item():
                raise ValueError(
                    "Covariate alignment: REPLICATE contains null/missing values."
                )
            try:
                mapping_tmp = mapping_tmp.with_columns(
                    pl.col("REPLICATE").cast(pl.Int64)
                )
            except Exception:
                raise ValueError(
                    "Covariate alignment: REPLICATE must be numeric (e.g., 1,2,3)."
                )

        mapping_tmp = mapping_tmp.with_columns(
            (
                pl.col("CONDITION")
                + pl.lit("#R")
                + pl.col("REPLICATE").cast(pl.Utf8)
            ).alias("ALIGN_KEY")
        )

        if using_covariate:
            main = mapping_tmp.filter(pl.col("IS_COVARIATE") == False)
            cov = mapping_tmp.filter(pl.col("IS_COVARIATE") == True)
            if main.height and cov.height:
                rep_main = (
                    main.group_by("CONDITION")
                    .agg(pl.len().alias("N"))
                    .sort("CONDITION")
                )
                rep_cov = (
                    cov.group_by("CONDITION")
                    .agg(pl.len().alias("N"))
                    .sort("CONDITION")
                )
                cmp = rep_main.join(
                    rep_cov, on="CONDITION", how="outer", suffix="_FT"
                )
                # Strict parity checks:
                # 1) CONDITION levels must match between main and covariate.
                # 2) Replicate counts per CONDITION must match.
                # NOTE: comparisons against NULL yield NULL in Polars and would silently bypass filters;
                # therefore we validate NULL explicitly before comparing.
                missing_main = cmp.filter(pl.col("N").is_null()).select(pl.col("CONDITION_FT").alias("CONDITION"))
                missing_cov = cmp.filter(pl.col("N_FT").is_null()).select("CONDITION")
                if missing_main.height or missing_cov.height:
                    raise ValueError(
                        "Main/covariate CONDITION levels do not match. "
                        f"Only-in-covariate={missing_main.get_column('CONDITION').to_list()} "
                        f"Only-in-main={missing_cov.get_column('CONDITION').to_list()}. "
                        "Ensure covariate uses the same CONDITION labels as the main dataset."
                    )

                bad = cmp.filter(pl.col("N") != pl.col("N_FT"))
                if bad.height:
                    raise ValueError(
                        "Covariate has different replicate counts per CONDITION than the main dataset. "
                        f"Mismatch table:\n{bad}"
                    )

        self.intermediate_results.add_df("condition_pivot", mapping_tmp)

        if (
            not self.covariate_assays
            and "IS_COVARIATE" in condition_mapping.columns
        ):
            cov = (
                condition_mapping.filter(pl.col("IS_COVARIATE") == True)
                .select("ASSAY")
                .drop_nulls()
                .unique()
                .to_series()
                .to_list()
            )
            self.covariate_assays = [str(a).lower() for a in cov]
            if self.covariate_assays:
                log_info(
                    f"Covariates inferred from injected runs: assays={self.covariate_assays}"
                )

    # -------------------------------------------------------------------------
    # Pivot helpers
    # -------------------------------------------------------------------------

    # using a trick because polars>=1.31 does aggregate filling with 0s ! Instead of nan !
    def _pivot_df(
        self,
        df: pl.DataFrame,
        sample_col: str,
        protein_col: str,
        values_col: str,
        aggregate_fn: str,
    ) -> pl.DataFrame:
        """Pivot helper: long → wide (index=protein_col, columns=sample_col).

        Aggregates by (protein, sample) using the requested function, preserving nulls
        when a group has no valid values.
        """
        fn_map = {
            "sum": pl.col(values_col).sum(),
            "mean": pl.col(values_col).mean(),
            "min": pl.col(values_col).min(),
            "max": pl.col(values_col).max(),
            "median": pl.col(values_col).median(),
            "count": pl.count(),
        }
        if aggregate_fn not in fn_map:
            raise ValueError(f"Unsupported aggregate_fn '{aggregate_fn}'")

        agg_expr = fn_map[aggregate_fn].alias(values_col)
        n_valid = pl.col(values_col).is_not_null().sum().alias("_NVALID")

        df_agg = (
            df.group_by([protein_col, sample_col], maintain_order=True)
            .agg([agg_expr, n_valid])
            .with_columns(
                pl.when(pl.col("_NVALID") == 0)
                .then(pl.lit(None))
                .otherwise(pl.col(values_col))
                .alias(values_col)
            )
            .drop("_NVALID")
        )

        pivot_df = df_agg.pivot(
            index=protein_col, columns=sample_col, values=values_col
        )
        pivot_df = pivot_df.fill_null(np.nan)

        return pivot_df

    @log_time("Top 3")
    def _pivot_df_top_n(
        self,
        df: pl.DataFrame,
        sample_col: str,
        protein_col: str,
        values_col: str,
        peptide_col: str = "PEPTIDE_LSEQ",
        n: int = 3,
    ) -> pl.DataFrame:
        """
        Pivot helper for Top-N roll-up at peptide level.

        For each (protein, sample):
          1) aggregate intensity per cleaned peptide sequence
          2) rank peptides by intensity (descending)
          3) sum the top-N peptides
        Returns a protein × sample matrix (same layout as _pivot_df).
        """
        if peptide_col not in df.columns:
            raise ValueError(
                f"Top{n} quantification requires column '{peptide_col}' in input DataFrame."
            )

        seq_clean = expr_peptide_index_seq(
            peptide_col,
            collapse_met_oxidation=self.collapse_met_oxidation,
            collapse_all_ptms=self.collapse_all_ptms,
        ).alias("PEPTIDE_SEQ")

        base = (
            df.select(
                pl.col(protein_col),
                pl.col(sample_col),
                pl.col(values_col),
                seq_clean,
            )
            .with_columns(
                pl.col(values_col)
                .cast(pl.Float64, strict=False)
                .alias(values_col)
            )
        )

        agg = (
            base.group_by(
                [protein_col, sample_col, "PEPTIDE_SEQ"], maintain_order=True
            )
            .agg(
                [
                    pl.col(values_col).sum().alias(values_col),
                    pl.col(values_col)
                    .is_not_null()
                    .sum()
                    .alias("_NVALID"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_NVALID") == 0)
                .then(pl.lit(None))
                .otherwise(pl.col(values_col))
                .alias(values_col)
            )
            .drop("_NVALID")
        )

        ranked = (
            agg.drop_nulls(values_col)
            .with_columns(
                pl.col(values_col)
                .rank(method="dense", descending=True)
                .over([protein_col, sample_col])
                .alias("_RANK")
            )
            .filter(pl.col("_RANK") <= n)
        )

        topn = (
            ranked.group_by(
                [protein_col, sample_col], maintain_order=True
            ).agg(pl.col(values_col).sum().alias(values_col))
        )

        pivot_df = topn.pivot(
            index=protein_col, columns=sample_col, values=values_col
        )

        pivot_df = pivot_df.fill_null(np.nan)
        return pivot_df

    @log_time("DirectLFQ")
    def _pivot_df_LFQ(
        self,
        df: pl.DataFrame,
        sample_col: str,
        protein_col: str,
        values_col: str,
        ion_col: str,
    ) -> pl.DataFrame:
        """Build protein-level LFQ matrix using directLFQ."""
        for c in (sample_col, protein_col, values_col, ion_col):
            if c not in df.columns:
                raise ValueError(
                    f"LFQ requires column '{c}' in input DataFrame."
                )

        is_valid = pl.col(values_col).is_not_null() & (pl.col(values_col) != 0)

        df_ion_agg = (
            df.group_by([protein_col, ion_col, sample_col], maintain_order=True)
            .agg(
                [
                    pl.col(values_col)
                    .filter(pl.col(values_col) > 0)
                    .sum()
                    .alias(values_col),
                    (
                        pl.col(values_col)
                        .is_not_null()
                        & (pl.col(values_col) > 0)
                    )
                    .sum()
                    .alias("_NVALID"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_NVALID") == 0)
                .then(pl.lit(None))
                .otherwise(pl.col(values_col))
                .alias(values_col)
            )
            .drop("_NVALID")
        )

        has_valid = (
            df.group_by([protein_col, sample_col], maintain_order=True)
            .agg(is_valid.any().alias("_HAS_VALID"))
            .pivot(index=protein_col, columns=sample_col, values="_HAS_VALID")
            .fill_null(False)
        )

        runs_in_order = (
            df.select(sample_col)
            .unique(maintain_order=True)
            .to_series()
            .to_list()
        )

        pep_wide = (
            df_ion_agg.pivot(
                index=[protein_col, ion_col],
                columns=sample_col,
                values=values_col,
            )
            .fill_null(np.nan)
        )

        pw = pep_wide.to_pandas().set_index([protein_col, ion_col]).sort_index(
            level=0
        )
        pw = np.log2(pw)  # NaNs stay NaNs

        dlcfg.PROTEIN_ID = protein_col
        dlcfg.QUANT_ID = ion_col
        prot_df, _ = estimate_protein_intensities(
            normed_df=pw,
            min_nonan=self.directlfq_min_nonan,
            num_samples_quadratic=2,
            num_cores=self.directlfq_cores,
        )
        if protein_col not in prot_df.columns:
            prot_df = prot_df.reset_index()
        prot_df = prot_df.set_index(protein_col).astype(float)

        prot_df = prot_df.replace(0.0, np.nan)

        mask_pd = has_valid.to_pandas().set_index(protein_col)
        mask_pd = mask_pd.reindex(
            index=prot_df.index, columns=prot_df.columns, fill_value=False
        )
        prot_df = prot_df.mask(~mask_pd)

        out = pl.DataFrame(prot_df.reset_index())
        keep_cols = [protein_col] + [c for c in runs_in_order if c in out.columns]
        out = out.select(keep_cols)
        out = out.with_columns(
            [
                pl.when(pl.col(c).is_nan())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in out.columns
                if c != protein_col
            ]
        )

        #log_info(
        #    "Zeros after pivot: "
        #    f"{int(out.select([(pl.col(c) == 0).sum().alias(c) for c, dt in zip(out.columns, out.dtypes) if c != protein_col and dt in pl.NUMERIC_DTYPES]).to_numpy().sum())}"
        #)

        return out

    # -------------------------------------------------------------------------
    # Peptide & precursor tables + consistency
    # -------------------------------------------------------------------------

    def _build_peptide_tables(self) -> Optional[Tuple[pl.DataFrame, pl.DataFrame]]:
        return build_peptide_tables(self)

    # -------------------------------------------------------------------------
    # Main pivot (proteins) + covariates
    # -------------------------------------------------------------------------

    @log_time("Pivoting data")
    def _pivot_data(self) -> None:
        """Convert long format to wide protein-level matrices; build peptide tables."""

        df = self.intermediate_results.dfs["filtered_final_censored"]

        _require_columns(
            df,
            {"INDEX", "FILENAME", "SIGNAL"},
            context="Pivoting",
        )

        df_main, df_cov = self._split_main_covariate_rows(df)

        main_piv = self._pivot_main_block(df_main)

        intensity_pivot = main_piv["intensity"]
        qvalue_pivot = main_piv["qv"]
        pep_pivot = main_piv["pep"]
        prec_pivot = main_piv["prec"]
        sc_pivot = main_piv["sc"]
        locprob_pivot = main_piv["lp"]
        ibaq_pivot = main_piv["ibaq"]

        # Peptide drill-down
        pep_tables = self._build_peptide_tables()
        if pep_tables is not None:
            raw_pep_pivot, centered_pep_pivot = pep_tables
        else:
            raw_pep_pivot = None
            centered_pep_pivot = None

        # Align secondary pivots to intensity order
        order_idx = intensity_pivot.select("INDEX")

        def _align_to_intensity(pvt: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
            if pvt is None:
                return None
            return order_idx.join(pvt, on="INDEX", how="left")

        qvalue_pivot = _align_to_intensity(qvalue_pivot)
        pep_pivot = _align_to_intensity(pep_pivot)
        sc_pivot = _align_to_intensity(sc_pivot)
        prec_pivot = _align_to_intensity(prec_pivot)
        locprob_pivot = _align_to_intensity(locprob_pivot)
        ibaq_pivot = _align_to_intensity(ibaq_pivot)

        filtered_intensity, qvalue_pivot, pep_pivot, prec_pivot, sc_pivot, locprob_pivot, ibaq_pivot = (
            self._apply_phospho_localization_filter(
                intensity_pivot=intensity_pivot,
                qvalue_pivot=qvalue_pivot,
                pep_pivot=pep_pivot,
                prec_pivot=prec_pivot,
                sc_pivot=sc_pivot,
                locprob_pivot=locprob_pivot,
                ibaq_pivot=ibaq_pivot,
            )
        )

        ir = self.intermediate_results
        ir.set_columns_and_index(filtered_intensity)
        ir.add_df("raw_df", filtered_intensity)

        ir.add_df("qvalues", qvalue_pivot)
        ir.add_df("pep", pep_pivot)
        ir.add_df("locprob", locprob_pivot)
        ir.add_df("spectral_counts", sc_pivot)
        ir.add_df("ibaq", ibaq_pivot)
        ir.dfs["peptides_wide"] = raw_pep_pivot
        ir.dfs["peptides_centered"] = centered_pep_pivot
        ir.add_metadata("quantification", "method", self.protein_rollup_method)
        ir.add_metadata("quantification", "directlfq_min_nonan", self.directlfq_min_nonan)

        # Covariate pivots and broadcast
        with log_indent():
            if self.covariates_enabled:
                log_info("Covariate broadcast/alignment")
                if df_cov is None or df_cov.height == 0:
                    raise ValueError(
                        "Covariates enabled but no covariate rows are present after filtering."
                    )

                key_col = {
                    "parent_peptide": "PARENT_PEPTIDE_ID",
                    "parent_protein": "PARENT_PROTEIN",
                }[self.cov_align_on]

                main_key_map = df_main.select(["INDEX", key_col]).unique()
                cov_key_map = df_cov.select(["INDEX", key_col]).unique()

                if key_col not in main_key_map.columns or key_col not in cov_key_map.columns:
                    raise ValueError(
                        f"Covariate alignment key '{key_col}' not available in input."
                    )

                log_info(
                    f"Covariate Quantification using {self.covariate_protein_rollup_method}"
                )

                cov_int_by_key = self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "SIGNAL", self.covariate_protein_rollup_method)
                cov_qv_by_key = self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "QVALUE", "mean") if "QVALUE" in df_cov.columns else None
                cov_pep_by_key = self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "PEP", "mean") if "PEP" in df_cov.columns else None

                cov_prec_by_key = (
                    self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "PRECURSORS_EXP", "mean")
                    if "PRECURSORS_EXP" in df_cov.columns
                    else None
                )
                cov_sc_by_key = (
                    self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "SPECTRAL_COUNTS", "max")
                    if "SPECTRAL_COUNTS" in df_cov.columns
                    else None
                )
                cov_ibaq_by_key = (
                    self._pivot_cov_by_key(df_cov, cov_key_map, key_col, "IBAQ", "mean")
                    if "IBAQ" in df_cov.columns
                    else None
                )

                order_idx = filtered_intensity.select("INDEX")
                main_with_key = order_idx.join(main_key_map, on="INDEX", how="left")

                def _broadcast(
                    by_key: Optional[pl.DataFrame],
                ) -> Optional[pl.DataFrame]:
                    if by_key is None:
                        return None
                    out = main_with_key.join(by_key, on=key_col, how="left")
                    return out.select(
                        ["INDEX"]
                        + [
                            c
                            for c in out.columns
                            if c not in ("INDEX", key_col)
                        ]
                    )

                cov_int = _broadcast(cov_int_by_key)
                cov_qv = _broadcast(cov_qv_by_key)
                cov_pep = _broadcast(cov_pep_by_key)
                cov_prec = _broadcast(cov_prec_by_key)
                cov_sc = _broadcast(cov_sc_by_key)
                cov_ibaq = _broadcast(cov_ibaq_by_key)

                ir.add_df(
                    "cov_index_to_key",
                    main_with_key.select(["INDEX", key_col]),
                )
                ir.add_metadata("covariate", "key_col", key_col)

                ir.add_df("raw_df_covariate", cov_int)
                ir.add_df("qvalues_covariate", cov_qv)
                ir.add_df("pep_covariate", cov_pep)
                # NOTE: Covariate "spectral counts" can be either a true SPECTRAL_COUNTS column
                # (vendor-provided) or, in many datasets, effectively equivalent to precursor evidence.
                # Prefer the explicit SPECTRAL_COUNTS pivot when available; otherwise fall back to
                # precursor evidence to preserve historical behavior.
                ir.add_df("spectral_counts_covariate", cov_sc if cov_sc is not None else cov_prec)
                ir.add_df("ibaq_covariate", cov_ibaq)
                log_info(
                    f"Built covariate pivot block - broadcast on {key_col}; shape matches main."
                )

    def _pivot_main_block(self, df: pl.DataFrame) -> Dict[str, Optional[pl.DataFrame]]:
        """Mechanical extraction of the original per-block pivot logic."""
        method = (self.protein_rollup_method or "sum").lower()
        log_info(f"Quantification using {method}")
        if method == "directlfq":
            intensity = self._pivot_df_LFQ(
                df=df,
                sample_col="FILENAME",
                protein_col="INDEX",
                values_col="SIGNAL",
                ion_col="PEPTIDE_LSEQ",
            )
        elif method in {"top3", "top_3"}:
            intensity = self._pivot_df_top_n(
                df=df,
                sample_col="FILENAME",
                protein_col="INDEX",
                values_col="SIGNAL",
                peptide_col="PEPTIDE_LSEQ",
                n=3,
            )
        else:
            intensity = self._pivot_df(
                df=df,
                sample_col="FILENAME",
                protein_col="INDEX",
                values_col="SIGNAL",
                aggregate_fn=method,
            )

        qv = pp = sc = prec = lp = None
        if "QVALUE" in df.columns:
            qv = self._pivot_df(df, "FILENAME", "INDEX", "QVALUE", "mean")
        if "PEP" in df.columns:
            pp = self._pivot_df(df, "FILENAME", "INDEX", "PEP", "mean")

        if self.analysis_type == "phospho":
            df_prec = (
                df.select(["INDEX", "FILENAME", "PEPTIDE_LSEQ", "CHARGE"])
                .drop_nulls(["PEPTIDE_LSEQ", "CHARGE"])
                .with_columns(
                    pl.concat_str(
                        [pl.col("PEPTIDE_LSEQ"), pl.lit("/"), pl.col("CHARGE").cast(pl.Utf8)]
                    ).alias("PREC_KEY")
                )
                .unique(subset=["INDEX", "FILENAME", "PREC_KEY"], maintain_order=True)
                .group_by(["INDEX", "FILENAME"], maintain_order=True)
                .agg(pl.len().alias("N_UNIQ_PREC"))
            )
            sc = self._pivot_df(df_prec, "FILENAME", "INDEX", "N_UNIQ_PREC", "max").fill_nan(0)
        elif "SPECTRAL_COUNTS" in df.columns:
            sc = self._pivot_df(df, "FILENAME", "INDEX", "SPECTRAL_COUNTS", "max")

        if "PRECURSORS_EXP" in df.columns:
            if self.analysis_type == "phospho":
                prec = self._pivot_df(
                    df, "FILENAME", "INDEX", "PRECURSORS_EXP", "max"
                )
            else:
                prec = self._pivot_df(
                    df, "FILENAME", "INDEX", "PRECURSORS_EXP", "mean"
                )

        if "LOC_PROB" in df.columns:
            lp = self._pivot_df(df, "FILENAME", "INDEX", "LOC_PROB", "max")

        ibaq = None
        if "IBAQ" in df.columns and "INDEX" in df.columns:
            df2 = df.with_columns(
                pl.col("IBAQ")
                .cast(pl.Utf8, strict=False)
                .str.split(";")
                .list.eval(pl.element().cast(pl.Float64, strict=False))
                .list.mean()
                .alias("IBAQ")
            )
            ibaq = self._pivot_df(df2, "FILENAME", "INDEX", "IBAQ", "mean")

        return {
            "intensity": intensity,
            "qv": qv,
            "pep": pp,
            "sc": sc,
            "prec": prec,
            "lp": lp,
            "ibaq": ibaq,
        }

    def _apply_phospho_localization_filter(
        self,
        *,
        intensity_pivot: pl.DataFrame,
        qvalue_pivot: Optional[pl.DataFrame],
        pep_pivot: Optional[pl.DataFrame],
        prec_pivot: Optional[pl.DataFrame],
        sc_pivot: Optional[pl.DataFrame],
        locprob_pivot: Optional[pl.DataFrame],
        ibaq_pivot: Optional[pl.DataFrame],
    ) -> tuple[pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """Mechanical extraction of the original phospho localization filter block."""
        filtered_intensity = intensity_pivot

        with log_indent():
            if self.analysis_type == "phospho" and (locprob_pivot is not None):
                log_info("Filtering (phospho localization)")
                cols = [c for c in locprob_pivot.columns if c != "INDEX"]
                lp_mat = locprob_pivot.select(cols).to_numpy()
                if self.phospho_loc_mode == "soft":
                    keep = np.nanmax(lp_mat, axis=1) >= self.phospho_loc_thr
                else:
                    keep = np.nanmin(lp_mat, axis=1) >= self.phospho_loc_thr
                keep = np.asarray(keep, dtype=bool)

                kept_n = int(keep.sum())
                total_n = int(lp_mat.shape[0])
                dropped_dict = {
                    "mode": self.phospho_loc_mode,
                    "threshold": self.phospho_loc_thr,
                    "number_kept": kept_n,
                    "number_dropped": int(total_n - kept_n),
                }
                self.intermediate_results.add_metadata(
                    "filtering", "meta_loc", dropped_dict
                )
                log_info(
                    f"Phospho localization filter ({self.phospho_loc_mode}, thr={self.phospho_loc_thr}): "
                    f"kept={kept_n}/{total_n}."
                )

                def _row_filter(pvt: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
                    if pvt is None:
                        return None
                    idx = pvt["INDEX"].to_numpy()
                    idx_keep = idx[keep]
                    return pvt.filter(pl.col("INDEX").is_in(idx_keep.tolist()))

                filtered_intensity = _row_filter(intensity_pivot)
                qvalue_pivot = _row_filter(qvalue_pivot)
                pep_pivot = _row_filter(pep_pivot)
                prec_pivot = _row_filter(prec_pivot)
                sc_pivot = _row_filter(sc_pivot)
                locprob_pivot = _row_filter(locprob_pivot)
                ibaq_pivot = _row_filter(ibaq_pivot)
            elif self.analysis_type == "phospho":
                skipped = {
                    "skipped": True,
                    "reason": "locprob matrix not available",
                    "mode": self.phospho_loc_mode,
                    "threshold": self.phospho_loc_thr,
                    "number_kept": len(intensity_pivot),
                    "number_dropped": 0,
                }
                self.intermediate_results.add_metadata(
                    "filtering", "meta_loc", skipped
                )

        return filtered_intensity, qvalue_pivot, pep_pivot, prec_pivot, sc_pivot, locprob_pivot, ibaq_pivot

    def _pivot_cov_by_key(
        self,
        frame: Optional[pl.DataFrame],
        cov_key_map: pl.DataFrame,
        key_col: str,
        value_col: str,
        agg: str,
    ) -> Optional[pl.DataFrame]:
        """Mechanical extraction of the original covariate pivot-by-key helper."""
        if frame is None or value_col not in frame.columns:
            return None

        long_with_key = frame.join(cov_key_map, on="INDEX", how="left").drop_nulls([key_col])

        if value_col == "IBAQ":
            long_with_key = long_with_key.with_columns(
                pl.col("IBAQ")
                .cast(pl.Utf8, strict=False)
                .str.split(";")
                .list.eval(pl.element().cast(pl.Float64, strict=False))
                .list.mean()
                .alias("IBAQ")
            )

        if value_col == "SIGNAL" and self.covariate_protein_rollup_method == "directlfq":
            if "PEPTIDE_LSEQ" not in long_with_key.columns:
                raise ValueError(
                    "Covariate LFQ requested but 'PEPTIDE_LSEQ' is missing in covariate runs. "
                    "Either provide peptide sequences in the covariate data or set "
                    "preprocessing.protein_rollup_method='sum'."
                )
            return self._pivot_df_LFQ(
                df=long_with_key.select([key_col, "FILENAME", "PEPTIDE_LSEQ", value_col]),
                sample_col="FILENAME",
                protein_col=key_col,
                values_col=value_col,
                ion_col="PEPTIDE_LSEQ",
            )

        return self._pivot_df(
            df=long_with_key.select([key_col, "FILENAME", value_col]),
            sample_col="FILENAME",
            protein_col=key_col,
            values_col=value_col,
            aggregate_fn=agg,
        )
    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    @log_time("Normalization")
    def _normalize(self) -> None:
        """Apply configured normalization(s) to main and (optionally) covariate blocks."""
        with log_indent():
            self._normalize_matrix(Block.MAIN)

        if self.covariates_enabled:
            log_info("Normalization (covariate)")
            with log_indent():
                self._normalize_matrix(Block.COV)

    def _normalize_matrix(self, block: Block) -> None:
        """Normalize a matrix (main or covariate block)."""

        normalization_method = self.normalization.get("method")

        if block is Block.COV:
            df = self.intermediate_results.dfs.get("raw_df_covariate")
            if df is None:
                return
            numeric_cols = [c for c in df.columns if c != "INDEX"]
            log_info("Normalization (covariate block)")
        else:
            df = self.intermediate_results.dfs.get("raw_df")
            numeric_cols = self.intermediate_results.columns
            log_info("Normalization (main block)")

        mat = df.select(numeric_cols).to_numpy()
        original_mat = mat.copy()
        postlog_mat = mat.copy()
        postlog_df = df.clone()
        models = None
        regression_scale_used = None
        regression_type_used = None
        tags = None
        tag_matches = None

        if isinstance(normalization_method, str):
            normalization_method = [normalization_method]

        for method in normalization_method:
            m = (method or "").lower()
            if m == "log10":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log10(mat)
                postlog_mat = mat.copy()
            elif m == "log2":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log2(mat)
                postlog_mat = mat.copy()
            elif m == "median_equalization":
                global_median = np.nanmedian(mat)
                medians = np.nanmedian(mat, axis=0, keepdims=True)
                mat = mat * (global_median / medians)
            elif m == "quantile":
                qt = sklearn.preprocessing.QuantileTransformer(random_state=42)
                mat = qt.fit_transform(mat)
            elif m in {
                "global_linear",
                "global_loess",
                "local_linear",
                "local_loess",
            }:
                regression_scale_used = (
                    "global" if "global" in m else "local"
                )
                regression_type_used = "loess" if "loess" in m else "linear"

                sample_names = numeric_cols
                cond_df = (
                    self.intermediate_results.dfs["condition_pivot"].to_pandas()
                )
                cond_df.columns = [c.capitalize() for c in cond_df.columns]
                cond_map = cond_df.set_index("Sample")["Condition"]

                condition_labels = [cond_map.get(s, s) for s in sample_names]

                mat, models = regression_normalization(
                    mat,
                    scale=regression_scale_used,
                    regression_type=regression_type_used,
                    span=self.normalization.get("loess_span"),
                    condition_labels=condition_labels,
                )
            elif m == "median_equalization_by_tag":
                tag_value = self.normalization.get("reference_tag")
                fasta_col = self.normalization.get(
                    "fasta_column", "FASTA_HEADERS"
                )

                if tag_value is None or (
                    isinstance(tag_value, str) and not tag_value.strip()
                ):
                    raise ValueError(
                        "median_equalization_by_tag requires normalization.reference_tag"
                    )

                tags = (
                    [tag_value]
                    if isinstance(tag_value, str)
                    else list(tag_value)
                )
                tags = [
                    t.strip()
                    for t in tags
                    if isinstance(t, str) and t.strip()
                ]

                meta_df = self.intermediate_results.dfs.get("protein_metadata")
                if meta_df is None or fasta_col not in meta_df.columns:
                    raise ValueError(
                        "median_equalization_by_tag requires protein_metadata with "
                        f"column '{fasta_col}'"
                    )

                idx_col = "INDEX"
                fasta_map = meta_df.select([idx_col, fasta_col])
                df_with_fasta = df.join(fasta_map, on=idx_col, how="left")

                fasta_vals = df_with_fasta.get_column(fasta_col).to_list()
                tag_lc = [t.lower() for t in tags]
                ref_mask = np.array(
                    [
                        (s is not None)
                        and any(t in str(s).lower() for t in tag_lc)
                        for s in fasta_vals
                    ],
                    dtype=bool,
                )

                n_ref = int(ref_mask.sum())
                if n_ref == 0:
                    log_info(
                        f"median_equalization_by_tag: no proteins matched tag(s) {tags} "
                        f"in '{fasta_col}'. Step skipped."
                    )
                else:
                    ref_sub = mat[ref_mask, :]
                    ref_counts = np.sum(np.isfinite(ref_sub), axis=0)

                    if np.any(ref_counts == 0):
                        sample_names = self.intermediate_results.columns
                        missing_samples = [
                            sample_names[i]
                            for i, c in enumerate(ref_counts)
                            if c == 0
                        ]
                        log_info(
                            "median_equalization_by_tag: no reference signal in samples: "
                            f"{missing_samples} → leaving those samples unscaled (factor=1)."
                        )

                    ref_col_meds = np.nanmedian(
                        ref_sub, axis=0, keepdims=True
                    )
                    ref_global = np.nanmedian(ref_sub)

                    scale = np.where(
                        np.isfinite(ref_col_meds),
                        ref_global / ref_col_meds,
                        1.0,
                    )
                    mat = mat * scale

                    self.normalization["tag_matches"] = n_ref
                    log_info(
                        f"median_equalization_by_tag: matched={n_ref} proteins by {tags} "
                        f"in '{fasta_col}'."
                    )
            elif m == "none":
                log_info("Skipping normalization (raw data)")
            else:
                raise ValueError(f"Invalid normalization method: {method}")

        df = df.with_columns(pl.DataFrame(mat, schema=numeric_cols))
        postlog_df = df.with_columns(
            pl.DataFrame(postlog_mat, schema=numeric_cols)
        )

        has_log_step = any(
            (mth or "").lower() in {"log2", "log10", "log", "ln"}
            for mth in (self.normalization.get("method") or [])
        )

        if not has_log_step:
            raw_lin = np.where(
                np.isnan(original_mat), np.nan, np.exp2(original_mat)
            )
            numeric_cols_main = self.intermediate_results.columns
            self.intermediate_results.dfs["raw_df"] = (
                self.intermediate_results.dfs["raw_df"].with_columns(
                    pl.DataFrame(raw_lin, schema=numeric_cols_main)
                )
            )
            original_mat = raw_lin
            log_info(
                "No log step detected → saving RAW DATA in linear space (assumed input log2)."
            )

        key_sfx = "" if block is Block.MAIN else "_covariate"
        ir = self.intermediate_results
        ir.add_matrix(f"raw_mat{key_sfx}", original_mat)
        ir.add_matrix(f"postlog{key_sfx}", postlog_mat)
        ir.add_matrix(f"normalized{key_sfx}", mat)
        ir.add_df(f"postlog{key_sfx}", postlog_df)
        ir.add_df(f"normalized{key_sfx}", df)
        ir.add_model("normalization", models)
        ir.add_metadata("normalization", "method", self.normalization.get("method"))
        ir.add_metadata("normalization", "regression_scale_used", regression_scale_used)
        ir.add_metadata("normalization", "regression_type_used", regression_type_used)
        ir.add_metadata("normalization", "span", self.normalization.get("loess_span"))
        ir.add_metadata("normalization", "tags", tags)
        ir.add_metadata("normalization", "tag_matches", tag_matches)

    # -------------------------------------------------------------------------
    # Imputation
    # -------------------------------------------------------------------------

    @log_time("Imputation")
    def _impute(self) -> None:
        """Impute missing values in main (and optional covariate) blocks."""
        self._impute_matrix()

    def _impute_matrix(self) -> None:
        """Apply imputation; broadcast & center covariate block when enabled."""

        def _impute_block(
            df_norm: pl.DataFrame, sample_cols: List[str], suffix: str
        ):
            not_imp = df_norm.select(sample_cols).to_numpy().copy()
            cond_map = self.intermediate_results.dfs[
                "condition_pivot"
            ].to_pandas()

            imputer = get_imputer(
                **self.imputation,
                condition_map=cond_map,
                sample_index=sample_cols,
            )
            imp_mat = not_imp if imputer is None else imputer.fit_transform(
                not_imp
            )
            out_df = df_norm.with_columns(
                pl.DataFrame(imp_mat, schema=sample_cols)
            )
            imp_only = np.where(np.isnan(not_imp), imp_mat, np.nan)
            self.intermediate_results.add_matrix(
                f"imputed_only{suffix}", imp_only
            )
            self.intermediate_results.add_matrix(f"imputed{suffix}", imp_mat)
            self.intermediate_results.add_df(f"imputed{suffix}", out_df)
            self.intermediate_results.add_metadata(
                "imputation", f"method{suffix}", self.imputation.get("method")
            )
            return out_df

        # MAIN
        df_main = self.intermediate_results.dfs["normalized"]
        cols_main = self.intermediate_results.columns
        out_main = _impute_block(df_main, cols_main, suffix="")
        log_info("Imputed (main) ready.")

        # COVARIATE
        if self.covariates_enabled:
            df_cov_norm = self.intermediate_results.dfs.get(
                "normalized_covariate"
            )
            if df_cov_norm is None or df_cov_norm.height == 0:
                return

            sample_cols = [c for c in df_cov_norm.columns if c != "INDEX"]

            idx2key = self.intermediate_results.dfs.get("cov_index_to_key")
            key_col = (
                self.intermediate_results.metadata.get("covariate", {}).get(
                    "key_col"
                )
            )

            if idx2key is None or key_col is None:
                raise ValueError(
                    "Covariate key mapping is missing. Ensure _pivot_data stored "
                    "'cov_index_to_key' and 'covariate.key_col'."
                )

            cov_with_key = df_cov_norm.join(
                idx2key, on="INDEX", how="left"
            )
            agg_exprs = [pl.first(c).alias(c) for c in sample_cols]
            cov_norm_by_key = cov_with_key.group_by(
                key_col, maintain_order=True
            ).agg(agg_exprs)

            not_imp = cov_norm_by_key.select(sample_cols).to_numpy().copy()
            cond_map = self.intermediate_results.dfs[
                "condition_pivot"
            ].to_pandas()
            imputer = get_imputer(
                **self.imputation,
                condition_map=cond_map,
                sample_index=sample_cols,
            )
            imp_key = not_imp if imputer is None else imputer.fit_transform(
                not_imp
            )
            cov_imp_by_key = cov_norm_by_key.with_columns(
                pl.DataFrame(imp_key, schema=sample_cols)
            )

            cov_imp_broadcast = idx2key.join(
                cov_imp_by_key, on=key_col, how="left"
            ).select(["INDEX"] + sample_cols)

            self.intermediate_results.add_df(
                "imputed_covariate", cov_imp_broadcast
            )
            self.intermediate_results.add_metadata(
                "imputation", "method_covariate", self.imputation.get("method")
            )
            log_info("Imputed (covariate) ready.")

            cov_mat = cov_imp_broadcast.select(sample_cols).to_numpy()
            row_meds = np.nanmedian(cov_mat, axis=1, keepdims=True)
            cov_mat_centered = cov_mat - row_meds
            centered_cov_df = cov_imp_broadcast.with_columns(
                pl.DataFrame(cov_mat_centered, schema=sample_cols)
            )
            self.intermediate_results.dfs[
                "centered_covariate"
            ] = centered_cov_df
            self.intermediate_results.matrices[
                "centered_covariate"
            ] = cov_mat_centered
            log_info(
                "Covariate centering: per-feature median across all samples applied."
            )

