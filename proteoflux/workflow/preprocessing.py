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
from proteoflux.utils.directlfq import estimate_protein_intensities
import proteoflux.utils.directlfq_config as dlcfg
from enum import Enum


class Block(str, Enum):
    MAIN = "main"
    COV = "covariate"


class Preprocessor:
    """Handles filtering, pivoting, normalization, and imputation for proteomics data."""

    available_normalization = ["zscore", "vst", "linear"]
    available_imputation = ["mean", "median", "knn", "randomforest"]

    def __init__(self, config: Optional[dict] = None):
        """Initialize from a config dict mirroring `config.yaml` sections."""

        self.intermediate_results = IntermediateResults()
        self.analysis_type = (config or {}).get("analysis_type", "").strip().lower()

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

        # Pivoting
        self.pivot_signal_method = config.get("pivot_signal_method", "sum")
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
            phospho_cfg.get("localization_mode", "filter_soft") or "filter_soft"
        ).lower()
        if _mode not in {"filter_soft", "filter_strict"}:
            raise ValueError(
                "preprocessing.phospho.localization_mode must be 'filter_soft' or 'filter_strict'"
            )
        self.phospho_loc_mode = _mode
        self.phospho_loc_thr = float(
            phospho_cfg.get("localization_threshold", 0.75)
        )
        self.covariate_pivot_method = phospho_cfg.get(
            "covariate_pivot_method", "directlfq"
        )

        # Covariates: list of assay labels to extract & center after imputation
        cov_cfg = config.get("covariates") or {}
        self.covariate_assays: List[str] = [
            str(a).lower() for a in (cov_cfg.get("assays") or [])
        ]
        self.covariates_enabled = bool(cov_cfg.get("enabled", False))

        level = str(
            phospho_cfg.get("stochio_correction_level", "protein")
        ).strip().lower()
        self.cov_align_on = "parent_protein" if level == "protein" else "parent_peptide"

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

    @log_time("Filtering")
    def _filter(self, df: pl.DataFrame) -> None:
        """Apply first-pass filtering to main and (optionally) covariate blocks."""

        if "IS_COVARIATE" in df.columns:
            # treat nulls as False
            is_cov = pl.coalesce([pl.col("IS_COVARIATE"), pl.lit(False)])
            df_cov = df.filter(is_cov)
            df_main = df.filter(~is_cov)
        else:
            # If IS_COVARIATE is absent, infer covariate rows from ASSAY matching configured assays.
            has_assay = "ASSAY" in df.columns
            assay_lc = set(a.lower() for a in (self.covariate_assays or []))
            if has_assay and assay_lc:
                df_cov = df.filter(
                    pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))
                )
                df_main = df.filter(
                    ~pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))
                )
            else:
                df_main = df
                df_cov = None

        # If covariates are enabled but missing, fail early.
        if self.covariates_enabled and (df_cov is None or len(df_cov) == 0):
            raise ValueError(
                "Covariates are enabled in config, but no covariate rows were found "
                "(check ASSAY/IS_COVARIATE or config.covariates.assays)."
            )

        log_info("  Filtering (main block)")
        if df_cov is not None and len(df_cov):
            log_info("  Filtering (covariate block)")

        def _filter_block(_df: pl.DataFrame, tag: str) -> pl.DataFrame:
            x = self._filter_contaminants(_df)
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

        log_info("Filtering (main)")
        with log_indent():
            main_f = _filter_block(df_main, "main")

        if df_cov is not None and len(df_cov):
            log_info("Filtering (covariate)")
            with log_indent():
                cov_f = _filter_block(df_cov, "covariate")
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

        mask = (pl.col("SIGNAL") < thr) & pl.col("SIGNAL").is_not_null()
        raw_vals_np = df["SIGNAL"].to_numpy()
        will_censor = int(df.select(mask.sum()).item())
        df_kept = len(df) - will_censor

        df = df.with_columns(
            pl.when(mask).then(None).otherwise(pl.col("SIGNAL")).alias("SIGNAL")
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
        if self.analysis_type != "DIA":
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

        # Detect FragPipe
        is_fp = ("SOURCE" in df_full.columns) and (
            "fragpipe"
            in df_full.select(pl.col("SOURCE").unique())
            .to_series()
            .to_list()
        )

        if is_fp and {"PEPTIDE_LSEQ", "PRECURSORS_EXP"}.issubset(df_full.columns):
            prec_df = (
                df_full.select(["INDEX", "PEPTIDE_LSEQ", "PRECURSORS_EXP"])
                .drop_nulls("PRECURSORS_EXP")
                .unique()
                .group_by("INDEX")
                .agg(
                    pl.col("PRECURSORS_EXP")
                    .sum()
                    .cast(pl.Int64)
                    .alias("PRECURSORS_EXP")
                )
            )
            base_meta = (
                base_meta.drop("PRECURSORS_EXP", strict=False)
                .join(prec_df, on="INDEX", how="left")
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
                #missing_main = cmp.filter(pl.col("N").is_null()).select("CONDITION")
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
                #bad = cmp.filter(pl.col("N") != pl.col("N_FT"))
                #if bad.height:
                #    raise ValueError(
                #        "Covariate alignment: replicate counts differ per CONDITION. "
                #        f"Details:\n{bad}"
                #    )

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

        seq_clean = (
            pl.col(peptide_col)
            .str.replace_all(r"(?i)\[Oxidation[^\]]*\]", "")
            .str.replace_all(r"^_+|_+$", "")
            .alias("PEPTIDE_SEQ")
        )

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

        log_info(
            "Zeros after pivot: "
            f"{int(out.select([(pl.col(c) == 0).sum().alias(c) for c, dt in zip(out.columns, out.dtypes) if c != protein_col and dt in pl.NUMERIC_DTYPES]).to_numpy().sum())}"
        )

        return out

    # -------------------------------------------------------------------------
    # Peptide & precursor tables + consistency
    # -------------------------------------------------------------------------

    def _build_peptide_tables(self) -> Optional[Tuple[pl.DataFrame, pl.DataFrame]]:
        """Build peptide- and precursor-level tables and per-condition consistency."""

        df = self.intermediate_results.dfs["filtered_final_censored"]

        needed = {"INDEX", "FILENAME", "SIGNAL", "PEPTIDE_LSEQ"}
        if not needed.issubset(df.columns):
            return None

        analysis = (self.analysis_type or "").strip().lower()
        precursor_only = analysis in {"peptidomics", "phospho"}

        seq_clean = (
            pl.col("PEPTIDE_LSEQ")
            .str.replace_all(r"(?i)\[Oxidation[^\]]*\]", "")
            .str.replace_all(r"^_+|_+$", "")
            .alias("PEPTIDE_SEQ")
        )

        # 1) Clean peptide sequence
        pep_wide = None
        pep_centered = None
        if not precursor_only:
            base = (
                df.select(
                    pl.col("INDEX"),
                    pl.col("FILENAME"),
                    pl.col("SIGNAL"),
                    seq_clean,
                ).with_columns(
                    (
                        pl.col("INDEX").cast(pl.Utf8)
                        + pl.lit("|")
                        + pl.col("PEPTIDE_SEQ")
                    ).alias("PEPTIDE_ID")
                )
            )

        # 2) Distributions: precursors per peptide / per protein, peptides per protein
        if not precursor_only:
            # Proteomics: peptide_id = INDEX|PEPTIDE_SEQ; precursor = peptide_id + charge
            prec_stats = (
                df.select(
                    pl.col("INDEX"),
                    seq_clean,
                    pl.col("CHARGE"),
                ).with_columns(
                    (
                        pl.col("INDEX").cast(pl.Utf8)
                        + pl.lit("|")
                        + pl.col("PEPTIDE_SEQ")
                    ).alias("PEPTIDE_ID")
                )
            )

            prec_per_pep = (
                prec_stats.select(["PEPTIDE_ID", "CHARGE"])
                .drop_nulls(["PEPTIDE_ID", "CHARGE"])
                .unique()
                .group_by("PEPTIDE_ID", maintain_order=True)
                .agg(pl.len().alias("N_PREC_PER_PEP"))
            )
            self.intermediate_results.add_array(
                "num_precursors_per_peptide",
                prec_per_pep["N_PREC_PER_PEP"].to_numpy(),
            )

            pep_per_prot = (
                base.select(["INDEX", "PEPTIDE_ID"])
                .drop_nulls(["INDEX", "PEPTIDE_ID"])
                .unique()
                .group_by("INDEX", maintain_order=True)
                .agg(pl.len().alias("N_PEP_PER_PROT"))
            )
            self.intermediate_results.add_array(
                "num_peptides_per_protein",
                pep_per_prot["N_PEP_PER_PROT"].to_numpy(),
            )

            prec_per_prot = (
                prec_stats.select(["INDEX", "PEPTIDE_ID", "CHARGE"])
                .drop_nulls(["PEPTIDE_ID", "CHARGE"])
                .unique()
                .group_by("INDEX", maintain_order=True)
                .agg(pl.len().alias("N_PREC_PER_PROT"))
            )
            self.intermediate_results.add_array(
                "num_precursors_per_protein",
                prec_per_prot["N_PREC_PER_PROT"].to_numpy(),
            )
        else:
            # peptidomics/phospho: precursor = peptide_key + charge
            analysis_lc = (self.analysis_type or "").strip().lower()

            _require_cols = {"FILENAME", "SIGNAL", "CHARGE"}
            missing = _require_cols - set(df.columns)
            if missing:
                raise ValueError(
                    f"Missing required columns for precursor distributions: {sorted(missing)!r}"
                )

            if analysis_lc == "phospho":
                # peptide key is PEPTIDE_INDEX; protein key is PARENT_PROTEIN
                req = {"PEPTIDE_INDEX", "PARENT_PROTEIN"}
                missing = req - set(df.columns)
                if missing:
                    raise ValueError(
                        f"[PHOSPHO] Missing required columns for distributions: {sorted(missing)!r}"
                    )

                base_prec = (
                    df.select(
                        pl.col("PEPTIDE_INDEX").cast(pl.Utf8).alias("PEP_KEY"),
                        pl.col("PARENT_PROTEIN").cast(pl.Utf8).alias("PROT_KEY"),
                        pl.col("CHARGE"),
                    )
                    .drop_nulls(["PEP_KEY", "PROT_KEY", "CHARGE"])
                    .unique()
                )
            else:
                # peptidomics: peptide key is INDEX (strip anything after '|'); protein key is PARENT_PROTEIN
                req = {"INDEX", "PARENT_PROTEIN"}
                missing = req - set(df.columns)
                if missing:
                    raise ValueError(
                        f"[PEPTIDO] Missing required columns for distributions: {sorted(missing)!r}"
                    )

                pep_key = (
                    pl.col("INDEX")
                    .cast(pl.Utf8)
                    .str.split("|")
                    .list.get(0)
                    .alias("PEP_KEY")
                )
                base_prec = (
                    df.select(
                        pep_key,
                        pl.col("PARENT_PROTEIN").cast(pl.Utf8).alias("PROT_KEY"),
                        pl.col("CHARGE"),
                    )
                    .drop_nulls(["PEP_KEY", "PROT_KEY", "CHARGE"])
                    .unique()
                )

            # a) Precursors per peptide: count unique charge per peptide key
            prec_per_pep = (
                base_prec.group_by("PEP_KEY", maintain_order=True)
                .agg(pl.len().alias("N_PREC_PER_PEP"))
            )
            self.intermediate_results.add_array(
                "num_precursors_per_peptide",
                prec_per_pep["N_PREC_PER_PEP"].to_numpy(),
            )

            # b) Peptides per protein: count unique peptide keys per protein
            pep_per_prot = (
                base_prec.select(["PROT_KEY", "PEP_KEY"])
                .unique()
                .group_by("PROT_KEY", maintain_order=True)
                .agg(pl.len().alias("N_PEP_PER_PROT"))
            )
            self.intermediate_results.add_array(
                "num_peptides_per_protein",
                pep_per_prot["N_PEP_PER_PROT"].to_numpy(),
            )

            # c) Precursors per protein: count unique (peptide,charge) per protein
            prec_per_prot = (
                base_prec.group_by("PROT_KEY", maintain_order=True)
                .agg(pl.len().alias("N_PREC_PER_PROT"))
            )
            self.intermediate_results.add_array(
                "num_precursors_per_protein",
                prec_per_prot["N_PREC_PER_PROT"].to_numpy(),
            )

        # 3) Peptide-wide matrices (sum duplicates)
        if not precursor_only:
            pep_pivot = self._pivot_df(
                df=base.select(["PEPTIDE_ID", "FILENAME", "SIGNAL"]),
                sample_col="FILENAME",
                protein_col="PEPTIDE_ID",
                values_col="SIGNAL",
                aggregate_fn="sum",
            )

            id_map = base.select(["PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ"]).unique()
            pep_wide = pep_pivot.join(id_map, on="PEPTIDE_ID", how="left")

            sample_cols = [
                c
                for c in pep_wide.columns
                if c not in ("PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ")
            ]

            pep_centered = (
                pep_wide.with_columns(
                    pl.mean_horizontal(
                        [
                            pl.when(pl.col(c).is_nan())
                            .then(None)
                            .otherwise(pl.col(c))
                            for c in sample_cols
                        ]
                    ).alias("__rowmean__")
                )
                .with_columns(
                    [
                        (pl.col(c) / pl.col("__rowmean__")).alias(c)
                        for c in sample_cols
                    ]
                )
                .drop("__rowmean__")
            )

        # 4) Precursor-wide matrices (peptidomics/phospho: same feature + charge)
        # IMPORTANT:
        # - In peptido/phospho the "feature" is the peptide (INDEX).
        # - A precursor is therefore INDEX + CHARGE (not a different sequence).
        # - Some upstream steps may have already produced INDEX values containing '|...'.
        #   We make the feature id explicit and stable by taking the part before '|'.

        if analysis == "phospho":
            if "PEPTIDE_INDEX" not in df.columns:
                raise ValueError("[PHOSPHO] Missing required column 'PEPTIDE_INDEX' for precursor tables.")

            prec_base = (
                df.select(
                    pl.col("INDEX").cast(pl.Utf8).alias("SITE_ID"),
                    pl.col("PEPTIDE_INDEX").cast(pl.Utf8).alias("PEPTIDE_INDEX"),
                    pl.col("FILENAME"),
                    pl.col("SIGNAL"),
                    pl.col("CHARGE"),
                )
                .drop_nulls(["SITE_ID", "PEPTIDE_INDEX", "CHARGE"])
                .with_columns(
                    (
                        pl.col("PEPTIDE_INDEX")
                        + pl.lit("/+")
                        + pl.col("CHARGE").cast(pl.Utf8)
                    ).alias("PREC_ID")
                )
            )
        else:
            index_clean = (
                pl.col("INDEX")
                .cast(pl.Utf8)
                .str.split("|")
                .list.get(0)
                .alias("_INDEX_FEATURE")
            )
            prec_base = (
                df.select(
                    index_clean,
                    pl.col("FILENAME"),
                    pl.col("SIGNAL"),
                    pl.col("CHARGE"),
                )
                .drop_nulls(["_INDEX_FEATURE", "CHARGE"])
                .with_columns(
                    (
                        pl.col("_INDEX_FEATURE")
                        + pl.lit("/+")
                        + pl.col("CHARGE").cast(pl.Utf8)
                    ).alias("PREC_ID")
                )
                .rename({"_INDEX_FEATURE": "INDEX"})
                .with_columns(pl.col("INDEX").alias("PEPTIDE_SEQ"))
            )
#        if analysis == "phospho":
#            if "PEPTIDE_INDEX" not in df.columns:
#                raise ValueError("[PHOSPHO] Missing required column 'PEPTIDE_INDEX' for precursor tables.")
#            index_clean = pl.col("PEPTIDE_INDEX").cast(pl.Utf8).alias("_INDEX_FEATURE")
#        else:
#            index_clean = (
#                pl.col("INDEX")
#                .cast(pl.Utf8)
#                .str.split("|")
#                .list.get(0)
#                .alias("_INDEX_FEATURE")
#            )
#
#        prec_base = (
#            df.select(
#                index_clean,
#                pl.col("FILENAME"),
#                pl.col("SIGNAL"),
#                pl.col("CHARGE"),
#            )
#            .drop_nulls(["_INDEX_FEATURE", "CHARGE"])
#            .with_columns(
#                (
#                    pl.col("_INDEX_FEATURE")
#                    + pl.lit("/+")
#                    + pl.col("CHARGE").cast(pl.Utf8)
#                ).alias("PREC_ID")
#            )
#            .rename({"_INDEX_FEATURE": "INDEX"})
#            .with_columns(pl.col("INDEX").alias("PEPTIDE_SEQ"))
#        )

        prec_pivot = self._pivot_df(
            df=prec_base.select(["PREC_ID", "FILENAME", "SIGNAL"]),
            sample_col="FILENAME",
            protein_col="PREC_ID",
            values_col="SIGNAL",
            aggregate_fn="sum",
        )

        if analysis == "phospho":
            prec_id_map = (
                prec_base
                .select(["PREC_ID", "SITE_ID", "PEPTIDE_INDEX", "CHARGE"])
                .unique(subset=["PREC_ID"], maintain_order=True)
            )
        else:
            prec_id_map = (
                prec_base
                .select(["PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE"])
                .unique(subset=["PREC_ID"], maintain_order=True)
            )
#        prec_id_map = (
#            prec_base
#            .select(["PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE"])
#            .unique(subset=["PREC_ID"], maintain_order=True)
#        )

        prec_wide = prec_pivot.join(prec_id_map, on="PREC_ID", how="left")

        id_cols = {"PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE", "SITE_ID", "PEPTIDE_INDEX"}
        prec_sample_cols = [
            c
            for c, dt in zip(prec_wide.columns, prec_wide.dtypes)
            if (c not in id_cols) and (dt in pl.NUMERIC_DTYPES)
        ]
#        prec_sample_cols = [
#            c
#            for c in prec_wide.columns
#            if c not in ("PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE")
#        ]
        prec_centered = (
            prec_wide.with_columns(
                pl.mean_horizontal(
                    [
                        pl.when(pl.col(c).is_nan())
                        .then(None)
                        .otherwise(pl.col(c))
                        for c in prec_sample_cols
                    ]
                ).alias("__rowmean__")
            )
            .with_columns(
                [
                    (pl.col(c) / pl.col("__rowmean__")).alias(c)
                    for c in prec_sample_cols
                ]
            )
            .drop("__rowmean__")
        )

        self.intermediate_results.dfs["precursors_wide"] = prec_wide
        self.intermediate_results.dfs["precursors_centered"] = prec_centered

        # 5) Per-condition consistency (peptides vs precursors)
        cond_map = self.intermediate_results.dfs.get("condition_pivot")
        if cond_map is None:
            return pep_wide, pep_centered

        cond_map = cond_map.select(
            ["Sample", "CONDITION"]
            + (
                ["IS_COVARIATE"]
                if "IS_COVARIATE" in cond_map.columns
                else []
            )
        ).rename({"Sample": "FILENAME"})

        if "IS_COVARIATE" in cond_map.columns:
            cond_map = cond_map.filter(~pl.col("IS_COVARIATE")).drop("IS_COVARIATE")

        if cond_map.height == 0:
            return pep_wide, pep_centered

        conditions = (
            cond_map.select("CONDITION").unique().to_series().to_list()
        )
        cond_to_runs = {
            cond: (
                cond_map.filter(pl.col("CONDITION") == cond)
                .select("FILENAME")
                .to_series()
                .to_list()
            )
            for cond in conditions
        }

        analysis = self.analysis_type or ""

        if analysis in ["peptidomics", "phospho"]:
            # Precursor key = INDEX/CHARGE, same INDEX as roll-up
            prec_long = (
                df.select(
                    ["INDEX", "PEPTIDE_LSEQ", "CHARGE", "FILENAME", "SIGNAL"]
                )
                .drop_nulls(["INDEX", "CHARGE"])
                .with_columns(
                    [
                        pl.concat_str(
                            [
                                pl.col("INDEX").cast(pl.Utf8),
                                pl.lit("/+"),
                                pl.col("CHARGE").cast(pl.Utf8),
                            ]
                        ).alias("PREC_KEY"),
                        (
                            pl.col("SIGNAL").is_not_null()
                            & ~pl.col("SIGNAL").is_nan()
                        ).alias("HAS_SIGNAL"),
                    ]
                )
                .group_by(["PREC_KEY", "FILENAME"], maintain_order=True)
                .agg(pl.col("HAS_SIGNAL").any().alias("HAS_SIGNAL"))
            )

            prec2idx = (
                df.select(["INDEX", "CHARGE"])
                .drop_nulls(["INDEX", "CHARGE"])
                .with_columns(
                    pl.concat_str(
                        [
                            pl.col("INDEX").cast(pl.Utf8),
                            pl.lit("/+"),
                            pl.col("CHARGE").cast(pl.Utf8),
                        ]
                    ).alias("PREC_KEY")
                )
                .select(["PREC_KEY", "INDEX"])
                .unique()
            )

            wide_prec = (
                prec_long.pivot(
                    index="PREC_KEY", columns="FILENAME", values="HAS_SIGNAL"
                ).fill_null(False)
            )
            bool_sample_cols = [
                c for c in wide_prec.columns if c != "PREC_KEY"
            ]

            per_cond_prec: List[pl.DataFrame] = []

            for cond, runs in cond_to_runs.items():
                cond_cols = [c for c in bool_sample_cols if c in runs]
                if not cond_cols:
                    continue

                cons_flag = pl.all_horizontal(
                    [pl.col(c) for c in cond_cols]
                ).alias("_cons")

                cons_prec = (
                    wide_prec.select(["PREC_KEY"] + cond_cols)
                    .with_columns(cons_flag)
                    .select(
                        [
                            "PREC_KEY",
                            pl.col("_cons")
                            .cast(pl.Int64)
                            .alias("_CONS_INT"),
                        ]
                    )
                )

                cons_counts = (
                    cons_prec.join(prec2idx, on="PREC_KEY", how="left")
                    .group_by("INDEX", maintain_order=True)
                    .agg(
                        pl.col("_CONS_INT")
                        .sum()
                        .alias(f"CONSISTENT_PRECURSOR_{cond}")
                    )
                )

                per_cond_prec.append(cons_counts)

            if per_cond_prec:
                cons_df = per_cond_prec[0]
                for extra in per_cond_prec[1:]:
                    cons_df = cons_df.join(extra, on="INDEX", how="outer")
                    keep_cols = ["INDEX"] + [
                        c
                        for c in cons_df.columns
                        if c.startswith("CONSISTENT_PRECURSOR_")
                    ]
                    cons_df = cons_df.select(keep_cols)

                self.intermediate_results.add_df(
                    "consistent_precursors_per_condition", cons_df
                )

        else:
            # Proteomics: keep peptide-based consistency
            per_cond_pep: List[pl.DataFrame] = []

            for cond, runs in cond_to_runs.items():
                cond_cols = [c for c in sample_cols if c in runs]
                if not cond_cols:
                    continue

                consistent_flag = pl.all_horizontal(
                    [
                        pl.col(c).is_not_null() & ~pl.col(c).is_nan()
                        for c in cond_cols
                    ]
                ).alias("_cons")

                tmp = (
                    pep_wide.select(["INDEX"] + cond_cols)
                    .with_columns(consistent_flag)
                    .group_by("INDEX", maintain_order=True)
                    .agg(
                        pl.col("_cons")
                        .sum()
                        .cast(pl.Int64)
                        .alias(f"CONSISTENT_PEPTIDE_{cond}")
                    )
                )
                per_cond_pep.append(tmp)

            if per_cond_pep:
                cons_df = per_cond_pep[0]
                for extra in per_cond_pep[1:]:
                    cons_df = cons_df.join(extra, on="INDEX", how="outer")
                    keep_cols = ["INDEX"] + [
                        c
                        for c in cons_df.columns
                        if c.startswith("CONSISTENT_PEPTIDE_")
                    ]
                    cons_df = cons_df.select(keep_cols)

                self.intermediate_results.add_df(
                    "consistent_peptides_per_condition", cons_df
                )

        return pep_wide, pep_centered
        return (pep_wide, pep_centered) if not precursor_only else None

    # -------------------------------------------------------------------------
    # Main pivot (proteins) + covariates
    # -------------------------------------------------------------------------

    @log_time("Pivoting data")
    def _pivot_data(self) -> None:
        """Convert long format to wide protein-level matrices; build peptide tables."""

        df = self.intermediate_results.dfs["filtered_final_censored"]

        required_cols = {"INDEX", "FILENAME", "SIGNAL"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for pivoting: {missing_cols}"
            )

        if "IS_COVARIATE" in df.columns:
            is_cov = pl.coalesce([pl.col("IS_COVARIATE"), pl.lit(False)])
            df_cov = df.filter(is_cov)
            df_main = df.filter(~is_cov)
        else:
            has_assay = "ASSAY" in df.columns
            assay_lc = set(a.lower() for a in (self.covariate_assays or []))
            if has_assay and assay_lc:
                df_cov = df.filter(
                    pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))
                )
                df_main = df.filter(
                    ~pl.col("ASSAY").str.to_lowercase().is_in(list(assay_lc))
                )
            else:
                df_main = df
                df_cov = None

        def _pivot_block(_df: pl.DataFrame) -> Dict[str, Optional[pl.DataFrame]]:
            method = (self.pivot_signal_method or "sum").lower()
            log_info(f"Quantification using {method}")
            if method == "directlfq":
                intensity = self._pivot_df_LFQ(
                    df=_df,
                    sample_col="FILENAME",
                    protein_col="INDEX",
                    values_col="SIGNAL",
                    ion_col="PEPTIDE_LSEQ",
                )
            elif method in {"top3", "top_3"}:
                intensity = self._pivot_df_top_n(
                    df=_df,
                    sample_col="FILENAME",
                    protein_col="INDEX",
                    values_col="SIGNAL",
                    peptide_col="PEPTIDE_LSEQ",
                    n=3,
                )
            else:
                intensity = self._pivot_df(
                    df=_df,
                    sample_col="FILENAME",
                    protein_col="INDEX",
                    values_col="SIGNAL",
                    aggregate_fn=self.pivot_signal_method,
                )

            qv = pp = sc = prec = lp = None
            if "QVALUE" in _df.columns:
                qv = self._pivot_df(
                    _df, "FILENAME", "INDEX", "QVALUE", "mean"
                )
            if "PEP" in _df.columns:
                pp = self._pivot_df(
                    _df, "FILENAME", "INDEX", "PEP", "mean"
                )
            if self.analysis_type == "phospho":
                df_prec = (
                    _df.select(
                        ["INDEX", "FILENAME", "PEPTIDE_LSEQ", "CHARGE"]
                    )
                    .drop_nulls(["PEPTIDE_LSEQ", "CHARGE"])
                    .with_columns(
                        pl.concat_str(
                            [
                                pl.col("PEPTIDE_LSEQ"),
                                pl.lit("/"),
                                pl.col("CHARGE").cast(pl.Utf8),
                            ]
                        ).alias("PREC_KEY")
                    )
                    .unique(
                        subset=["INDEX", "FILENAME", "PREC_KEY"],
                        maintain_order=True,
                    )
                    .group_by(["INDEX", "FILENAME"], maintain_order=True)
                    .agg(pl.len().alias("N_UNIQ_PREC"))
                )
                sc = self._pivot_df(
                    df_prec,
                    "FILENAME",
                    "INDEX",
                    "N_UNIQ_PREC",
                    "max",
                ).fill_nan(0)
            elif "SPECTRAL_COUNTS" in _df.columns:
                sc = self._pivot_df(
                    _df,
                    "FILENAME",
                    "INDEX",
                    "SPECTRAL_COUNTS",
                    "max",
                )

            if self.analysis_type == "phospho":
                prec = self._pivot_df(
                    _df, "FILENAME", "INDEX", "PRECURSORS_EXP", "max"
                )
            elif "PRECURSORS_EXP" in _df.columns:
                prec = self._pivot_df(
                    _df, "FILENAME", "INDEX", "PRECURSORS_EXP", "mean"
                )
            if "LOC_PROB" in _df.columns:
                lp = self._pivot_df(
                    _df, "FILENAME", "INDEX", "LOC_PROB", "max"
                )
            ibaq = None
            if "IBAQ" in _df.columns and "INDEX" in _df.columns:
                _df = _df.with_columns(
                    pl.col("IBAQ")
                    .cast(pl.Utf8, strict=False)
                    .str.split(";")
                    .list.eval(pl.element().cast(pl.Float64, strict=False))
                    .list.mean()
                    .alias("IBAQ")
                )
                ibaq = self._pivot_df(
                    _df, "FILENAME", "INDEX", "IBAQ", "mean"
                )
            return {
                "intensity": intensity,
                "qv": qv,
                "pep": pp,
                "sc": sc,
                "prec": prec,
                "lp": lp,
                "ibaq": ibaq,
            }

        main_piv = _pivot_block(df_main)

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

        # Optional phospho localization filtering
        filtered_intensity = intensity_pivot
        with log_indent():
            if self.analysis_type == "phospho" and (locprob_pivot is not None):
                log_info("Filtering (phospho localization)")
                cols = [c for c in locprob_pivot.columns if c != "INDEX"]
                lp_mat = locprob_pivot.select(cols).to_numpy()
                if self.phospho_loc_mode == "filter_soft":
                    keep = np.nanmax(lp_mat, axis=1) >= self.phospho_loc_thr
                else:
                    keep = np.nanmin(lp_mat, axis=1) >= self.phospho_loc_thr
                keep = np.asarray(keep, dtype=bool)

                kept_n = int(keep.sum())
                total_n = int(lp_mat.shape[0])
                dropped = total_n - kept_n
                log_info(
                    f"Phospho localization filter ({self.phospho_loc_mode}, thr={self.phospho_loc_thr}): "
                    f"kept={kept_n}/{total_n}."
                )

                def _row_filter(
                    pvt: Optional[pl.DataFrame],
                ) -> Optional[pl.DataFrame]:
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
        ir.add_metadata("quantification", "method", self.pivot_signal_method)
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

                def _pivot_cov_by_key(
                    frame: Optional[pl.DataFrame], value_col: str, agg: str
                ) -> Optional[pl.DataFrame]:
                    if frame is None or value_col not in frame.columns:
                        return None
                    long_with_key = frame.join(
                        cov_key_map, on="INDEX", how="left"
                    ).drop_nulls([key_col])

                    if value_col == "IBAQ":
                        long_with_key = long_with_key.with_columns(
                            pl.col("IBAQ")
                            .cast(pl.Utf8, strict=False)
                            .str.split(";")
                            .list.eval(
                                pl.element().cast(
                                    pl.Float64, strict=False
                                )
                            )
                            .list.mean()
                            .alias("IBAQ")
                        )

                    if (
                        value_col == "SIGNAL"
                        and self.covariate_pivot_method.lower()
                        == "directlfq"
                    ):
                        if "PEPTIDE_LSEQ" not in long_with_key.columns:
                            raise ValueError(
                                "Covariate LFQ requested but 'PEPTIDE_LSEQ' is missing in covariate runs. "
                                "Either provide peptide sequences in the covariate data or set "
                                "preprocessing.pivot_signal_method='sum'."
                            )
                        return self._pivot_df_LFQ(
                            df=long_with_key.select(
                                [key_col, "FILENAME", "PEPTIDE_LSEQ", value_col]
                            ),
                            sample_col="FILENAME",
                            protein_col=key_col,
                            values_col=value_col,
                            ion_col="PEPTIDE_LSEQ",
                        )
                    else:
                        return self._pivot_df(
                            df=long_with_key.select(
                                [key_col, "FILENAME", value_col]
                            ),
                            sample_col="FILENAME",
                            protein_col=key_col,
                            values_col=value_col,
                            aggregate_fn=agg,
                        )

                log_info(
                    f"Covariate Quantification using {self.covariate_pivot_method}"
                )
                cov_int_by_key = _pivot_cov_by_key(
                    df_cov, "SIGNAL", self.covariate_pivot_method
                )
                cov_qv_by_key = _pivot_cov_by_key(
                    df_cov, "QVALUE", "mean"
                ) if "QVALUE" in df_cov.columns else None
                cov_pep_by_key = _pivot_cov_by_key(
                    df_cov, "PEP", "mean"
                ) if "PEP" in df_cov.columns else None
                cov_prec_by_key = _pivot_cov_by_key(
                    df_cov, "PRECURSORS_EXP", "max"
                )
                cov_sc_by_key = _pivot_cov_by_key(
                    df_cov, "SPECTRAL_COUNTS", "max"
                ) if "SPECTRAL_COUNTS" in df_cov.columns else None
                cov_ibaq_by_key = _pivot_cov_by_key(
                    df_cov, "IBAQ", "mean"
                ) if "IBAQ" in df_cov.columns else None

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
                ir.add_df("spectral_counts_covariate", cov_prec)
                ir.add_df("ibaq_covariate", cov_ibaq)
                log_info(
                    f"Built covariate pivot block - broadcast on {key_col}; shape matches main."
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

    # -------------------------------------------------------------------------
    # Debug helper
    # -------------------------------------------------------------------------

    def _log_pivot_health(
        self, label: str, df: pl.DataFrame, head_rows: int = 3, preview_cols: int = 50
    ) -> None:
        """Log shape, total NA count, and tiny head() preview for debugging."""
        if df is None:
            return
        cols = [c for c in df.columns if c != "INDEX"]
        rows, cols_n = df.height, len(cols)

        na_counts = df.select(
            [pl.col(c).is_null().sum().alias(c) for c in cols]
        )
        total_nas = int(
            na_counts.select(pl.all().sum()).to_numpy().ravel()[0]
        )

        log_info(f"{label}: shape={rows}×{cols_n}, total_NAs={total_nas}")

        preview = df.select(["INDEX"] + cols[:preview_cols]).head(head_rows)
        log_info(f"{label} preview:\n{preview}")

