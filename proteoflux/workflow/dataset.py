"""Dataset loader and converter to AnnData.

This module loads raw quant data (CSV/TSV), harmonizes and preprocesses it,
optionally injects additional runs (including covariates), and assembles an
AnnData object with layers and metadata for downstream analysis.
"""

from copy import deepcopy
from typing import Optional, Tuple, Union, List
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.csv as pv_csv
import csv

from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.utils.utils import polars_matrix_to_numpy, log_time, logger, log_info
from proteoflux.utils.analysis_type import normalize_analysis_type

pl.Config.set_tbl_rows(100)

# Suppress the ImplicitModificationWarning from AnnData
warnings.filterwarnings("ignore", category=UserWarning, message=".*Transforming to str index.*")


class Dataset:
    """Main entry point for loading, preprocessing, and exporting to AnnData."""

    def __init__(self, **kwargs) -> None:
        """Initialize from a nested config dict (`dataset`, `preprocessing`)."""

        # Dataset-specific config (read once; keep original for injected runs)
        dataset_cfg = kwargs.get("dataset", {}) or {}
        self._dataset_cfg_original = deepcopy(dataset_cfg)

        self.file_path = dataset_cfg.get("input_file", None)
        self.load_method = dataset_cfg.get("load_method", "polars")
        self.input_layout = dataset_cfg.get("input_layout", "long")
        self.analysis_type = normalize_analysis_type(
            dataset_cfg.get("analysis_type", "proteomics")
        )

        self.inject_runs_cfg: dict = dataset_cfg.get("inject_runs", {}) or {}

        self.exclude_runs = self._parse_exclude_runs(dataset_cfg.get("exclude_runs"))

        # Preprocessing config and setup
        preprocessing_cfg = deepcopy(kwargs.get("preprocessing", {}) or {})

        # PTMs
        collapse_met_oxidation = bool(preprocessing_cfg.get("collapse_met_oxidation", True))
        drop_ptms = bool(preprocessing_cfg.get("drop_ptms", False))

        # Harmonizer setup
        eff_dataset_cfg = deepcopy(dataset_cfg) or {}
        eff_dataset_cfg["analysis_type"] = self.analysis_type
        eff_dataset_cfg["collapse_met_oxidation"] = collapse_met_oxidation
        eff_dataset_cfg["drop_ptms"] = drop_ptms

        self.harmonizer = DataHarmonizer(eff_dataset_cfg)

        # Derive covariate assays from inject_runs.*.is_covariate (no extra config burden)
        preprocessing_cfg["analysis_type"] = self.analysis_type
        cov_assays = [
            str(name)
            for name, cfg in (self.inject_runs_cfg or {}).items()
            if cfg and bool(cfg.get("is_covariate", False))
        ]

        cov_block = deepcopy(preprocessing_cfg.get("covariates", {}) or {})
        # enabled strictly from inject_runs.*.is_covariate
        cov_block["enabled"] = bool(cov_assays)
        if cov_assays:
            # merge/extend existing list if user had one
            prev = set(map(str.lower, cov_block.get("assays", []) or []))
            prev |= set(map(str.lower, cov_assays))
            cov_block["assays"] = sorted(prev)
        preprocessing_cfg["covariates"] = cov_block

        self.preprocessor = Preprocessor(preprocessing_cfg)

        # Process
        self._load_and_process()

    @staticmethod
    def _parse_exclude_runs(x) -> set[str]:
        """Accept None / str / iterable; return a normalized set of run identifiers."""
        if x is None:
            return set()
        if isinstance(x, str):
            return {x.strip()} if x.strip() else set()
        try:
            return {str(v).strip() for v in x if str(v).strip()}
        except TypeError:
            # not iterable (e.g. int); fall back to single-item set
            s = str(x).strip()
            return {s} if s else set()

    @property
    def is_proteomics(self) -> bool:
        return self.analysis_type == "proteomics"

    @property
    def is_peptidomics(self) -> bool:
        return self.analysis_type == "peptidomics"

    @property
    def is_phospho(self) -> bool:
        return self.analysis_type == "phospho"

    @log_time("Runs Injection")
    def _load_injected_runs(self) -> list[pl.DataFrame]:
        """Load, harmonize, and tag each configured injected run."""

        frames = []

        for inject_name, inject_cfg in self.inject_runs_cfg.items():
            if not inject_cfg:
                continue
            fpath = inject_cfg.get("input_file")
            if not fpath:
                # null path → skip cleanly
                continue

            # 1) load raw (same as primary)
            df_inj = self._load_rawdata(fpath)

            # 2) build a per-injection harmonizer config
            #    start from the original dataset section so column mappings are identical by default
            eff_cfg = deepcopy(self._dataset_cfg_original)
            eff_cfg["input_layout"] = inject_cfg.get(
                "input_layout", eff_cfg.get("input_layout", self.input_layout)
            )
            eff_cfg["analysis_type"] = normalize_analysis_type(
                inject_cfg.get("analysis_type", self.analysis_type)
            )
            eff_cfg["annotation_file"] = inject_cfg.get("annotation_file", None)
            eff_cfg["is_covariate_run"] = bool((inject_cfg or {}).get("is_covariate", False))
            # Keep peptide identity normalization consistent with the main dataset:
            # canonical source is preprocessing config.
            eff_cfg["collapse_met_oxidation"] = bool(self.preprocessor.collapse_met_oxidation)
            eff_cfg["drop_ptms"] = bool(self.preprocessor.drop_ptms)

            # 3) harmonize with its own annot (no cross-merge!)
            inj_harmonizer = DataHarmonizer(eff_cfg)
            df_inj = inj_harmonizer.harmonize(df_inj)

            # 4) provenance tags (create THEN cast)
            if inject_name not in df_inj.columns:
                df_inj = df_inj.with_columns(pl.lit(True).alias(inject_name))

            df_inj = df_inj.with_columns(pl.col(inject_name).cast(pl.Boolean))

            if "ASSAY" not in df_inj.columns:
                df_inj = df_inj.with_columns(pl.lit(str(inject_name)).alias("ASSAY"))

            is_cov = bool((inject_cfg or {}).get("is_covariate", False))
            df_inj = df_inj.with_columns(pl.lit(is_cov).alias("IS_COVARIATE"))

            # logging: shape & covariate status
            log_info(
                f"Injected run '{inject_name}': rows={len(df_inj)}, "
                f"assay='{df_inj.select(pl.col('ASSAY').first()).item()}', "
                f"is_covariate={is_cov}"
            )

            # normalize dtypes we care about (avoids mixed Utf8/Null later)
            df_inj = df_inj.with_columns(pl.col("ASSAY").cast(pl.Utf8))

            frames.append(df_inj)

        return frames

    def _concat_relaxed(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        """Row-bind frames with schema alignment (columns + dtypes) across inputs."""

        if not frames:
            return pl.DataFrame()

        # stable column order from the first df, then append new ones in discovery order
        ordered_cols = list(frames[0].columns)
        for df in frames[1:]:
            for c in df.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)

        # choose a target dtype per column (ignore Null if a real dtype exists elsewhere)
        target_dtype: dict[str, pl.datatypes.DataType] = {}
        for c in ordered_cols:
            chosen = None
            for df in frames:
                if c in df.columns:
                    dt = df.schema[c]
                    if dt != pl.Null:
                        chosen = dt if chosen is None else chosen
            # fallback stays Null only if *all* frames have Null or missing -> cast to Utf8 for safety
            target_dtype[c] = chosen if chosen is not None else pl.Utf8

        # align each frame to the target schema
        fixed = []
        for df in frames:
            # add missing columns with proper dtype
            missing = [c for c in ordered_cols if c not in df.columns]
            if missing:
                df = df.with_columns(
                    [pl.lit(None).cast(target_dtype[c]).alias(c) for c in missing]
                )
            # cast existing columns to target dtype
            casts = []
            for c in ordered_cols:
                if c in df.columns:
                    cur = df.schema[c]
                    want = target_dtype[c]
                    if cur != want:
                        casts.append(pl.col(c).cast(want, strict=False).alias(c))
            if casts:
                df = df.with_columns(casts)

            # select in final order
            df = df.select(ordered_cols)
            fixed.append(df)

        return pl.concat(fixed, how="vertical", rechunk=True)

    def _maybe_concat_injected_runs(self, df_main: pl.DataFrame) -> pl.DataFrame:
        """Optionally load + harmonize injected runs and concat them onto the primary table."""
        if not self.inject_runs_cfg:
            return df_main

        injected_frames = self._load_injected_runs()
        if not injected_frames:
            return df_main

        # make sure primary has the same boolean tag columns with False
        for inject_name in self.inject_runs_cfg.keys():
            if inject_name not in df_main.columns:
                df_main = df_main.with_columns(pl.lit(False).alias(inject_name))

        # ensure IS_COVARIATE exists on the primary and is False
        if "IS_COVARIATE" not in df_main.columns:
            df_main = df_main.with_columns(pl.lit(False).alias("IS_COVARIATE"))
        else:
            df_main = df_main.with_columns(
                pl.coalesce([pl.col("IS_COVARIATE"), pl.lit(False)]).alias("IS_COVARIATE")
            )

        # ASSAY dtype guard on primary too (prevents Utf8/Null mismatches)
        if "ASSAY" in df_main.columns:
            df_main = df_main.with_columns(pl.col("ASSAY").cast(pl.Utf8))

        return self._concat_relaxed([df_main] + injected_frames)

    def _load_and_process(self):
        """Load raw data, harmonize, inject runs, exclude runs, preprocess, convert."""

        # Load data
        self.rawinput = self._load_rawdata(self.file_path)

        # Harmonize data
        self.rawinput = self.harmonizer.harmonize(self.rawinput)

        # Inject runs (if any)
        self.rawinput = self._maybe_concat_injected_runs(self.rawinput)

        # Exclude files (if any)
        self.rawinput = self._apply_exclude_runs(self.rawinput)

        # Apply preprocessing
        self.preprocessed_data = self._apply_preprocessing(self.rawinput)

        # Convert to AnnData format
        self._convert_to_anndata()

    def _apply_exclude_runs(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop runs (by FILENAME) if requested. Safe if annotation is absent."""
        if not self.exclude_runs:
            return df
        if "FILENAME" not in df.columns:
            log_info(
                "Exclude runs: 'FILENAME' not present after harmonization → skip."
            )
            return df

        present = set(df.select("FILENAME").to_series().to_list())
        to_drop = sorted(self.exclude_runs & present)
        missing = sorted(self.exclude_runs - present)

        if missing:
            head = ", ".join(missing[:10])
            tail = " ..." if len(missing) > 10 else ""
            log_info(
                f"Exclude runs: {len(missing)} not found in data → ignored: [{head}{tail}]"
            )

        if not to_drop:
            log_info("Exclude runs: nothing to drop.")
            return df

        n_before = df.height
        df2 = df.filter(~pl.col("FILENAME").is_in(to_drop))
        n_after = df2.height
        log_info(
            f"Exclude runs: dropped {len(to_drop)} run(s), removed {n_before - n_after} row(s)."
        )

        return df2

    @log_time("Data Loading")
    def _load_rawdata(self, file_path: str) -> Union[pl.DataFrame, pd.DataFrame]:
        """Load raw data from a CSV or TSV using Polars/PyArrow/Pandas backends."""
        if not file_path.endswith((".csv", ".tsv")):
            raise ValueError("Only CSV or TSV files are supported.")

        delimiter = "\t" if file_path.endswith(".tsv") else ","

        if self.load_method == "polars":
            # Force known statistical columns to float to avoid inference bugs
            # (e.g. many zeros early -> inferred as int -> later fails on decimals).
            with open(file_path, "r", newline="") as fh:
                header = next(csv.reader(fh, delimiter=delimiter))
            schema_overrides = {
                col: pl.Float64
                for col in header
                if ("Qvalue" in col) or ("Pvalue" in col) or col.endswith(".PEP") or col == "PEP"
            }
            df = pl.read_csv(
                file_path,
                separator=delimiter,
                infer_schema_length=10000,
                null_values=["NA", "NaN", "N/A", ""],
                schema_overrides=schema_overrides or None,
            )
            return df
        elif self.load_method == "pyarrow":
            parse_options = pv_csv.ParseOptions(delimiter=delimiter)
            arrow_table = pv_csv.read_csv(file_path, parse_options=parse_options)
            return pl.from_arrow(arrow_table)
        elif self.load_method == "pandas":
            df = pd.read_csv(file_path, delimiter=delimiter)
            return pl.from_pandas(df)
        else:
            raise ValueError(f"Unknown load method: {self.load_method}")

    @log_time("Data Processing")
    def _apply_preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run preprocessing (normalization, imputation, pivots) via `Preprocessor`."""

        preprocessed_results = self.preprocessor.fit_transform(df)
        return preprocessed_results

    @log_time("Conversion to AnnData")
    def _convert_to_anndata(self) -> None:
        """Convert preprocessed results into an AnnData object (layers + metadata)."""

        # Extract matrices from PreprocessResults
        filtered_mat = self.preprocessed_data.filtered  # filtered before log
        lognorm_mat = self.preprocessed_data.lognormalized  # post log
        normalized_mat = self.preprocessed_data.normalized  # post norm
        processed_mat = self.preprocessed_data.processed  # post impu
        centered_cov_mat = getattr(self.preprocessed_data, "centered_covariate", None)
        qval_mat = self.preprocessed_data.qvalues
        pep_mat = self.preprocessed_data.pep
        locprob_mat = getattr(self.preprocessed_data, "locprob", None)
        sc_mat = self.preprocessed_data.spectral_counts
        ibaq_mat = self.preprocessed_data.ibaq
        condition_df = (
            self.preprocessed_data.condition_pivot.to_pandas().set_index("Sample")
        )

        pep_wide_mat = self.preprocessed_data.peptides_wide
        pep_cent_mat = self.preprocessed_data.peptides_centered

        protein_meta_df = (
            self.preprocessed_data.protein_meta.to_pandas().set_index("INDEX")
        )

        parent_protein_map = None
        if "PARENT_PROTEIN" in protein_meta_df.columns:
            tmp = protein_meta_df["PARENT_PROTEIN"].copy()
            tmp.index = tmp.index.astype(str)
            parent_protein_map = tmp

        # Use processed_mat to infer sample names (columns) and protein IDs (rows)
        X, protein_index = polars_matrix_to_numpy(processed_mat, index_col="INDEX")
        X = np.asarray(X, dtype=np.float32)

        # Access all intermediate-results dataframes once
        dfs_ir = getattr(self.preprocessor.intermediate_results, "dfs", {})

        # Attach per-condition consistent peptide/precursor counts, if available.
        # Prefer peptide consistency (proteomics), fall back to precursor consistency (peptidomics).
        cons_pl = dfs_ir.get("consistent_peptides_per_condition")
        uns_consistent_key = "consistent_peptides_per_condition"
        if cons_pl is None:
            cons_pl = dfs_ir.get("consistent_precursors_per_condition")
            uns_consistent_key = "consistent_precursors_per_condition"

        consistent_payload = None
        if cons_pl is not None:
            cons_pd = cons_pl.to_pandas().set_index("INDEX")
            cons_pd.index = cons_pd.index.astype(str)

            # align to the same order as protein_index / var
            cons_pd = cons_pd.reindex(protein_index)

            # merge into var metadata as integer (nullable) columns
            for col in cons_pd.columns:
                protein_meta_df[col] = cons_pd[col].astype("Int64")

            # compact representation for .uns
            consistent_payload = {
                "index": cons_pd.index.astype(str).tolist(),
                "columns": cons_pd.columns.tolist(),
                "values": cons_pd.to_numpy(dtype=np.int64, copy=True),
            }

        # Ensure the metadata index matches the order of protein_index from X
        protein_meta_df = protein_meta_df.loc[protein_index]

        # Helper to convert optional matrices
        def _to_np(opt_df):
            arr, idx = polars_matrix_to_numpy(opt_df, index_col="INDEX")
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float32)
            return arr, idx

        # protein level data
        qval, _ = _to_np(qval_mat)
        pep, _ = _to_np(pep_mat)
        locprob, _ = _to_np(locprob_mat)
        sc, _ = _to_np(sc_mat)
        ibaq, _ = _to_np(ibaq_mat)
        lognorm, _ = _to_np(lognorm_mat)
        normalized, _ = _to_np(normalized_mat)
        raw, _ = _to_np(filtered_mat)

        # Create var and obs metadata
        sample_names = [col for col in processed_mat.columns if col != "INDEX"]
        obs = condition_df.loc[sample_names]

        self.adata = ad.AnnData(X=X.T, obs=obs, var=protein_meta_df)
        self._attach_layers_main(
            raw=raw,
            lognorm=lognorm,
            normalized=normalized,
            qval=qval,
            pep=pep,
            locprob=locprob,
            sc=sc,
            ibaq=ibaq,
        )

        # Covariate (centered, imputed) - only if present
        if centered_cov_mat is not None:
            filtered_cov_mat = getattr(self.preprocessed_data, "raw_covariate", None)
            lognorm_cov_mat = getattr(
                self.preprocessed_data, "lognormalized_covariate", None
            )
            normalized_cov_mat = getattr(
                self.preprocessed_data, "normalized_covariate", None
            )
            processed_cov_mat = getattr(
                self.preprocessed_data, "processed_covariate", None
            )
            qval_cov_mat = getattr(self.preprocessed_data, "qvalues_covariate", None)
            pep_cov_mat = getattr(self.preprocessed_data, "pep_covariate", None)
            sc_cov_mat = getattr(
                self.preprocessed_data, "spectral_counts_covariate", None
            )
            ibaq_cov_mat = getattr(self.preprocessed_data, "ibaq_covariate", None)

            filtered_cov_np, _ = _to_np(filtered_cov_mat)
            lognorm_cov_np, _ = _to_np(lognorm_cov_mat)
            normalized_cov_np, _ = _to_np(normalized_cov_mat)
            processed_cov_np, _ = _to_np(processed_cov_mat)
            centered_cov_np, _ = _to_np(centered_cov_mat)
            qval_cov_np, _ = _to_np(qval_cov_mat)
            pep_cov_np, _ = _to_np(pep_cov_mat)
            sc_cov_np, _ = _to_np(sc_cov_mat)
            ibaq_cov_np, _ = _to_np(ibaq_cov_mat)

            self._attach_layers_covariate(
                raw=filtered_cov_np,
                lognorm=lognorm_cov_np,
                normalized=normalized_cov_np,
                processed=processed_cov_np,
                centered=centered_cov_np,
                qval=qval_cov_np,
                pep=pep_cov_np,
                sc=sc_cov_np,
                ibaq=ibaq_cov_np,
            )

        # Preprocessing summary in .uns
        self._attach_uns_preprocessing()

        # Distribution diagnostics from IntermediateResults
        dist_arrays = getattr(self.preprocessor.intermediate_results, "arrays", {})
        self.adata.uns["preprocessing"]["distributions"] = {
            "num_precursors_per_peptide": dist_arrays.get(
                "num_precursors_per_peptide"
            ),
            "num_precursors_per_protein": dist_arrays.get(
                "num_precursors_per_protein"
            ),
            "num_peptides_per_protein": dist_arrays.get("num_peptides_per_protein"),
        }

        # Missed cleavages per sample (computed in ProteoFlux; viewer only consumes).
        dfs_ir = getattr(self.preprocessor.intermediate_results, "dfs", {})
        mc = dfs_ir.get("missed_cleavages_per_sample")
        if mc is not None:
            mc_pd = mc.to_pandas()
            if not {"Sample", "MISSED_CLEAVAGE_FRACTION"}.issubset(mc_pd.columns):
                raise ValueError(
                    "Invalid missed_cleavages_per_sample table. "
                    f"Expected columns={{'Sample','MISSED_CLEAVAGE_FRACTION'}}, got={list(mc_pd.columns)!r}"
                )
            payload = {
                "samples": mc_pd["Sample"].astype(str).tolist(),
                "fraction": mc_pd["MISSED_CLEAVAGE_FRACTION"].astype(float).to_numpy(copy=True),
            }

            # Optional intensity-weighted series
            if "MISSED_CLEAVAGE_FRACTION_WEIGHTED" in mc_pd.columns:
                payload["fraction_weighted"] = (
                    mc_pd["MISSED_CLEAVAGE_FRACTION_WEIGHTED"]
                    .astype(float)
                    .to_numpy(copy=True)
                )

            self.adata.uns["preprocessing"]["distributions"]["missed_cleavages_per_sample"] = payload

        # Drill-down trend tables
        sample_names = [c for c in processed_mat.columns if c != "INDEX"]
        self._attach_uns_drilldown_peptides(sample_names, pep_wide_mat, pep_cent_mat)

        # --- Precursor trend tables (for peptidomics; optional) ---
        prec_wide_mat = dfs_ir.get("precursors_wide")
        prec_cent_mat = dfs_ir.get("precursors_centered")

        if (prec_wide_mat is not None) and (prec_cent_mat is not None):
            prec_cols = [c for c in sample_names if c in prec_wide_mat.columns]
            use_cols_pw = ["PREC_ID"] + prec_cols
            use_cols_pc = ["PREC_ID"] + prec_cols

            prec_wide_numeric = prec_wide_mat.select(use_cols_pw)
            prec_cent_numeric = prec_cent_mat.select(use_cols_pc)

            prec_raw, prec_idx = polars_matrix_to_numpy(
                prec_wide_numeric, index_col="PREC_ID"
            )
            prec_cent, prec_idx2 = polars_matrix_to_numpy(
                prec_cent_numeric, index_col="PREC_ID"
            )
            assert list(prec_idx) == list(prec_idx2)

            if self.analysis_type == "phospho":
                meta_cols = ["PREC_ID", "SITE_ID", "PEPTIDE_INDEX", "CHARGE"]
            else:
                meta_cols = ["PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE"]

            prec_meta = (
                prec_wide_mat.select(meta_cols)
                .unique(subset=["PREC_ID"], maintain_order=True)
                .to_pandas()
                .set_index("PREC_ID")
                .reindex(prec_idx)
            )

            if prec_meta.isna().any().any():
                # reindex introduces NaNs if a PREC_ID is missing; fail-fast with examples
                miss = prec_meta.index[prec_meta["INDEX"].isna()].astype(str).tolist()[:10]
                raise ValueError(f"Missing precursor metadata for PREC_ID (examples={miss})")

            # hard shape invariants (run-or-crash)
            if prec_raw.shape[0] != len(prec_idx) or prec_cent.shape[0] != len(prec_idx):
                raise ValueError(
                    f"Precursor matrix/idx mismatch: idx={len(prec_idx)} raw={prec_raw.shape} centered={prec_cent.shape}"
                )
            if len(prec_meta) != len(prec_idx):
                raise ValueError(f"Precursor meta/idx mismatch: meta={len(prec_meta)} idx={len(prec_idx)}")

            if self.analysis_type == "phospho":
                prec_protein_index = prec_meta["SITE_ID"].astype(str).tolist()
                prec_seq = prec_meta["PEPTIDE_INDEX"].astype(str).tolist()
            else:
                prec_protein_index = prec_meta["INDEX"].astype(str).tolist()
                prec_seq = prec_meta["PEPTIDE_SEQ"].astype(str).tolist()
            prec_charge = prec_meta["CHARGE"].astype(str).tolist()

            self.adata.uns["precursors"] = {
                "rows": [str(x) for x in prec_idx],  # PREC_ID "{INDEX}|{SEQ}/{CHARGE}"
                "protein_index": prec_protein_index,
                "peptide_seq": prec_seq,
                "charge": prec_charge,
                "cols": prec_cols,
                "raw": np.asarray(prec_raw, dtype=np.float32),
                "centered": np.asarray(prec_cent, dtype=np.float32),
            }

        # Analysis parameters
        self.adata.uns["analysis"] = {
            "de_method": "limma_ebayes",
            "analysis_type": self.analysis_type,
        }

        # Per-condition consistency payload in .uns
        if consistent_payload is not None:
            # Always expose the legacy key for viewers
            self.adata.uns["consistent_peptides_per_condition"] = consistent_payload

            # If we actually used precursors, also expose the more precise name
            if uns_consistent_key == "consistent_precursors_per_condition":
                self.adata.uns["consistent_precursors_per_condition"] = consistent_payload

        # check index is fine between proteins and matrix index
        assert list(protein_meta_df.index) == list(protein_index)

    def _attach_layers_main(
        self,
        *,
        raw: np.ndarray,
        lognorm: np.ndarray,
        normalized: np.ndarray,
        qval: np.ndarray | None,
        pep: np.ndarray | None,
        locprob: np.ndarray | None,
        sc: np.ndarray | None,
        ibaq: np.ndarray | None,
    ) -> None:
        self.adata.layers["raw"] = raw.T
        self.adata.layers["lognorm"] = lognorm.T
        self.adata.layers["normalized"] = normalized.T
        if qval is not None:
            self.adata.layers["qvalue"] = qval.T
        if pep is not None:
            self.adata.layers["pep"] = pep.T
        if locprob is not None:
            self.adata.layers["locprob"] = locprob.T
        if sc is not None:
            self.adata.layers["spectral_counts"] = sc.T
        if ibaq is not None:
            self.adata.layers["ibaq"] = ibaq.T

    def _attach_layers_covariate(
        self,
        *,
        raw: np.ndarray,
        lognorm: np.ndarray,
        normalized: np.ndarray,
        processed: np.ndarray,
        centered: np.ndarray,
        qval: np.ndarray,
        pep: np.ndarray,
        sc: np.ndarray,
        ibaq: np.ndarray,
    ) -> None:
        self.adata.layers["raw_covariate"] = raw.T
        self.adata.layers["lognorm_covariate"] = lognorm.T
        self.adata.layers["normalized_covariate"] = normalized.T
        self.adata.layers["processed_covariate"] = processed.T
        self.adata.layers["centered_covariate"] = centered.T
        self.adata.layers["qval_covariate"] = qval.T
        self.adata.layers["pep_covariate"] = pep.T
        self.adata.layers["sc_covariate"] = sc.T
        self.adata.layers["ibaq_covariate"] = ibaq.T

    def _attach_uns_preprocessing(self) -> None:
        self.adata.uns["preprocessing"] = {
            "input_layout": self.input_layout,
            "analysis_type": self.analysis_type,
            "filtering": {
                "cont": self.preprocessed_data.meta_cont,
                "qvalue": self.preprocessed_data.meta_qvalue,
                "pep": self.preprocessed_data.meta_pep,
                "prec": self.preprocessed_data.meta_prec,
                "censor": self.preprocessed_data.meta_censor,
            },
            "quantification": self.preprocessed_data.meta_quant,
            "normalization": self.preprocessor.normalization,
            "imputation": self.preprocessor.imputation,
        }

    def _attach_uns_drilldown_peptides(
        self,
        sample_names: list[str],
        pep_wide_mat: pl.DataFrame,
        pep_cent_mat: pl.DataFrame,
    ) -> None:
        if self.is_proteomics:
            use_cols_w = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_wide_mat.columns]
            use_cols_c = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_cent_mat.columns]

            pep_wide_numeric = pep_wide_mat.select(use_cols_w)
            pep_cent_numeric = pep_cent_mat.select(use_cols_c)

            pep_raw, pep_idx = polars_matrix_to_numpy(pep_wide_numeric, index_col="PEPTIDE_ID")
            pep_cent, pep_idx2 = polars_matrix_to_numpy(pep_cent_numeric, index_col="PEPTIDE_ID")
            assert list(pep_idx) == list(pep_idx2)

            rowmeta = (
                pep_wide_mat.select(["PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ"])
                .unique()
                .to_pandas()
                .set_index("PEPTIDE_ID")
                .loc[pep_idx]
            )
            pep_protein_index = rowmeta["INDEX"].astype(str).tolist()
            peptide_seq = rowmeta["PEPTIDE_SEQ"].astype(str).tolist()

            self.adata.uns["peptides"] = {
                "rows": [str(x) for x in pep_idx],
                "protein_index": pep_protein_index,
                "peptide_seq": peptide_seq,
                "cols": [c for c in sample_names if c in pep_wide_mat.columns],
                "raw": np.asarray(pep_raw, dtype=np.float32),
                "centered": np.asarray(pep_cent, dtype=np.float32),
            }
            return

        if self.is_peptidomics or self.is_phospho:
            # do not export peptide tables in peptidomics/phospho
            return

        raise ValueError(
            f"Unsupported analysis_type='{self.analysis_type}' for drill-down trend tables."
        )

    def get_anndata(self) -> ad.AnnData:
        """Export the processed dataset as an AnnData object."""
        return self.adata


if __name__ == "__main__":
    # for testing
    file_path = "searle_test2.tsv"

    # load dataset
    dataset = Dataset(file_path=file_path)

    adata = dataset.get_anndata()
    print(adata)

