import sys
import pyarrow.csv as pv_csv
import polars as pl
import pandas as pd
import numpy as np
import time
import anndata as ad
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import warnings

from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.utils.utils import polars_matrix_to_numpy, log_time, logger, log_info

pl.Config.set_tbl_rows(100)
# Supnress the ImplicitModificationWarning from AnnData
warnings.filterwarnings("ignore", category=UserWarning, message=".*Transforming to str index.*")

class Dataset:
    """The main class for processing the dataset, loading raw data, and converting to AnnData."""
    def __init__(self, **kwargs):
        """
        Initialize the dataset object.

        Args:
            kwargs: dict with all the config elements
        """

        # Dataset-specific config
        dataset_cfg = kwargs.get("dataset", {})
        self.file_path = dataset_cfg.get("input_file", None)
        self.load_method = dataset_cfg.get("load_method", 'polars')
        self.input_layout = dataset_cfg.get("input_layout", "long")
        self.analysis_type = dataset_cfg.get("analysis_type", "DIA")

        dataset_cfg = kwargs.get("dataset", {}) or {}

        self._dataset_cfg_original = deepcopy(dataset_cfg)
        self.inject_runs_cfg = dataset_cfg.get("inject_runs", {}) or {}

        # Accept string OR list for exclude_runs
        raw_excl = dataset_cfg.get("exclude_runs")

        def _to_set(x):
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

        self.exclude_runs = _to_set(raw_excl)

        # Harmonizer setup
        self.harmonizer = DataHarmonizer(dataset_cfg)

        # Preprocessing config and setup

        preprocessing_cfg = deepcopy(kwargs.get("preprocessing", {}) or {})

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
        #if cov_assays:
        #    cov_block = deepcopy(preprocessing_cfg.get("covariates", {}) or {})
        #    # merge/extend existing list if user had one
        #    prev = set(map(str.lower, cov_block.get("assays", []) or []))
        #    prev |= set(map(str.lower, cov_assays))
        #    cov_block["assays"] = sorted(prev)  # store lower-cased; Preprocessor matches case-insensitively
        #    preprocessing_cfg["covariates"] = cov_block

        self.preprocessor = Preprocessor(preprocessing_cfg)

        # Process
        self._load_and_process()

    @log_time("Runs Injection")
    def _load_injected_runs(self) -> list[pl.DataFrame]:
        """
        For each injection:
          - load with _load_rawdata
          - harmonize with a fresh DataHarmonizer fed ONLY that injection’s config
          - tag ASSAY=<inject_name> and a boolean column named exactly <inject_name>
        """
        frames = []
        #if not self.inject_runs_cfg:
        #    return frames

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
            eff_cfg["input_layout"]    = inject_cfg.get("input_layout",   eff_cfg.get("input_layout", self.input_layout))
            eff_cfg["analysis_type"]   = inject_cfg.get("analysis_type",  "DIA")
            eff_cfg["annotation_file"] = inject_cfg.get("annotation_file", None)

            # 3) harmonize with its own annot (no cross-merge!)
            inj_harmonizer = DataHarmonizer(eff_cfg)
            df_inj = inj_harmonizer.harmonize(df_inj)

            # 4) provenance tags (create THEN cast)
            if inject_name not in df_inj.columns:
                df_inj = df_inj.with_columns(pl.lit(True).alias(inject_name))

            df_inj = df_inj.with_columns(pl.col(inject_name).cast(pl.Boolean))

            if "ASSAY" not in df_inj.columns:
                df_inj = df_inj.with_columns(pl.lit(str(inject_name)).alias("ASSAY"))

            #df_inj = df_inj.with_columns(
            #    pl.lit(True).alias(str(inject_name))  # traceability tag
            #)
            #if "ASSAY" not in df_inj.columns:
            #    df_inj = df_inj.with_columns(pl.lit(str(inject_name)).alias("ASSAY"))

            is_cov = bool((inject_cfg or {}).get("is_covariate", False))
            df_inj = df_inj.with_columns(pl.lit(is_cov).alias("IS_COVARIATE"))

            # logging: shape & covariate status
            log_info(f"Injected run '{inject_name}': rows={len(df_inj)}, "
                     f"assay='{df_inj.select(pl.col('ASSAY').first()).item()}', "
                     f"is_covariate={is_cov}")

            # normalize dtypes we care about (avoids mixed Utf8/Null later)
            df_inj = df_inj.with_columns(pl.col("ASSAY").cast(pl.Utf8))

            frames.append(df_inj)

        return frames

    def _concat_relaxed(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        """
        Vertically concat frames with different schemas, aligning both columns and dtypes:
        - union of columns
        - pick a target dtype per column (prefer first non-Null encountered)
        - add missing cols as None cast to target dtype
        - cast existing cols to target dtype
        - regular vertical concat
        """
        if not frames:
            return pl.DataFrame()

        # 1) stable column order from the first df, then append new ones in discovery order
        ordered_cols = list(frames[0].columns)
        for df in frames[1:]:
            for c in df.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)

        # 2) choose a target dtype per column (ignore Null if a real dtype exists elsewhere)
        target_dtype: dict[str, pl.datatypes.DataType] = {}
        for c in ordered_cols:
            chosen = None
            for df in frames:
                if c in df.columns:
                    dt = df.schema[c]
                    if dt != pl.Null:
                        chosen = dt if chosen is None else chosen
            # fallback stays Null only if *all* frames have Null or missing → cast to Utf8 for safety
            target_dtype[c] = chosen if chosen is not None else pl.Utf8

        # 3) align each frame to the target schema
        fixed = []
        for df in frames:
            # add missing columns with proper dtype
            missing = [c for c in ordered_cols if c not in df.columns]
            if missing:
                df = df.with_columns([
                    pl.lit(None).cast(target_dtype[c]).alias(c) for c in missing
                ])
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

    def _load_and_process(self):

        # Load data
        self.rawinput = self._load_rawdata(self.file_path)

        # Harmonize data
        self.rawinput = self.harmonizer.harmonize(self.rawinput)

        # 2) inject runs (simple, no base cfg juggling)
        if self.inject_runs_cfg:
            injected_frames = self._load_injected_runs()
            if injected_frames:
                # make sure primary has the same boolean tag columns with False
                for inject_name in self.inject_runs_cfg.keys():
                    if inject_name not in self.rawinput.columns:
                        self.rawinput = self.rawinput.with_columns(pl.lit(False).alias(inject_name))

                # ensure IS_COVARIATE exists on the primary and is False
                if "IS_COVARIATE" not in self.rawinput.columns:
                    self.rawinput = self.rawinput.with_columns(pl.lit(False).alias("IS_COVARIATE"))
                else:
                    self.rawinput = self.rawinput.with_columns(pl.coalesce([pl.col("IS_COVARIATE"), pl.lit(False)]).alias("IS_COVARIATE"))

                # ASSAY dtype guard on primary too (prevents Utf8/Null mismatches)
                if "ASSAY" in self.rawinput.columns:
                    self.rawinput = self.rawinput.with_columns(pl.col("ASSAY").cast(pl.Utf8))
                self.rawinput = self._concat_relaxed([self.rawinput] + injected_frames)

        # Exclude files if any
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
            log_info("Exclude runs: 'FILENAME' not present after harmonization → skip.")
            return df

        present = set(df.select("FILENAME").to_series().to_list())
        to_drop = sorted(self.exclude_runs & present)
        missing = sorted(self.exclude_runs - present)

        if missing:
            head = ", ".join(missing[:10])
            tail = " ..." if len(missing) > 10 else ""
            log_info(f"Exclude runs: {len(missing)} not found in data → ignored: [{head}{tail}]")

        if not to_drop:
            log_info("Exclude runs: nothing to drop.")
            return df

        n_before = df.height
        df2 = df.filter(~pl.col("FILENAME").is_in(to_drop))
        n_after = df2.height
        log_info(f"Exclude runs: dropped {len(to_drop)} run(s), removed {n_before - n_after} row(s).")

        return df2

    @log_time("Data Loading")
    def _load_rawdata(self, file_path: str) -> Union[pl.DataFrame, pd.DataFrame]:
        """Load raw data from a CSV or TSV file using different libraries."""
        if not file_path.endswith((".csv", ".tsv")):
            raise ValueError("Only CSV or TSV files are supported.")

        delimiter = "\t" if file_path.endswith(".tsv") else ","

        if self.load_method == 'polars':
            # Use Polars to load the data, like a normal person
            df = pl.read_csv(file_path,
                             separator=delimiter,
                             infer_schema_length=10000,
                             null_values=["NA", "NaN", "N/A", ""])
            return df
        elif self.load_method == 'pyarrow':
            # Use pyarrow, eh it's good but slow
            parse_options = pv_csv.ParseOptions(delimiter=delimiter)
            arrow_table = pv_csv.read_csv(file_path, parse_options=parse_options)
            return pl.from_arrow(arrow_table)
        elif self.load_method == 'pandas':
            # Use pandas to load the data, why would I ever do this
            df = pd.read_csv(file_path, delimiter=delimiter)
            return pl.from_pandas(df)
        else:
            raise ValueError(f"Unknown load method: {self.load_method}")

    @log_time("Data Processing")
    def _apply_preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Placeholder for preprocessing steps like normalization, imputation, etc.

        Args:
            df (pl.DataFrame): The raw input data to preprocess.

        Returns:
            pl.DataFrame: The preprocessed data.
        """
        preprocessed_results = self.preprocessor.fit_transform(df)

        return preprocessed_results

    @log_time("Conversion to AnnData")
    def _convert_to_anndata(self):
        """Convert the data to an AnnData object for downstream analysis."""

        # Extract matrices
        filtered_mat = self.preprocessed_data.filtered #filtered before log
        lognorm_mat = self.preprocessed_data.lognormalized #post log
        normalized_mat = self.preprocessed_data.normalized # post norm
        processed_mat = self.preprocessed_data.processed # post impu
        centered_cov_mat = getattr(self.preprocessed_data, "centered_covariate", None)
        qval_mat = self.preprocessed_data.qvalues
        pep_mat = self.preprocessed_data.pep
        locprob_mat = getattr(self.preprocessed_data, "locprob", None)
        sc_mat = self.preprocessed_data.spectral_counts
        condition_df = self.preprocessed_data.condition_pivot.to_pandas().set_index("Sample")

        pep_wide_mat  = self.preprocessed_data.peptides_wide
        pep_cent_mat  = self.preprocessed_data.peptides_centered

        protein_meta_df = self.preprocessed_data.protein_meta.to_pandas().set_index("INDEX")

        # Use filtered_mat to infer sample names (columns) and protein IDs (rows)
        X, protein_index = polars_matrix_to_numpy(processed_mat, index_col="INDEX")

        # Ensure the metadata index matches the order of protein_index from X
        protein_meta_df = protein_meta_df.loc[protein_index]

        # Layers
        def _to_np(opt_df):
            return polars_matrix_to_numpy(opt_df, index_col="INDEX")

        qval, _    = _to_np(qval_mat)
        pep, _    = _to_np(pep_mat)
        locprob, _  = _to_np(locprob_mat)
        sc, _    = _to_np(sc_mat)
        lognorm, _ = _to_np(lognorm_mat)
        normalized, _ = _to_np(normalized_mat)
        raw, _  = _to_np(filtered_mat)

        # Create var and obs metadata
        sample_names = [col for col in processed_mat.columns if col != "INDEX"]

        obs = condition_df.loc[sample_names]

        # Final AnnData
        self.adata = ad.AnnData(
            X=X.T,
            obs=obs,
            var=protein_meta_df,
        )

        df = pd.DataFrame(self.adata.X.T, columns=self.adata.obs.index.tolist())

        # Attach layers
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

        # Covariate (centered, imputed) — only if present
        if centered_cov_mat is not None:
            filtered_cov_mat            = getattr(self.preprocessed_data, "raw_covariate", None)
            lognorm_cov_mat            = getattr(self.preprocessed_data, "lognormalized_covariate", None)
            normalized_cov_mat         = getattr(self.preprocessed_data, "normalized_covariate", None)
            processed_cov_mat           = getattr(self.preprocessed_data, "processed_covariate", None)
            qval_cov_mat               = getattr(self.preprocessed_data, "qvalues_covariate", None)
            pep_cov_mat                = getattr(self.preprocessed_data, "pep_covariate", None)
            sc_cov_mat                 = getattr(self.preprocessed_data, "spectral_counts_covariate", None)

            filtered_cov_np, _ = _to_np(filtered_cov_mat)
            lognorm_cov_np, _ = _to_np(lognorm_cov_mat)
            normalized_cov_np, _ = _to_np(normalized_cov_mat)
            processed_cov_np, _ = _to_np(processed_cov_mat)
            centered_cov_np, _ = _to_np(centered_cov_mat)
            qval_cov_np, _ = _to_np(qval_cov_mat)
            pep_cov_np, _ = _to_np(pep_cov_mat)
            sc_cov_np, _ = _to_np(sc_cov_mat)

            self.adata.layers["raw_covariate"] = filtered_cov_np.T
            self.adata.layers["lognorm_covariate"] = lognorm_cov_np.T
            self.adata.layers["normalized_covariate"] = normalized_cov_np.T
            self.adata.layers["processed_covariate"] = processed_cov_np.T
            self.adata.layers["centered_covariate"] = centered_cov_np.T
            self.adata.layers["qval_covariate"] = qval_cov_np.T
            self.adata.layers["pep_covariate"] = pep_cov_np.T
            self.adata.layers["sc_covariate"] = sc_cov_np.T

        # Attach filtered data

        self.adata.uns["preprocessing"] = {
            "input_layout": self.input_layout,
            "analysis_type" : self.analysis_type,
            "filtering": {
                "cont":    self.preprocessed_data.meta_cont,
                "qvalue":  self.preprocessed_data.meta_qvalue,
                "pep":     self.preprocessed_data.meta_pep,
                "prec":     self.preprocessed_data.meta_prec,
            },
            "quantification_method": self.preprocessor.pivot_signal_method,
            "normalization": self.preprocessor.normalization,
            "imputation":    self.preprocessor.imputation,
        }

        # Peptide tables
        sample_names = [c for c in processed_mat.columns if c != "INDEX"]

        # numeric sample cols that exist in peptide tables
        use_cols_w = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_wide_mat.columns]
        use_cols_c = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_cent_mat.columns]

        pep_wide_numeric = pep_wide_mat.select(use_cols_w)
        pep_cent_numeric = pep_cent_mat.select(use_cols_c)

        # matrices (+ row order)
        pep_raw, pep_idx  = polars_matrix_to_numpy(pep_wide_numeric, index_col="PEPTIDE_ID")
        pep_cent, pep_idx2 = polars_matrix_to_numpy(pep_cent_numeric, index_col="PEPTIDE_ID")
        assert list(pep_idx) == list(pep_idx2)

        # row meta (INDEX, PEPTIDE_SEQ) aligned to pep_idx — keep as lists (not DataFrame)
        rowmeta = (
            pep_wide_mat.select(["PEPTIDE_ID","INDEX","PEPTIDE_SEQ"])
                        .unique()
                        .to_pandas()
                        .set_index("PEPTIDE_ID")
                        .loc[pep_idx]
        )
        pep_protein_index = rowmeta["INDEX"].astype(str).tolist()
        peptide_seq   = rowmeta["PEPTIDE_SEQ"].astype(str).tolist()

        # final HDF5-friendly structure
        self.adata.uns["peptides"] = {
            "rows": [str(x) for x in pep_idx],                         # PEPTIDE_ID "{INDEX}|{SEQ}"
            "protein_index": pep_protein_index,                             # one per row
            "peptide_seq": peptide_seq,                                 # one per row
            "cols": [c for c in sample_names if c in pep_wide_mat.columns],  # samples
            "raw":     np.asarray(pep_raw,  dtype=np.float32),          # (n_peptides × n_samples)
            "centered":np.asarray(pep_cent, dtype=np.float32),
        }

        # Store the *analysis* parameters (you can later read these
        # when building your summary in the app)
        self.adata.uns["analysis"] = {
            "de_method":     "limma_ebayes",
            "analysis_type": self.analysis_type,
        }

        # check index is fine between proteins and matrix index
        assert list(protein_meta_df.index) == list(protein_index)

    def get_anndata(self) -> ad.AnnData:
        """Export the processed dataset as an AnnData object."""
        start_time = time.perf_counter()
        return self.adata

if __name__ == "__main__":
    # for testing
    file_path = "searle_test2.tsv"

    # load dataset
    dataset = Dataset(file_path)

    adata = dataset.get_anndata()
    print(adata)

