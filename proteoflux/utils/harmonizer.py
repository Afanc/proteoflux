import polars as pl
from typing import Dict, Optional
from proteoflux.utils.utils import log_time, logger

class DataHarmonizer:
    """Harmonizes input data by renaming columns to a common format."""

    DEFAULT_COLUMN_MAP = {
        "index_column": "INDEX",
        "signal_column": "SIGNAL",
        "qvalue_column": "QVALUE",
        "pep_column": "PEP",
        "condition_column": "CONDITION",
        "replicate_column": "REPLICATE",
        "filename_column": "FILENAME",
        "run_evidence_column": "RUN_EVIDENCE_COUNT",
        "fasta_column": "FASTA_HEADERS",
        "protein_weight": "PROTEIN_WEIGHT",
        "protein_descriptions": "PROTEIN_DESCRIPTIONS",
        "gene_names": "GENE_NAMES",
        "precursors_exp_column": "PRECURSORS_EXP",
        "ibaq_column": "IBAQ",
        "peptide_seq_column": "PEPTIDE_LSEQ",
    }

    def __init__(self, column_config: dict):
        """Initialize column mappings with user-defined config."""
        self.column_map: Dict[str, str] = {}
        self.annotation_file = column_config.get("annotation_file", None)
        # New: explicit input format (spectronaut default; fragpipe_tmt supported)
        self.input_format = (column_config.get("format") or "").strip().lower()

        for config_key, std_name in self.DEFAULT_COLUMN_MAP.items():
            original_col = column_config.get(config_key)
            if original_col:
                self.column_map[original_col] = std_name

    # ---------- helpers ----------

    def _load_annotation(self) -> pl.DataFrame:
        if not self.annotation_file:
            raise ValueError(
                "annotation_file is required for this data format but was not provided."
            )
        ann = pl.read_csv(self.annotation_file, separator="\t", ignore_errors=True)

        # Normalize to a single join key "FILENAME":
        # - Spectronaut-style sheets often use "File Name"
        # - FragPipe TMT channel maps may use "Channel"
        if "FILENAME" not in ann.columns:
            if "File Name" in ann.columns:
                ann = ann.rename({"File Name": "FILENAME"})
            elif "Channel" in ann.columns:
                ann = ann.rename({"Channel": "FILENAME"})

        if "FILENAME" not in ann.columns:
            raise ValueError(
                f"Annotation file {self.annotation_file!r} must contain either "
                "'FILENAME', 'File Name', or 'Channel' column."
            )

        # Be tolerant about whitespace
        ann = ann.with_columns(pl.col("FILENAME").cast(pl.Utf8).str.strip_chars())

        return ann

    def _inject_annotation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Injects annotation information into the dataset if missing, using pure Polars."""
        annotation = self._load_annotation()

        # Strip whitespace on DF key as well
        if "FILENAME" in df.columns:
            df = df.with_columns(pl.col("FILENAME").cast(pl.Utf8).str.strip_chars())
        else:
            raise ValueError("Cannot inject annotation: 'FILENAME' column not found in data.")

        # Hard check 1: all DF filenames have annotation
        ann_files = set(annotation.select("FILENAME").unique().to_series().to_list())
        df_files = set(df.select("FILENAME").unique().to_series().to_list())

        only_in_ann = sorted(ann_files - df_files)
        only_in_df = sorted(df_files - ann_files)

        if only_in_ann or only_in_df:
            msg_parts = []
            if only_in_ann:
                msg_parts.append(
                    f"Annotation file {self.annotation_file!r} contains {len(only_in_ann)} names not present in data: {only_in_ann}"
                )
            if only_in_df:
                msg_parts.append(
                    f"Data contains {len(only_in_df)} names not found in annotation: {only_in_df}"
                )
            full_msg = "Filename mismatch detected:\n  " + "\n  ".join(msg_parts)
            logger.error(full_msg)
            raise ValueError(full_msg)

        # Safe to join
        df = df.join(annotation, on="FILENAME", how="left")

        # Coalesce condition/replicate from various header styles into canonical uppercase
        # Prefer explicit R.* columns if present (common in DIA sheets)
        # Fall back to plain "Condition"/"Replicate" if those exist
        condition_sources = [c for c in ["R.Condition", "Condition"] if c in df.columns]
        replicate_sources = [r for r in ["R.Replicate", "Replicate"] if r in df.columns]

        if "CONDITION" in df.columns or condition_sources:
            if condition_sources:
                src = condition_sources[0]
                df = df.with_columns(
                    pl.when(pl.col(src).is_not_null()).then(pl.col(src)).otherwise(pl.col("CONDITION") if "CONDITION" in df.columns else None).alias("CONDITION")
                )
                # Drop the auxiliary column to avoid ambiguity
                df = df.drop(src, strict=False)

        if "REPLICATE" in df.columns or replicate_sources:
            if replicate_sources:
                src = replicate_sources[0]
                df = df.with_columns(
                    pl.when(pl.col(src).is_not_null()).then(pl.col(src)).otherwise(pl.col("REPLICATE") if "REPLICATE" in df.columns else None).alias("REPLICATE")
                )
                df = df.drop(src, strict=False)

        return df

    def _harmonize_fragpipe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        FragPipe TMT path:
        - Use annotation 'Channel'/'FILENAME' as source of truth for channel columns
        - Melt wide FP table to canonical long with FILENAME + SIGNAL
        - Then reuse the standard renaming + annotation injection
        """
        logger.info("Detected format=fragpipe; harmonizing via melt → annotate.")

        # 1) Load annotation and derive channel list
        ann = self._load_annotation()
        channels = ann.select("FILENAME").unique().to_series().to_list()

        # 2) Validate channels exist in FP data
        df_cols = set(df.columns)
        missing_in_data = sorted([c for c in channels if c not in df_cols])
        extra_in_data = sorted([c for c in df_cols if c in channels and c not in channels])  # kept for symmetry

        if missing_in_data:
            msg = (
                f"{len(missing_in_data)} annotated channels are not present in the FragPipe table: {missing_in_data}. "
                "Make sure the annotation 'Channel' (or 'FILENAME') matches the FP column headers exactly."
            )
            logger.error(msg)
            raise ValueError(msg)

        # 3) Melt: id_vars = everything except the annotated channels
        id_vars = [c for c in df.columns if c not in channels]
        if not id_vars:
            raise ValueError("No identifier columns left after selecting channels; check your annotation and FP table.")

        long_df = df.melt(
            id_vars=id_vars,
            value_vars=channels,
            variable_name="FILENAME",
            value_name="SIGNAL",
        )

        # Ensure types are sane
        long_df = long_df.with_columns(
            pl.col("FILENAME").cast(pl.Utf8),
            pl.col("SIGNAL").cast(pl.Float64, strict=False),
        )

        rename_fp = {}
        if "Spectrum Number" in long_df.columns:
            rename_fp["Spectrum Number"] = "PRECURSORS_EXP"
        # (We deliberately do NOT introduce NUMBER_PSM; keep one canonical field)

        if rename_fp:
            long_df = long_df.rename(rename_fp)

        # Tag rows so downstream can do FP-specific aggregation where needed
        long_df = long_df.with_columns(pl.lit("fragpipe").alias("SOURCE"))

        # 4) Standard rename of metadata columns (keeps Spectronaut-compatible names identical)
        long_df = self._rename_columns_safely(long_df)

        # 5) Inject annotation (adds CONDITION/REPLICATE/etc.)
        long_df = self._inject_annotation(long_df)

        logger.info(
            f"FragPipe TMT melt completed: rows={long_df.height}, cols={long_df.width}. "
            "Capabilities: has_qvalue=False, has_run_evidence=False, pep_direction=higher, input_is_log2=True."
        )
        return long_df

    def _rename_columns_safely(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rename columns to standardized names from config, but only raise if we are
        actually renaming *into* an already-existing different column name.

        This avoids false positives when a standardized column (e.g., FILENAME) already
        exists naturally (like after melting).
        """
        rename_map: Dict[str, str] = {}

        for original, target in self.column_map.items():
            if original in df.columns:
                # If we intend to rename, ensure we won't clobber a different existing column
                if target in df.columns and original != target:
                    raise ValueError(
                        f"Cannot rename '{original}' to standardized '{target}' because "
                        f"'{target}' already exists in the dataset."
                    )
                rename_map[original] = target
            else:
                # Keep the prior behavior (warn if user configured a column that isn't present)
                logger.warning(f"Column '{original}' not found in input data, skipping harmonization for it.")

        return df.rename(rename_map) if rename_map else df

    # ---------- public API ----------

    @log_time("Data Harmonizing")
    def harmonize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rename columns to standardized names, and (optionally) inject annotation.
        For FragPipe TMT format, melt wide channels to long first, then proceed.
        """
        if self.input_format == "fragpipe":
            # Full FP path (melt → rename → inject)
            return self._harmonize_fragpipe(df)

        # Default path (Spectronaut etc.): rename-in-place, then (optionally) inject
        df = self._rename_columns_safely(df)

        if self.annotation_file:
            df = self._inject_annotation(df)

        return df
