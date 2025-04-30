import polars as pl
from typing import Dict, Optional
import polars as pl
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
    }

    def __init__(self, column_config: dict):
        """Initialize column mappings with user-defined config."""
        self.column_map = {}
        self.annotation_file = column_config.get("annotation_file", None)

        for config_key, std_name in self.DEFAULT_COLUMN_MAP.items():
            original_col = column_config.get(config_key)
            if original_col:
                self.column_map[original_col] = std_name


    def _inject_annotation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Injects annotation information into the dataset if missing, using pure Polars."""

        # Prepare annotation for clean matching
        annotation = pl.read_csv(self.annotation_file, separator="\t", ignore_errors=True)
        annotation = annotation.rename({"File Name": "FILENAME"})

        # Strip whitespace
        annotation = annotation.with_columns(pl.col("FILENAME").str.strip_chars())
        df = df.with_columns(pl.col("FILENAME").str.strip_chars())

        # ERROR when any annotation/data filename mismatches. Doesn't make sense to continue
        ann_files = set(annotation.select("FILENAME").unique().to_series().to_list())
        df_files  = set(df.select("FILENAME").unique().to_series().to_list())

        only_in_ann = sorted(ann_files - df_files)
        only_in_df  = sorted(df_files - ann_files)

        if only_in_ann or only_in_df:
            msg_parts = []
            if only_in_ann:
                msg_parts.append(
                    f"Annotation file {self.annotation_file!r} contains {len(only_in_ann)} filenames not present in the data: {only_in_ann}"
                )
            if only_in_df:
                msg_parts.append(
                    f"Data contains {len(only_in_df)} filenames not found in annotation: {only_in_df}"
                )
            full_msg = "Filename mismatch detected:\n  " + "\n  ".join(msg_parts)
            logger.error(full_msg)
            raise ValueError(full_msg)

        # Now safe to join
        df = df.join(annotation, on="FILENAME", how="left")

        # Now conditionally override if missing
        if "CONDITION" in df.columns and "Condition" in df.columns:
            # Print preview before
            df = df.with_columns(
                pl.when(pl.col("Condition").is_not_null())
                  .then(pl.col("Condition"))
                  .otherwise(pl.col("CONDITION"))
                  .alias("CONDITION")
            ).drop("Condition")

        if "REPLICATE" in df.columns and "Replicate" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("Replicate").is_not_null())
                  .then(pl.col("Replicate"))
                  .otherwise(pl.col("REPLICATE"))
                  .alias("REPLICATE")
            ).drop("Replicate")

        return df

    @log_time("Data Harmonizing")
    def harmonize(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename columns to standardized names, skipping any missing ones and warning if needed."""
        rename_map = {}
        for original, target in self.column_map.items():
            if target in df.columns:
                raise ValueError(
                    f"Column name '{target}' already exists in the dataset. "
                    "Please rename your input columns before harmonization."
                )
            if original in df.columns:
                rename_map[original] = target
            else:
                logger.warning(f"Column '{original}' not found in input data, skipping harmonization for it.")

        df = df.rename(rename_map)

        if self.annotation_file:
            df = self._inject_annotation(df)

        return df
