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
        "fasta_column": "FASTA_HEADERS",  # or FASTA_FILES, depending on use
        "protein_weight": "PROTEIN_WEIGHT",
        "protein_descriptions": "PROTEIN_DESCRIPTIONS",
        "gene_names": "GENE_NAMES",
    }

    def __init__(self, column_config: dict):
        """Initialize column mappings with user-defined config."""
        self.column_map = {}

        for config_key, std_name in self.DEFAULT_COLUMN_MAP.items():
            original_col = column_config.get(config_key)
            if original_col:
                self.column_map[original_col] = std_name

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

        return df.rename(rename_map)
