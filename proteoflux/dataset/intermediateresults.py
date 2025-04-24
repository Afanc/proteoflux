from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import polars as pl

@dataclass
class IntermediateResults:
    # Store matrices at various stages of preprocessing
    matrices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Store Polars DataFrames at various stages
    dfs: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Store normalization & imputation models
    models: Dict[str, Any] = field(default_factory=lambda: {"normalization": {}, "imputation": {}})

    # Metadata for normalization and imputation steps
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "filtering": {},
        "normalization": {},
        "imputation": {}})

    # DataFrame columns post-pivot (samples)
    columns: Optional[list] = None

    # Index columns (proteins) used for matching consistently across DataFrames
    index: Optional[np.ndarray] = None

    def set_columns_and_index(self, df: pl.DataFrame, index_col: str = "INDEX"):
        """Set columns and index once from a pivoted DataFrame."""
        self.columns = [col for col in df.columns if col != index_col]
        self.index = df.select(index_col).to_series().to_numpy()

    def add_array(self, name: str, array: np.ndarray):
        """Store lightweight numeric arrays (e.g., raw values used for filtering plots)."""
        if not hasattr(self, "arrays"):
            self.arrays = {}
        self.arrays[name] = array

    def add_matrix(self, name: str, matrix: np.ndarray):
        """Add a matrix with automatic shape validation."""
        if self.index is not None and matrix.shape[0] != len(self.index):
            raise ValueError(f"Matrix '{name}' has inconsistent row dimension.")
        self.matrices[name] = matrix

    def add_df(self, name: str, df: pl.DataFrame, index_col: str = "INDEX"):
        """Add a DataFrame ensuring correct columns and indices."""
        if self.index is not None:
            df_index = df.select(index_col).to_series().to_numpy()
            if not np.array_equal(df_index, self.index):
                raise ValueError(f"DataFrame '{name}' index does not match stored index.")
        self.dfs[name] = df

    def add_metadata(self, step: str, key: str, value: Any):
        """Store metadata like regression type, scale, etc."""
        if step not in ["filtering", "normalization", "imputation"]:
            raise ValueError("step must be 'normalization' or 'imputation'")
        self.metadata[step][key] = value

    def add_model(self, step: str, model: Any):
        """Add normalization or imputation model."""
        if step not in ["normalization", "imputation"]:
            raise ValueError("step must be 'normalization' or 'imputation'")
        self.models[step] = model

