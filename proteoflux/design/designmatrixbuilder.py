import pandas as pd
import numpy as np
import patsy
from typing import Optional, Dict, Any, List
from proteoflux.utils.utils import logger, log_time

class DesignMatrixBuilder:
    def __init__(
        self,
        sample_metadata: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.meta = sample_metadata.copy()
        self.config = config or {}
        self.formula: Optional[str] = None
        self.design_matrix: Optional[np.ndarray] = None
        self.design_info: Optional[patsy.DesignInfo] = None

    def build(self):
        mode = self.config.get("mode", "default")
        if mode == "default":
            self._build_default_design()
        elif mode == "paired":
            self._build_paired_design()
        elif mode == "timecourse":
            self._build_timecourse_design()
        else:
            raise ValueError(f"Unknown design mode: {mode}")

        return self.design_matrix, self.design_info

    def _build_default_design(self):
        group_col = self.config.get("group_column", "Condition")
        if group_col not in self.meta.columns:
            raise ValueError(f"{group_col} not found in sample metadata.")

        self.meta[group_col] = self.meta[group_col].astype("category")
        self.formula = f"1 + C({group_col})"
        self.design_df = patsy.dmatrix(self.formula, self.meta, return_type="dataframe")

        self.design_matrix = self.design_df.to_numpy()
        self.design_info = self.design_df.design_info

#    def _build_paired_design(self):
#        group_col = self.config.get("group_column", "Condition")
#        pair_col = self.config.get("pair_column", "Subject")
#        if group_col not in self.meta.columns or pair_col not in self.meta.columns:
#            raise ValueError(f"{group_col} or {pair_col} not found in sample metadata.")
#
#        self.meta[group_col] = self.meta[group_col].astype("category")
#        self.meta[pair_col] = self.meta[pair_col].astype("category")
#        self.formula = f"1 + C({group_col}) + C({pair_col})"
#        self.design_matrix = patsy.dmatrix(self.formula, self.meta, return_type="dataframe")
#        self.design_info = self.design_matrix.design_info
#
#    def _build_timecourse_design(self):
#        group_col = self.config.get("group_column", "Condition")
#        time_col = self.config.get("time_column", "Time")
#        if group_col not in self.meta.columns or time_col not in self.meta.columns:
#            raise ValueError(f"{group_col} or {time_col} not found in sample metadata.")
#
#        self.formula = f"1 + C({group_col}) + {time_col} + C({group_col}):{time_col}"
#        self.design_matrix = patsy.dmatrix(self.formula, self.meta, return_type="dataframe")
#        self.design_info = self.design_matrix.design_info
#
