from dataclasses import dataclass
import polars as pl
from typing import Dict, Optional

@dataclass
class PreprocessResults:
    lognormalized: pl.DataFrame
    normalized: pl.DataFrame
    processed: pl.DataFrame
    filtered: pl.DataFrame
    qvalues: Optional[pl.DataFrame]          # <- optional
    pep: Optional[pl.DataFrame]              # <- optional
    spectral_counts: Optional[pl.DataFrame]  # <- optional
    condition_pivot: pl.DataFrame
    protein_meta: pl.DataFrame
    peptides_wide: Optional[pl.DataFrame]    # <- optional
    peptides_centered: Optional[pl.DataFrame]# <- optional
    meta_cont: Dict
    meta_qvalue: Dict
    meta_pep: Dict
    meta_rec: Dict
