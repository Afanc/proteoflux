from dataclasses import dataclass
import polars as pl
from typing import Dict

@dataclass
class PreprocessResults:
    lognormalized: pl.DataFrame
    normalized: pl.DataFrame
    processed: pl.DataFrame
    filtered: pl.DataFrame
    qvalues: pl.DataFrame
    pep: pl.DataFrame
    spectral_counts: pl.DataFrame
    condition_pivot: pl.DataFrame
    protein_meta: pl.DataFrame
    peptides_wide: pl.DataFrame
    peptides_centered: pl.DataFrame
    meta_cont: Dict
    meta_qvalue: Dict
    meta_pep: Dict
    meta_rec: Dict
