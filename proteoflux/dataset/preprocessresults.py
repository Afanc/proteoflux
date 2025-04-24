from dataclasses import dataclass
import polars as pl

@dataclass
class PreprocessResults:
    processed: pl.DataFrame
    filtered: pl.DataFrame
    lognormalized: pl.DataFrame
    qvalues: pl.DataFrame
    pep: pl.DataFrame
    condition_pivot: pl.DataFrame
    protein_meta: pl.DataFrame
