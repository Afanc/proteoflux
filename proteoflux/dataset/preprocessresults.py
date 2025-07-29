from dataclasses import dataclass
import polars as pl

@dataclass
class PreprocessResults:
    lognormalized: pl.DataFrame
    normalized: pl.DataFrame
    processed: pl.DataFrame
    filtered: pl.DataFrame
    qvalues: pl.DataFrame
    pep: pl.DataFrame
    condition_pivot: pl.DataFrame
    protein_meta: pl.DataFrame
    removed_contaminants: pl.DataFrame #wrong, dict, correct later
    removed_qvalue: pl.DataFrame
    removed_pep: pl.DataFrame
    removed_RE: pl.DataFrame
