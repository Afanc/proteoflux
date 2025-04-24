import sys
import polars as pl
import pandas as pd
import numpy as np
import anndata as ad
from typing import Tuple
from pathlib import Path
from functools import wraps
import time
import logging
from contextlib import contextmanager
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

_thread_local = threading.local()

@contextmanager
def log_indent():
    if not hasattr(_thread_local, "indent"):
        _thread_local.indent = 0
    _thread_local.indent += 2
    try:
        yield
    finally:
        _thread_local.indent -= 2

def log_time(task_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            indent = getattr(_thread_local, "indent", 0)
            logger.info(" " * indent + f"{task_name} started")
            with log_indent():
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(" " * indent + f"{task_name} completed, took {duration:.2f} seconds.")
            return result
        return wrapper
    return decorator

def polars_matrix_to_numpy(df: pl.DataFrame, index_col: str = "INDEX") -> Tuple[np.ndarray, pd.Index]:
    """
    Converts a Polars matrix with an explicit index column (e.g., 'INDEX') to:
    - NumPy matrix (rows = proteins, cols = samples)
    - Protein index (row labels for AnnData.var)

    Args:
        df (pl.DataFrame): Input matrix with index column.
        index_col (str): The name of the column to use as index (default: 'INDEX').

    Returns:
        Tuple[np.ndarray, pd.Index]: Matrix as numpy array, and index labels.
    """
    if index_col not in df.columns:
        raise ValueError(f"Expected index column '{index_col}' not found in DataFrame.")

    sample_cols = [col for col in df.columns if col != index_col]
    matrix = df.select(sample_cols).to_numpy()
    index = df.get_column(index_col).to_pandas()

    return matrix, index

def load_contaminant_accessions(fasta_path: str | Path) -> set[str]:
    accessions = set()
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                acc = line.split("|")[1]  # sp|Q13515|...
                accessions.add(acc.strip())
    return accessions

def debug_protein_view(adata, protein_id):
    """
    Print raw/lognorm expression values and DE stats for a single protein.
    Includes per-condition means. `protein_id` should match adata.var.index.
    """
    if protein_id not in adata.var.index:
        print(adata.var.index)
        raise ValueError(f"Protein {protein_id} not found in adata.var.index")

    idx = list(adata.var.index).index(protein_id)
    contrast_names = adata.uns['contrast_names']
    condition_labels = adata.obs["CONDITION"].values

    raw_vals = adata.layers["raw"][:, idx]
    lognorm_vals = adata.layers["lognorm"][:, idx]
    model_vals = adata.X[:, idx]

    print(f"\n🧬 Protein: {protein_id}")
    print("=" * 50)

    print("Raw Intensity (per sample):")
    print(raw_vals)

    print("\nLog2-normalized Intensity (adata.layers['lognorm']):")
    print(lognorm_vals)

    print("\nFinal Intensity (adata.X):")
    print(model_vals)

    # 🧮 Per-condition means
    def cond_means(name, values):
        df = pd.DataFrame({"value": values, "condition": condition_labels})
        means = df.groupby("condition", observed=False)["value"].mean()
        print(f"\n▶ {name} mean per condition:")
        print(means)

    cond_means("Raw", raw_vals)
    cond_means("Log2-normalized", lognorm_vals)
    cond_means("Final (model input)", model_vals)

    print("\nlog2FC (per contrast):")
    print(pd.Series(adata.varm['log2fc'][idx], index=contrast_names))

    print("\nt-statistics:")
    print(pd.Series(adata.varm['t'][idx], index=contrast_names))

    print("\np-values:")
    print(pd.Series(adata.varm['p'][idx], index=contrast_names))

    print("\nq-values:")
    print(pd.Series(adata.varm['q'][idx], index=contrast_names))

    print("\nadjusted p-values:")
    print(pd.Series(adata.varm['p_ebayes'][idx], index=contrast_names))

    print("\nadjusted q-values:")
    print(pd.Series(adata.varm['q_ebayes'][idx], index=contrast_names))
