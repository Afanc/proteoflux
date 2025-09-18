import sys
import pyarrow.csv as pv_csv
import polars as pl
import pandas as pd
import numpy as np
import time
import anndata as ad
from typing import List, Optional, Tuple, Union
import warnings

from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.utils.utils import polars_matrix_to_numpy, log_time, logger

pl.Config.set_tbl_rows(100)
# Supnress the ImplicitModificationWarning from AnnData
warnings.filterwarnings("ignore", category=UserWarning, message=".*Transforming to str index.*")

class Dataset:
    """The main class for processing the dataset, loading raw data, and converting to AnnData."""
    def __init__(self, **kwargs):
        """
        Initialize the dataset object.

        Args:
            kwargs: dict with all the config elements
        """

        # Dataset-specific config
        dataset_cfg = kwargs.get("dataset", {})
        self.file_path = dataset_cfg.get("input_file", None)
        self.load_method = dataset_cfg.get("load_method", 'polars')
        self.input_layout = dataset_cfg.get("input_layout", "long")
        self.analysis_type = dataset_cfg.get("analysis_type", "DIA")

        # Harmonizer setup
        self.harmonizer = DataHarmonizer(dataset_cfg)

        # Preprocessing config and setup
        preprocessing_cfg = kwargs.get("preprocessing", {})
        self.preprocessor = Preprocessor(preprocessing_cfg)

        # Process
        self._load_and_process()

    def _load_and_process(self):

        # Load data
        self.rawinput = self._load_rawdata(self.file_path)

        # Harmonize data
        self.rawinput = self.harmonizer.harmonize(self.rawinput)

        # Apply preprocessing
        self.preprocessed_data = self._apply_preprocessing(self.rawinput)

        # Convert to AnnData format
        self._convert_to_anndata()

    @log_time("Data Loading")
    def _load_rawdata(self, file_path: str) -> Union[pl.DataFrame, pd.DataFrame]:
        """Load raw data from a CSV or TSV file using different libraries."""
        if not file_path.endswith((".csv", ".tsv")):
            raise ValueError("Only CSV or TSV files are supported.")

        delimiter = "\t" if file_path.endswith(".tsv") else ","

        if self.load_method == 'polars':
            # Use Polars to load the data, like a normal person
            df = pl.read_csv(file_path,
                             separator=delimiter,
                             infer_schema_length=10000,
                             null_values=["NA", "NaN", "N/A", ""])
            return df
        elif self.load_method == 'pyarrow':
            # Use pyarrow, eh it's good but slow
            parse_options = pv_csv.ParseOptions(delimiter=delimiter)
            arrow_table = pv_csv.read_csv(file_path, parse_options=parse_options)
            return pl.from_arrow(arrow_table)
        elif self.load_method == 'pandas':
            # Use pandas to load the data, why would I ever do this
            df = pd.read_csv(file_path, delimiter=delimiter)
            return pl.from_pandas(df)
        else:
            raise ValueError(f"Unknown load method: {self.load_method}")

    @log_time("Data Processing")
    def _apply_preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Placeholder for preprocessing steps like normalization, imputation, etc.

        Args:
            df (pl.DataFrame): The raw input data to preprocess.

        Returns:
            pl.DataFrame: The preprocessed data.
        """
        preprocessed_results = self.preprocessor.fit_transform(df)

        return preprocessed_results

    @log_time("Conversion to AnnData")
    def _convert_to_anndata(self):
        """Convert the data to an AnnData object for downstream analysis."""

        # Extract matrices
        # store unfiltered ?
        filtered_mat = self.preprocessed_data.filtered #filtered before log
        lognorm_mat = self.preprocessed_data.lognormalized #post log
        normalized_mat = self.preprocessed_data.normalized # post norm
        processed_mat = self.preprocessed_data.processed # post impu
        qval_mat = self.preprocessed_data.qvalues
        pep_mat = self.preprocessed_data.pep
        sc_mat = self.preprocessed_data.spectral_counts
        condition_df = self.preprocessed_data.condition_pivot.to_pandas().set_index("Sample")

        pep_wide_mat  = self.preprocessed_data.peptides_wide
        pep_cent_mat  = self.preprocessed_data.peptides_centered

        protein_meta_df = self.preprocessed_data.protein_meta.to_pandas().set_index("INDEX")

        # Use filtered_mat to infer sample names (columns) and protein IDs (rows)
        X, protein_index = polars_matrix_to_numpy(processed_mat, index_col="INDEX")

        # Ensure the metadata index matches the order of protein_index from X
        protein_meta_df = protein_meta_df.loc[protein_index]

        # Layers
        def _to_np(opt_df):
            return polars_matrix_to_numpy(opt_df, index_col="INDEX")

        qval, _    = _to_np(qval_mat)
        pep, _    = _to_np(pep_mat)
        sc, _    = _to_np(sc_mat)
        lognorm, _ = _to_np(lognorm_mat)
        normalized, _ = _to_np(normalized_mat)
        raw, _  = _to_np(filtered_mat)

        # Create var and obs metadata
        sample_names = [col for col in processed_mat.columns if col != "INDEX"]

        obs = condition_df.loc[sample_names]

        # Final AnnData
        self.adata = ad.AnnData(
            X=X.T,
            obs=obs,
            var=protein_meta_df,
        )

        df = pd.DataFrame(self.adata.X.T, columns=self.adata.obs.index.tolist())

        # Attach layers
        self.adata.layers["raw"] = raw.T
        self.adata.layers["lognorm"] = lognorm.T
        self.adata.layers["normalized"] = normalized.T
        if qval is not None:
            self.adata.layers["qvalue"] = qval.T
        if pep is not None:
            self.adata.layers["pep"] = pep.T
        if sc is not None:
            self.adata.layers["spectral_counts"] = sc.T

        # Attach filtered data

        self.adata.uns["preprocessing"] = {
            "input_layout": self.input_layout,
            "analysis_type" : self.analysis_type,
            "filtering": {
                "cont":    self.preprocessed_data.meta_cont,
                "qvalue":  self.preprocessed_data.meta_qvalue,
                "pep":     self.preprocessed_data.meta_pep,
                "rec":     self.preprocessed_data.meta_rec,
            },
            "normalization": self.preprocessor.normalization,
            "imputation":    self.preprocessor.imputation,
        }

        # Peptide tables
        sample_names = [c for c in processed_mat.columns if c != "INDEX"]

        # numeric sample cols that exist in peptide tables
        use_cols_w = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_wide_mat.columns]
        use_cols_c = ["PEPTIDE_ID"] + [c for c in sample_names if c in pep_cent_mat.columns]

        pep_wide_numeric = pep_wide_mat.select(use_cols_w)
        pep_cent_numeric = pep_cent_mat.select(use_cols_c)

        # matrices (+ row order)
        pep_raw, pep_idx  = polars_matrix_to_numpy(pep_wide_numeric, index_col="PEPTIDE_ID")
        pep_cent, pep_idx2 = polars_matrix_to_numpy(pep_cent_numeric, index_col="PEPTIDE_ID")
        assert list(pep_idx) == list(pep_idx2)

        # row meta (INDEX, PEPTIDE_SEQ) aligned to pep_idx — keep as lists (not DataFrame)
        rowmeta = (
            pep_wide_mat.select(["PEPTIDE_ID","INDEX","PEPTIDE_SEQ"])
                        .unique()
                        .to_pandas()
                        .set_index("PEPTIDE_ID")
                        .loc[pep_idx]
        )
        pep_protein_index = rowmeta["INDEX"].astype(str).tolist()
        peptide_seq   = rowmeta["PEPTIDE_SEQ"].astype(str).tolist()

        # final HDF5-friendly structure
        self.adata.uns["peptides"] = {
            "rows": [str(x) for x in pep_idx],                         # PEPTIDE_ID "{INDEX}|{SEQ}"
            "protein_index": pep_protein_index,                             # one per row
            "peptide_seq": peptide_seq,                                 # one per row
            "cols": [c for c in sample_names if c in pep_wide_mat.columns],  # samples
            "raw":     np.asarray(pep_raw,  dtype=np.float32),          # (n_peptides × n_samples)
            "centered":np.asarray(pep_cent, dtype=np.float32),
        }

        # Store the *analysis* parameters (you can later read these
        # when building your summary in the app)
        self.adata.uns["analysis"] = {
            "de_method":     "limma_ebayes",
        }

        # check index is fine between proteins and matrix index
        assert list(protein_meta_df.index) == list(protein_index)

    def get_anndata(self) -> ad.AnnData:
        """Export the processed dataset as an AnnData object."""
        start_time = time.perf_counter()
        return self.adata

if __name__ == "__main__":
    # for testing
    file_path = "searle_test2.tsv"

    # load dataset
    dataset = Dataset(file_path)

    adata = dataset.get_anndata()
    print(adata)

