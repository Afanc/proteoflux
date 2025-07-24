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
            df = pl.read_csv(file_path, separator=delimiter, infer_schema_length=10000)
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
        processed_mat = self.preprocessed_data.processed # post norm
        qval_mat = self.preprocessed_data.qvalues
        pep_mat = self.preprocessed_data.pep
        condition_df = self.preprocessed_data.condition_pivot.to_pandas().set_index("Sample")

        protein_meta_df = self.preprocessed_data.protein_meta.to_pandas().set_index("INDEX")

        # Use filtered_mat to infer sample names (columns) and protein IDs (rows)
        X, protein_index = polars_matrix_to_numpy(processed_mat, index_col="INDEX")

        # Ensure the metadata index matches the order of protein_index from X
        protein_meta_df = protein_meta_df.loc[protein_index]

        # Layers
        qval, _ = polars_matrix_to_numpy(qval_mat, index_col="INDEX")
        pep, _ = polars_matrix_to_numpy(pep_mat, index_col="INDEX")
        lognorm, _ = polars_matrix_to_numpy(lognorm_mat, index_col="INDEX")
        raw, _ = polars_matrix_to_numpy(filtered_mat, index_col="INDEX")

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
        self.adata.layers["qvalue"] = qval.T
        self.adata.layers["pep"] = pep.T

        # Attach filtered data
        if self.preprocessed_data.removed_contaminants is not None:
            self.adata.uns["removed_contaminants"] = self.preprocessed_data.removed_contaminants
        if self.preprocessed_data.removed_qvalue is not None:
            self.adata.uns["removed_qvalue"] = self.preprocessed_data.removed_qvalue
        if self.preprocessed_data.removed_pep is not None:
            self.adata.uns["removed_pep"] = self.preprocessed_data.removed_pep
        if self.preprocessed_data.removed_RE is not None:
            self.adata.uns["removed_RE"] = self.preprocessed_data.removed_RE

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

