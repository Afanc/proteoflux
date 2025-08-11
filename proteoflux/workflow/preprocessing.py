import time
import sys
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

import sklearn.impute
import sklearn.preprocessing
import sklearn.ensemble
from skmisc.loess import loess

from proteoflux.workflow.normalizers.regression_normalization import regression_normalization
from proteoflux.workflow.imputer_factory import get_imputer
from proteoflux.dataset.preprocessresults import PreprocessResults
from proteoflux.dataset.intermediateresults import IntermediateResults
from proteoflux.evaluation.normalization_evaluator import NormalizerPlotter
from proteoflux.evaluation.imputation_evaluator import ImputeEvaluator
from proteoflux.utils.utils import load_contaminant_accessions, log_time, logger

class Preprocessor:
    """Handles filtering, normalization, and imputation for proteomics data."""

    available_normalization = ["zscore", "vst", "linear"]
    available_imputation = ["mean", "median", "knn", "randomforest"]

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config (dict) with all the params
        """

        self.intermediate_results = IntermediateResults()

        # Filtering
        self.filtering = config.get("filtering")
        self.remove_contaminants = self.filtering.get("contaminants_files", [])
        self.filter_qvalue = self.filtering.get("qvalue", 0.01)
        self.filter_pep = self.filtering.get("pep", 0.2)
        self.filter_run_evidence_count = self.filtering.get("run_evidence_count", 1)

        # Pivoting
        self.pivot_signal_method = config.get("pivot_signal_method", "sum")

        # Normalization
        self.normalization = config.get("normalization")

        # Imputation
        self.imputation = config.get("imputation")

        self.exports = config.get("exports")

        # Plotting
        #self.plot_normalization = self.exports.get("normalization_plots")
        #self.plot_imputation = self.exports.get("imputation_plots")

        #self.normalization_plot_path = self.exports.get("normalization_plot_path")
        #self.imputation_plot_path = self.exports.get("imputation_plot_path")

    def fit_transform(self, df: pl.DataFrame) -> PreprocessResults:
        """Applies preprocessing steps in sequence and returns structured results."""
        # Step 1: Filtering
        self._filter(df)

        # Fetch protein metadata
        self._get_protein_metadata()

        # Fech sample metadata
        self._get_condition_map()

        # Step 2: Pivoting
        self._pivot_data()

        # Step 3: Normalization
        self._normalize()

        # Step 4: Imputation
        self._impute()

        # Final step: returning the structured data class
        return PreprocessResults(
            filtered=self.intermediate_results.dfs.get("raw_df"),
            lognormalized=self.intermediate_results.dfs.get("postlog"),
            normalized=self.intermediate_results.dfs.get("normalized"),
            processed=self.intermediate_results.dfs.get("imputed"),
            qvalues=self.intermediate_results.dfs.get("qvalues"),
            pep=self.intermediate_results.dfs.get("pep"),
            condition_pivot=self.intermediate_results.dfs.get("condition_pivot"),
            protein_meta=self.intermediate_results.dfs.get("protein_metadata"),
            meta_cont = self.intermediate_results.metadata.get("filtering").get("meta_cont"),
            meta_qvalue = self.intermediate_results.metadata.get("filtering").get("meta_qvalue"),
            meta_pep =  self.intermediate_results.metadata.get("filtering").get("meta_pep"),
            meta_rec =  self.intermediate_results.metadata.get("filtering").get("meta_rec"),
        )

    @log_time("Filtering")
    def _filter(self, df: pl.DataFrame) -> None:
        df_filtered = self._filter_contaminants(df)
        self.intermediate_results.add_df("filtered_contaminants", df_filtered)

        df_filtered = self._filter_by_stat(df_filtered, "QVALUE")
        self.intermediate_results.add_df("filtered_QVALUE", df_filtered)

        df_filtered = self._filter_by_stat(df_filtered, "PEP")
        self.intermediate_results.add_df("filtered_PEP", df_filtered)

        df_filtered = self._filter_by_run_evidence(df_filtered)
        self.intermediate_results.add_df("filtered_final/RE", df_filtered)

    def _filter_contaminants(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.remove_contaminants or "INDEX" not in df.columns:
            return df

        # Load all contaminant accessions
        all_contaminants = set()
        for path in self.remove_contaminants:
            all_contaminants |= load_contaminant_accessions(path)

        mask_keep = ~pl.col("INDEX").is_in(list(all_contaminants))

        # keep & drop
        df_kept    = df.filter(mask_keep)
        df_dropped = df.filter(~mask_keep)
        mask_bools = df.select(mask_keep.alias("keep")).get_column("keep").to_list()

        dropped_dict = {
            "files": self.remove_contaminants,
            "values": mask_bools,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }

        self.intermediate_results.add_metadata(
            "filtering",
            "meta_cont",
            dropped_dict
        )

        return df_kept

    def _filter_by_stat(self, df: pl.DataFrame, stat: str) -> pl.DataFrame:
        """Filters out rows based on Q-value and PEP thresholds."""
        if stat not in ["PEP", "QVALUE"]:
            raise ValueError(f"Invalid filtering stat '{stat}'")

        if stat in df.columns:
            threshold = self.filter_qvalue if stat == "QVALUE" else self.filter_pep
            values = df[stat].to_numpy()
            self.intermediate_results.add_array(f"{stat.lower()}_array", values)

            df_kept    = df.filter(df[stat] <= threshold)
            df_dropped = df.filter(df[stat] >  threshold)
            dropped_dict = {
                "threshold": threshold,
                "raw_values": values,
                "number_kept": len(df_kept),
                "number_dropped": len(df_dropped),
            }

            self.intermediate_results.add_metadata("filtering", f"meta_{stat.lower()}", dropped_dict)

        return df_kept

    def _filter_by_run_evidence(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out proteins with insufficient run evidence."""
        if "RUN_EVIDENCE_COUNT" in df.columns:
            values = df["RUN_EVIDENCE_COUNT"].to_numpy()
            self.intermediate_results.add_array("run_evidence_count_array", values)

            df_kept    = df.filter(pl.col("RUN_EVIDENCE_COUNT") >= self.filter_run_evidence_count)
            df_dropped = df.filter(pl.col("RUN_EVIDENCE_COUNT") <  self.filter_run_evidence_count)
            dropped_dict = {
                "threshold": self.filter_run_evidence_count,
                "raw_values": values,
                "number_kept": len(df_kept),
                "number_dropped": len(df_dropped),
            }

            self.intermediate_results.add_metadata("filtering", f"meta_rec", dropped_dict)

        return df_kept

    def _get_protein_metadata(self) -> None:
        """
        Extract protein-level metadata such as FASTA_HEADERS, GENE_NAMES, DESCRIPTION, etc.,
        grouped by protein INDEX. Assumes all column names are capitalized.
        """
        df = self.intermediate_results.dfs["filtered_final/RE"]
        keep_cols = {"INDEX",
                     "FASTA_HEADERS",
                     "GENE_NAMES",
                     "PROTEIN_DESCRIPTIONS",
                     "PROTEIN_WEIGHT",
                     "RUN_EVIDENCE_COUNT",
                     "PG.IBAQ"}
        existing = [col for col in df.columns if col in keep_cols]

        # --- make sure numeric columns are truly numeric ---
        df = df.select(existing)
        casts = []
        if "PG.IBAQ" in df.columns:
            casts.append(pl.col("PG.IBAQ").cast(pl.Float64, strict=False))       # non-numeric -> null
        if "PROTEIN_WEIGHT" in df.columns:
            casts.append(pl.col("PROTEIN_WEIGHT").cast(pl.Float64, strict=False))
        if "RUN_EVIDENCE_COUNT" in df.columns:
            casts.append(pl.col("RUN_EVIDENCE_COUNT").cast(pl.Int64, strict=False))
        if casts:
            df = df.with_columns(casts)

        df = df.select(existing).unique(subset=["INDEX"]).sort("INDEX")

        self.intermediate_results.add_df("protein_metadata", df)

    def _get_condition_map(self):
        df = self.intermediate_results.dfs["filtered_final/RE"]
        condition_mapping = df.select(["FILENAME", "CONDITION", "REPLICATE"]).unique()
        condition_mapping = condition_mapping.rename({"FILENAME": "Sample"}).sort("Sample")

        self.intermediate_results.add_df("condition_pivot", condition_mapping)

    # using a trick because polars>=1.31 does aggregate filling with 0s ! Instead of nan ! 
    def _pivot_df(self,
                  df: pl.DataFrame,
                  sample_col: str,
                  protein_col: str,
                  values_col: str,
                  aggregate_fn: str) -> pl.DataFrame:
         # 1) Map the requested agg name to a Polars expression
         fn_map = {
             "sum":    pl.col(values_col).sum(),
             "mean":   pl.col(values_col).mean(),
             "min":    pl.col(values_col).min(),
             "max":    pl.col(values_col).max(),
             "median": pl.col(values_col).median(),
         }
         if aggregate_fn not in fn_map:
             raise ValueError(f"Unsupported aggregate_fn '{aggregate_fn}'")

         # 2) Pre-aggregate so (protein, sample) is unique
         df_agg = (
             df
             .group_by([protein_col, sample_col], maintain_order=True)
             .agg(fn_map[aggregate_fn].alias(values_col))
         )

         # 3) Pivot WITHOUT a second aggregation step:
         #    missing groups → Polars nulls
         pivot_df = df_agg.pivot(
             index=protein_col,
             columns=sample_col,
             values=values_col,
         )

         # 4) Ensure those nulls become np.nan in your NumPy arrays
         pivot_df = pivot_df.fill_null(np.nan)

         # 5) Rename back to the original sample names
         return pivot_df.rename({c: c for c in df[sample_col].unique()})

    def _pivot_data(self) -> None:
        """
        Converts the dataset from long format (one row per protein per sample)
            to a wide format (one row per protein, with sample-specific intensity & q-value columns).
        """
        df = self.intermediate_results.dfs["filtered_final/RE"]

        required_cols = {"INDEX", "FILENAME", "SIGNAL", "QVALUE"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for pivoting: {missing_cols}")


        intensity_pivot = self._pivot_df(
                        df=df,
                        sample_col='FILENAME',
                        protein_col='INDEX',
                        values_col='SIGNAL',
                        aggregate_fn=self.pivot_signal_method,
                        )

        # Pivot Q values
        qvalue_pivot = self._pivot_df(
                                df=df,
                                sample_col='FILENAME',
                                protein_col='INDEX',
                                values_col='QVALUE',
                                aggregate_fn='mean'
                                )

        # Pivot PEP
        pep_pivot = None
        if "PEP" in df.columns :
            pep_pivot = self._pivot_df(
                                    df=df,
                                    sample_col='FILENAME',
                                    protein_col='INDEX',
                                    values_col='PEP',
                                    aggregate_fn='mean'
                                    )
        self.intermediate_results.set_columns_and_index(intensity_pivot)
        self.intermediate_results.add_df("raw_df", intensity_pivot)
        self.intermediate_results.add_df("qvalues", qvalue_pivot)
        self.intermediate_results.add_df("pep", pep_pivot)

    @log_time("Normalization")
    def _normalize(self) -> None:
        """
        Normalize intensity values and generate normalization diagnostic plots.

        Args:
            df (pl.DataFrame): Input dataframe.
            condition_pivot (pl.DataFrame): Mapping from sample to condition.

        Returns:
            pl.DataFrame: Normalized dataframe (Polars).
        """
        self._normalize_matrix()

    def _normalize_matrix(self) -> None:

        """
        Apply selected normalization(s) to intensity matrix extracted from a DataFrame.
        Supports chained normalization steps (e.g., log + loess).

        Args:
            df (pl.DataFrame): Input dataframe with intensity columns.
            condition_pivot (pl.DataFrame): Mapping of sample to condition.
            span (float): Smoothing span for loess.

        Returns:
            pl.DataFrame: DataFrame with normalized intensity values.
        """

        normalization_method = self.normalization.get("method")
        numeric_cols = self.intermediate_results.columns

        df = self.intermediate_results.dfs.get("raw_df")

        mat = df.select(numeric_cols).to_numpy()
        original_mat = mat.copy()
        postlog_mat = mat.copy()
        postlog_df = df.clone()
        models = None
        regression_scale_used = None
        regression_type_used = None

        if isinstance(normalization_method, str):
            normalization_method = [normalization_method]

        for method in normalization_method:
            if method == "log10":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log10(mat)
                postlog_mat = mat.copy()

            elif method == "log2":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log2(mat)
                postlog_mat = mat.copy()

            elif method == "median_equalization":
                global_median = np.nanmedian(mat)
                medians = np.nanmedian(mat, axis=0, keepdims=True)
                mat = mat * (global_median / medians)

            elif method == "quantile":
                qt = sklearn.preprocessing.QuantileTransformer(random_state=42)
                mat = qt.fit_transform(mat)

            elif method in {"global_linear", "global_loess", "local_linear", "local_loess"}:
                regression_scale_used = "global" if "global" in method else "local"
                regression_type_used = "loess" if "loess" in method else "linear"

                # build condition_labels in same column order as mat
                sample_names = numeric_cols
                cond_df = self.intermediate_results.dfs["condition_pivot"].to_pandas()
                cond_df.columns = [c.capitalize() for c in cond_df.columns]
                cond_map = cond_df.set_index("Sample")["Condition"]

                condition_labels = [cond_map[s] for s in sample_names]

                mat, models = regression_normalization(
                    mat,
                    scale=regression_scale_used,
                    regression_type=regression_type_used,
                    span=self.normalization.get("loess_span"),
                    condition_labels=condition_labels
                )

            elif method == "none":
                print("Skipping normalization (raw data)")

            else:
                raise ValueError(f"Invalid normalization method: {method}")

        df = df.with_columns(pl.DataFrame(mat, schema=numeric_cols))
        postlog_df = df.with_columns(pl.DataFrame(postlog_mat, schema=numeric_cols))

        self.intermediate_results.add_matrix("raw_mat", original_mat)
        self.intermediate_results.add_matrix("postlog", postlog_mat)
        self.intermediate_results.add_matrix("normalized", mat)
        self.intermediate_results.add_df("postlog", postlog_df)
        self.intermediate_results.add_df("normalized", df)
        self.intermediate_results.add_model("normalization", models)
        self.intermediate_results.add_metadata("normalization",
                                               "method",
                                               self.normalization.get("method"))
        self.intermediate_results.add_metadata("normalization",
                                               "regression_scale_used",
                                               regression_scale_used)
        self.intermediate_results.add_metadata("normalization",
                                               "regression_type_used",
                                               regression_type_used)
        self.intermediate_results.add_metadata("normalization",
                                               "span",
                                               self.normalization.get("loess_span"))


    @log_time("Imputation")
    def _impute(self) -> None:
        """
        Impute missivg values and generate diagnostic plots.

        Args:
            df (pl.DataFrame): Input dataframe.
            condition_pivot (pl.DataFrame): Mapping from sample to condition.

        Returns:
            pl.DataFrame: Normalized dataframe (Polars).
        """
        results = self._impute_matrix()

    def _impute_matrix(self) -> None:
        """Applies imputation to missing values."""

        numeric_cols = self.intermediate_results.columns
        df = self.intermediate_results.dfs["normalized"]
        not_imputed_mat = df.select(numeric_cols).to_numpy().copy()

        #imputer = get_imputer(**self.imputation)
        condition_map = self.intermediate_results.dfs["condition_pivot"].to_pandas()
        imputer = get_imputer(
            **self.imputation,
            condition_map=condition_map,
            sample_index=numeric_cols
        )

        imputed_mat = imputer.fit_transform(not_imputed_mat)
        df = df.with_columns(pl.DataFrame(imputed_mat,
                                          schema=numeric_cols))
        imputed_only = np.where(np.isnan(not_imputed_mat),
                                imputed_mat,
                                np.nan)

        self.intermediate_results.add_matrix("imputed_only", imputed_only)
        self.intermediate_results.add_matrix("imputed", imputed_mat)
        self.intermediate_results.add_df("imputed", df)
        self.intermediate_results.add_metadata("imputation",
                                               "method",
                                               self.imputation.get("method"))

