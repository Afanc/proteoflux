import numpy as np
import polars as pl
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import sklearn.metrics
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Optional

from proteoflux.workflow.imputer_factory import get_imputer
from proteoflux.dataset.intermediateresults import IntermediateResults
from proteoflux.evaluation.evaluation_utils import aggregate_metrics, prepare_long_df
from proteoflux.utils.random_utils import produce_balanced_mv
from proteoflux.utils.utils import logger, log_time
from proteoflux.export.plot_utils import (
    plot_histogram,
    plot_aggregated_violin,
    plot_bar_on_axis,
    plot_missing_corr_heatmap,
    plot_cluster_heatmap,
    get_color_palette,
    plot_regression_scatter
)

class ImputeEvaluator:
    """Class to evaluate and visualize missing value imputation."""

    def __init__(
        self,
        intermediate_results: IntermediateResults,
        imputation: dict = {}):
        """Initialize the evaluator with datasets and metadata.

        Args:
            imputation (str, optional): Name of the imputation method. Defaults to "".
        """
        self.results = intermediate_results
        columns = self.results.columns

        # Imputation method and model
        self.imputation = imputation

        # Column names for numeric data
        self.numeric_columns = columns[1:] if columns[0].lower() == "index" else columns

        # Convert condition mapping to pandas 
        condition_pivot = self.results.dfs["condition_pivot"]
        self.condition_map = condition_pivot.to_pandas().rename(
            columns={"Sample": "Sample", "CONDITION": "Condition"}
        )


        # Derived DataFrames for easy plotting

        self.df_original = pd.DataFrame(self.results.matrices["normalized"],
            columns=self.numeric_columns)

        self.df_imputed = pd.DataFrame(self.results.matrices["imputed"],
            columns=self.numeric_columns)

        self.df_imputed_only = pd.DataFrame(self.results.matrices["imputed_only"],
            columns=self.numeric_columns)

        self.mvi_df = self._get_mvi_count_per_sample_df(self.df_imputed_only)

        # --- Color maps
        # Compute condition-to-color mapping for all conditions.
        unique_conditions = self.condition_map["Condition"].unique()
        condition_colors = get_color_palette("tab20", len(unique_conditions))
        self.condition_color_map = dict(zip(unique_conditions, condition_colors))

        # Compute a mapping from Sample to Condition
        self.sample_to_condition = self.condition_map.set_index("Sample")["Condition"].to_dict()

        # Build the sample palette using the mappings
        self.sample_palette = {
            sample: self.condition_color_map[self.sample_to_condition[sample]]
            for sample in self.mvi_df["Sample"]
        }

        # Build a Series for heatmaps: for each numeric column (sample), map its condition to a color.
        self.condition_map_df = self.condition_map.set_index("Sample").loc[
            self.numeric_columns, "Condition"
        ].map(self.condition_color_map)
        # --- 

    @log_time("Imputation - plot")
    def plot_all(self, filename: Union[str, Path] = "imputation_plots.pdf"):
        """Generate all plots and save to a PDF file.

        Args:
            filename (Union[str, Path]): Filename for the output PDF.
        """
        with PdfPages(filename) as pdf:
            self.pdf = pdf
            # before imputation
            self._plot_mvi_per_condition()
            self._plot_mvi_per_sample()
            self._plot_mv_heatmap()
            self._plot_mv_correlation()

            # after imputation
            self._plot_distribution_histograms()
            self._plot_violin_metrics_by_condition()
            self._plot_imputed_intensity_violin_by_condition()

            # evaluation of imputation model
            results_df, values_df = self._evaluate_imputation()
            self._plot_regression(values_df["True Values"],
                                  values_df["Imputed Values"],
                                  results_df)

    def _plot_mvi_per_condition(self):
        """
        Plot missing value counts with global totals and per-condition bars.

        Args:
            pdf (PdfPages): PDF file to save plots.
        """
        mvi_df = self._get_mvi_count_per_condition_df(self.df_imputed_only)

        total_missing = mvi_df.loc[mvi_df["Condition"] == "Total (Imputed)", "MVI"].values[0]
        total_nonmissing = mvi_df.loc[mvi_df["Condition"] == "Total (Non-Imputed)", "MVI"].values[0]
        ratio = total_missing / (total_missing + total_nonmissing)

        ## Color mapping -> DICT
        condition_color_map = self.condition_color_map.copy()
        condition_color_map["Total (Non-Imputed)"] = '#999999'
        condition_color_map["Total (Imputed)"] = '#17becf'

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_bar_on_axis(
            ax=ax,
            data=mvi_df,
            x="Condition",
            y="MVI",
            title=f"Number of Missing Values Imputed (Imputation Ratio = {ratio:.2%})",
            xlabel="Condition",
            ylabel="Count",
            log_scale=True,
            xtick_rotation=45,
            annotate_values=True,
            palette=condition_color_map,
        )
        plt.subplots_adjust(bottom=0.25)
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_mvi_per_sample(self):
        """
        Plot number of imputed values per sample.
        """

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_bar_on_axis(
            ax=ax,
            data=self.mvi_df,
            x="Sample",
            y="MVI",
            title="Number of Missing Values Imputed per Sample",
            xlabel="Sample",
            ylabel="Count",
            xtick_rotation=45,
            palette=self.sample_palette,
        )
        plt.subplots_adjust(bottom=0.35)
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_mv_heatmap(self):
        """Heatmap of missing values with hierarchical clustering by samples.
        """
        mv_matrix = self.df_original[self.numeric_columns].isna().astype(int)

        g = plot_cluster_heatmap(
                data=mv_matrix,
                col_colors=self.condition_map_df,
                title="Heatmap of Missing Values, at Protein Level",
                legend_mapping=self.condition_color_map
        )

        self.pdf.savefig(g.fig)
        plt.close(g.fig)

    def _plot_mv_correlation(self):
        """Correlation heatmap of missing values.
        """
        plot_missing_corr_heatmap(
            data=self.df_original[self.numeric_columns],
            title="Missing Values Correlation Heatmap"
        )
        self.pdf.savefig()
        plt.close()

    def _plot_distribution_histograms(self):
        """
        Plot histograms showing intensity distributions for original and imputed values.
        """
        original_long = prepare_long_df(df=self.df_original,
                                        label_col="Imputation",
                                        label="Original",
                                        condition_mapping=None)
        imputed_long = prepare_long_df(self.df_imputed_only,
                                       label_col="Imputation",
                                       label="Imputed",
                                       condition_mapping=None)

        combined_long = pd.concat([original_long, imputed_long], ignore_index=True).dropna()

        plot_histogram(
            data=combined_long,
            pdf=self.pdf,
            stat="count",
            title="Distribution of Intensities of Imputed MV",
            group_col="Imputation",
            labels=["Original", "Imputed"]
        )

    def _plot_violin_metrics_by_condition(self):
        """
        Plot violin plots of CV and MAD grouped by condition.
        """
        agg_metrics = aggregate_metrics(
            before_df=self.df_original,
            after_df=self.df_imputed_only,
            condition_mapping=self.condition_map,
            metrics=["log_CV", "CV", "geometric_CV", "RMAD"],
            label_col="Imputation",
            before_label="Original",
            after_label="Imputed",
            condition_key="Condition",
            sample_key="Sample"
        )

        metrics = {}

        metrics['CV'] = agg_metrics["CV"]
        metrics['log_CV'] = agg_metrics["log_CV"]
        metrics['geometric_CV'] = agg_metrics["geometric_CV"]
        metrics['RMAD'] = agg_metrics["RMAD"]

        for k,v in metrics.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_aggregated_violin(
                ax=ax,
                data=v,
                hue="Imputation",
                title=f"{k} Per Condition (Original vs Imputed Values)",
                palette={"Original": "blue", "Imputed": "red"},
                ylabel=k,
                density_norm="width"
                #density_norm="area"
            )
            self.pdf.savefig(fig)
            plt.close(fig)

    def _plot_imputed_intensity_violin_by_condition(self):
        """
        Plot violin plots of intensity distributions (original vs imputed) per condition,
        using the shared violin plotting utility.
        """
        # Convert to long format
        df_original_long = prepare_long_df(df=self.df_original,
                                           label_col="Imputation",
                                           label="Original",
                                           condition_mapping=None,
                                           sample_col="Sample")
        df_imputed_long = prepare_long_df(df=self.df_imputed_only,
                                          label_col="Imputation",
                                          label="Imputed",
                                          condition_mapping=None)

        # Combine and filter out NaNs
        combined = pd.concat([df_original_long, df_imputed_long], ignore_index=True)
        combined = combined.dropna(subset=["Intensity"])

        # Merge with condition info
        combined = combined.merge(self.condition_map, on="Sample", how="left")

        # Rename and restructure to match plot_aggregated_violin expectations
        violin_df = pd.DataFrame({
            "Condition": combined["Condition"],
            "Value": combined["Intensity"],
            "Imputation": combined["Imputation"]
        })

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_aggregated_violin(
            ax=ax,
            data=violin_df,
            hue="Imputation",
            title="Intensity Distribution by Condition (Original vs Imputed)",
            palette={"Original": "blue", "Imputed": "red"},
            ylabel="Log10 Intensity",
            density_norm="area"
        )
        self.pdf.savefig(fig)
        plt.close(fig)

        # Rename and restructure to match plot_aggregated_violin expectations
        violin_df = pd.DataFrame({
            "Condition": combined["Sample"],
            "Value": combined["Intensity"],
            "Imputation": combined["Imputation"]
        })

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_aggregated_violin(
            ax=ax,
            data=violin_df,
            hue="Imputation",
            title="Intensity Distribution by Sample (Original vs Imputed)",
            palette={"Original": "blue", "Imputed": "red"},
            ylabel="Log10 Intensity",
            density_norm="area"
        )
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(left=0.15, bottom=0.45)
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_cv_missingness_example(self, X_original, mv_mask_original, mv_mask_cv, title):
        """
        Plot a heatmap distinguishing original missing values from artificially injected missing values
        for a single CV fold using hierarchical clustering by samples.

        Args:
            X_original (np.ndarray): The complete data matrix (imputed).
            mv_mask_original (np.ndarray): Boolean mask of original missing values.
            mv_mask_cv (np.ndarray): Boolean mask of additionally injected missing values for CV.
        """
        # Prepare combined mask:
        # 0: present, 1: artificially injected missing
        combined_mask = np.zeros(X_original.shape, dtype=int)
        combined_mask[mv_mask_cv] = 1

        combined_df = pd.DataFrame(combined_mask, columns=self.numeric_columns)
        combined_df = combined_df.sort_values(by=self.numeric_columns[0], ascending=True)

        g = plot_cluster_heatmap(
                data=combined_df,
                col_colors=self.condition_map_df,
                title=title,
                legend_mapping=self.condition_color_map,
                binary_labels=["Present", "Injected Missing"]

        )

        self.pdf.savefig(g.fig)
        plt.close(g.fig)

    def _get_mvi_count_per_sample_df(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute number of imputed values per sample.

        Args:
            matrix (pd.DataFrame): DataFrame containing only imputed values (NaNs represent non-imputed values).

        Returns:
            pd.DataFrame: DataFrame with 'Sample', 'MVI', and 'Condition'.
        """
        imputed_mask = ~matrix.isna()
        imputed_per_sample = imputed_mask.sum(axis=0)

        mv_df = pd.DataFrame({
            "Sample": matrix.columns,
            "MVI": imputed_per_sample.values
        })

        return mv_df.merge(self.condition_map, on="Sample", how="left")

    def _get_mvi_count_per_condition_df(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute imputed values per condition, with summary rows for total non-imputed and imputed counts.

        Args:
            matrix (pd.DataFrame): DataFrame containing only imputed values (NaNs represent non-imputed values).

        Returns:
            pd.DataFrame: DataFrame with columns 'Condition' and 'MVI', including totals.
        """
        imputed_mask = ~matrix.isna()
        imputed_per_sample = imputed_mask.sum(axis=0)

        total_elements = matrix.size
        total_imputed = imputed_mask.sum().sum()
        total_non_imputed = total_elements - total_imputed

        mvi_df = self.mvi_df.copy()

        per_condition = mvi_df.groupby("Condition")["MVI"].sum().reset_index()

        summary_rows = pd.DataFrame([
            {"Condition": "Total (Non-Imputed)", "MVI": total_non_imputed},
            {"Condition": "Total (Imputed)", "MVI": total_imputed},
        ])

        return pd.concat([summary_rows, per_condition], ignore_index=True)


    def _evaluate_imputation(self, n_cv: int = 10, p_miss: float = 0.1):
        """
        Run cross-validation to evaluate the imputation method.

        Args:
            n_cv (int, optional): Number of cross-validation folds. Defaults to 10.
            p_miss (float, optional): Proportion of missing values to inject. Defaults to 0.1.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - DataFrame with CV metrics ('fold', 'r2', 'rmae').
                - DataFrame with all true and imputed values across folds.

        #TODO more info, which model, percentage of NA and of MNAR
        """
        results = []
        X_complete = self.df_imputed[self.numeric_columns].to_numpy()

        all_true_vals = []
        all_imputed_vals = []

        for fold in tqdm(range(n_cv), leave=False):
            # Generate MV data
            mv_data = produce_balanced_mv(X_complete,
                                          p_miss_total=0.15,
                                          p_mnar=0.9,
                                          quantile=0.1,
                                          q_lod=0.05)

            mask = mv_data['mask'].numpy().astype(bool)
            X_incomp = mv_data['X_incomp'].numpy()

            # Impute again using the same imputation method
            #imputer = get_imputer(**self.imputation)
            condition_map = self.results.dfs["condition_pivot"].to_pandas()
            sample_index = list(self.df_imputed[self.numeric_columns].columns)
            imputer = get_imputer(
                **self.imputation,
                condition_map=condition_map,
                sample_index=sample_index,
            )


            X_reimputed = imputer.fit_transform(X_incomp)

            # Evaluate performance on artificially injected MV only
            true_vals = X_complete[mask]
            imputed_vals = X_reimputed[mask]

            r2 = sklearn.metrics.r2_score(true_vals, imputed_vals)
            mae = sklearn.metrics.median_absolute_error(true_vals, imputed_vals)
            rmae = mae / np.median(true_vals)

            results.append({
                "fold": fold,
                "r2": r2,
                "rmae": rmae
            })

            all_true_vals.extend(true_vals)
            all_imputed_vals.extend(imputed_vals)

            #TODO plotting a single fold
            if fold < 1:
                perc_of_na = np.isnan(X_incomp).sum() / X_incomp.size * 100
                mv_mask_original = np.isnan(self.df_original)
                self._plot_cv_missingness_example(
                    X_complete,
                    mv_mask_original,
                    mask,
                    f"Missing Values (CV Fold Example), {perc_of_na:.2f} % MV"
                )
                plot_missing_corr_heatmap(
                    data=pd.DataFrame(X_incomp, columns=self.numeric_columns),
                )
                self.pdf.savefig()
                plt.close()

        results_df = pd.DataFrame(results)

        values_df = pd.DataFrame({
            "True Values": all_true_vals,
            "Imputed Values": all_imputed_vals
        })

        return results_df, values_df

    def _plot_regression(self, true_vals: np.ndarray, imputed_vals: np.ndarray,
                         results_df: pd.DataFrame):
        """
        Plot scatter of imputed vs. true values with density coloring, and annotate with metrics.

        Args:
            true_vals (np.ndarray): True values.
            imputed_vals (np.ndarray): Imputed values.
            results_df (pd.DataFrame): DataFrame containing cross-validation results (with 'r2' and 'rmae').
        """
        fig = plot_regression_scatter(true_vals, imputed_vals, results_df)
        self.pdf.savefig(fig)
        plt.close(fig)
