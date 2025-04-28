from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skmisc.loess import loess

from proteoflux.export.plot_utils import plot_violin_on_axis, plot_aggregated_violin, plot_histogram, plot_MA
from proteoflux.evaluation.evaluation_utils import compute_metrics, aggregate_metrics, prepare_long_df
from proteoflux.dataset.intermediateresults import IntermediateResults
from proteoflux.utils.utils import logger, log_time

class NormalizerPlotter:
    def __init__(
        self,
        intermediate_results: IntermediateResults,
    ):
        self.results = intermediate_results

        # Direct conversion to pandas DataFrames using stored matrices and columns
        self.original_df = pd.DataFrame(
            self.results.matrices["raw_mat"], columns=self.results.columns
        )

        # Handle if postlog is None (meaning no log applied)
        postlog_mat = self.results.matrices.get("postlog", self.results.matrices["raw_mat"])
        self.postlog_df = pd.DataFrame(
            postlog_mat, columns=self.results.columns
        )

        self.normalized_df = pd.DataFrame(
            self.results.matrices["normalized"], columns=self.results.columns
        )

        # Convert conditions to pandas once here
        condition_pivot = self.results.dfs["condition_pivot"]
        self.condition_map = condition_pivot.to_pandas().rename(
            columns={"Sample": "Sample", "CONDITION": "Condition"}
        )

        self.normalization_method = self.results.metadata.get("normalization").get("method")
        self.models = self.results.models["normalization"]
        self.scale = self.results.metadata["normalization"].get("scale", "global")
        self.regression_type = self.results.metadata["normalization"].get("regression_type", "loess")

    @log_time("Normalization - plot")
    def plot_all(self, filename: Union[str, Path] = "normalization_plots.pdf") -> None:
        with PdfPages(filename) as pdf:
            self.pdf = pdf
            self._plot_filtering_histograms()
            self._plot_histogram_log()
            self._plot_violin_intensity_by_condition()
            self._plot_violin_intensity_by_sample()
            self._plot_violin_metrics_by_condition()
            self._plot_MA_plots()

    #TODO use plot utils, and plots next to each other. also only if cutoff < kept. also run_evidence_count. and smart with ranges on x, maybe log y not useful
    def _plot_filtering_histograms(self):
        thresholds = self.results.metadata.get("filtering", {})
        import seaborn as sns

        for stat_key, cutoff in thresholds.items():
            arr_key = stat_key.lower() + "_array"
            values = self.results.arrays.get(arr_key)

            if values is not None:
                fig, ax = plt.subplots()
                sns.histplot(values, ax=ax, bins=60)
                ax.axvline(cutoff, color='red', linestyle='--')
                ax.set_title(f"{stat_key} distribution (cutoff: {cutoff:.3g}, kept: {(values <= cutoff).sum()} / {len(values)})")
                ax.set_yscale("log")
                self.pdf.savefig(fig)
                plt.close(fig)

    def _plot_histogram_log(self):
        """
        Plot histogram plots of intensity distributions before and after log transformation.
        Combines data from both pre-log and post-log versions.
        """

        original_long = prepare_long_df(df=self.original_df,
                                        label_col="Normalization",
                                        label="Before",
                                        condition_mapping=None)
        postlog_long = prepare_long_df(df=self.postlog_df,
                                       label_col="Normalization",
                                       label="After",
                                       condition_mapping=None)
        combined_df = pd.concat([original_long,
                                 postlog_long], ignore_index=True).dropna()

        plot_histogram(
            data=combined_df,
            pdf=self.pdf,
            stat="probability",
            title="Distribution of Intensities")

    def _plot_violin_intensity_by_condition(self):
        """
        Plot violin plots of intensity distributions grouped by condition.
        Combines data from both original (post-log) and normalized versions.
        """

        # Prepare long-format DataFrames.
        original_df_long = prepare_long_df(df=self.postlog_df,
                                           label_col="Normalization",
                                           label="Before",
                                           condition_mapping=self.condition_map)
        normalized_df_long = prepare_long_df(df=self.normalized_df,
                                            label_col="Normalization",
                                            label="After",
                                            condition_mapping=self.condition_map)

        # Plot violins per condition
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        plot_violin_on_axis(
            ax=axes[0],
            data=original_df_long,
            x="Condition",
            y="Intensity",
            hue="Normalization",
            title="Before Normalization",
            xlabel="Condition",
            ylabel="Intensity"
        )

        plot_violin_on_axis(
            ax=axes[1],
            data=normalized_df_long,
            x="Condition",
            y="Intensity",
            hue="Normalization",
            title="After Normalization",
            xlabel="Condition",
            ylabel="Intensity"
        )

        plt.suptitle("Distribution by Condition (Before vs After Normalization), data log transformed")

        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_violin_metrics_by_condition(self):
        """
        Plot violin plots of CV and MAD grouped by condition.
        Combines data from both original (post-log) and normalized versions.
        """

        # Prepare metrics
        agg_metrics = aggregate_metrics(
            before_df=self.postlog_df,
            after_df=self.normalized_df,
            condition_mapping=self.condition_map,
            metrics=["CV", "RMAD", "geometric_CV"],
            label_col="Normalization",
            before_label="Before",
            after_label="After",
            condition_key="Condition",
            sample_key="Sample"
        )

        metrics = {}

        metrics['geometric_CV'] = agg_metrics["geometric_CV"]
        metrics['RMAD'] = agg_metrics["RMAD"]
        metrics['CV'] = agg_metrics["CV"]

        for k,v in metrics.items():
            #TODO remove CV altogether eventually
            title=f"{k} Per Condition (Before vs After Normalization)"\
                f" using {self.normalization_method}"
            if k == "CV":
                title += " | WARNING - don't trust CV in log scale ! "
            #--
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_aggregated_violin(
                ax=ax,
                data=v,
                hue="Normalization",
                title=title,
                palette={"Before": "blue", "After": "red"},
                ylabel=k,
                density_norm="width",
                #density_norm="area"
                #log_scale=log_scale,
            )
            self.pdf.savefig(fig)
            plt.close(fig)


    def _plot_violin_intensity_by_sample(self):
        """
        Plot violin plots of intensity distributions grouped by sample, side by side or stacked.

        Parameters:
          original_df (pd.DataFrame): DataFrame with the original (pre-normalized) data.
          normalized_df (pd.DataFrame): DataFrame with the normalized data.
          pdf (PdfPages): PdfPages object for saving the plots.
        """
        # Melt dataframes for easier plotting
        original_df_long = prepare_long_df(df=self.postlog_df,
                                           label_col="Normalization",
                                           label="Before",
                                           condition_mapping=None)
        normalized_df_long = prepare_long_df(df=self.normalized_df,
                                            label_col="Normalization",
                                            label="After",
                                            condition_mapping=None)

        # Create two subplots (side-by-side)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Use the helper function to plot the "Before" data
        plot_violin_on_axis(
            ax=axes[0],
            data=original_df_long,
            x="Sample",
            y="Intensity",
            hue="Normalization",
            title="Before Normalization",
            xlabel="Sample",
            ylabel="Intensity",
            xtick_rotation=45,
            xtick_fontsize=7,
        )

        # And for the "After" data
        plot_violin_on_axis(
            ax=axes[1],
            data=normalized_df_long,
            x="Sample",
            y="Intensity",
            hue="Normalization",
            title="After Normalization",
            xlabel="Sample",
            ylabel="Intensity",
            xtick_rotation=45,
            xtick_fontsize=7,
        )

        plt.suptitle("Distribution by Sample (Before vs After Normalization), data log transformed")
        plt.subplots_adjust(bottom=0.35)
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_MA_plots(
        self,
        scale: str = "global",
        regression_type: Optional[str] = "loess",
        span: float = 0.3,
        before_label: str = "Before",
        after_label: str = "After",
        before_color: str = "blue",
        after_color: str = "red",
        show_legend: bool = True,
    ) -> None:
        """
        Generate MA plots comparing values before vs after normalization.
        Uses class-stored matrices and metadata.
        """

        original_mat = np.clip(self.postlog_df.to_numpy(), 1e-10, None)
        normalized_mat = np.clip(self.normalized_df.to_numpy(), 1e-10, None)
        columns = self.original_df.columns

        scale = self.scale
        regression_type = self.regression_type

        # Choose reference (global or local)
        if scale == "global":
            reference_values = np.nanmean(original_mat, axis=1)
        else:
            reference_values = original_mat[:, 0]
            reference_values = np.where(np.isnan(reference_values), np.nanmean(original_mat, axis=1), reference_values)

        for sample_idx in range(original_mat.shape[1]):
            fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(15, 6))

            M_before = np.nanmean(original_mat, axis=1)
            A_before = original_mat[:, sample_idx] - reference_values

            M_after = np.nanmean(normalized_mat, axis=1)
            A_after = normalized_mat[:, sample_idx] - reference_values

            model = self.models[sample_idx] if self.models else None
            sample_name = columns[sample_idx]

            plot_MA(
                ax=ax_before,
                M=M_before,
                A=A_before,
                label=before_label,
                reg_label="Before",
                color=before_color,
                model=model,
                span=span,
                regression_type=regression_type,
                refit=False,
                random_state=42,
                #title=f"MA Plot ({before_label}) - {sample_name}",
                #title=f"{before_label} - {sample_name}",
                xlabel="M (Mean Intensity)",
                ylabel="A (Log-ratio or Difference)",
                show_legend=show_legend,
            )

            reg_label = after_label if model is not None else "Indicative"
            plot_MA(
                ax=ax_after,
                M=M_after,
                A=A_after,
                label=after_label,
                reg_label=reg_label,
                color=after_color,
                model=model,
                span=span,
                regression_type=regression_type,
                refit=True,
                random_state=42,
                #title=f"MA Plot ({after_label}) - {sample_name}",
                #title=f"{after_label} - {sample_name}",
                xlabel="M (Mean Intensity)",
                ylabel="A (Log-ratio or Difference)",
                show_legend=show_legend,
            )

            suptitle=f"MA plots for {sample_name} using {self.normalization_method}"
            plt.suptitle(suptitle, fontsize=14, y=0.955)

            self.pdf.savefig(fig)
            plt.close(fig)
