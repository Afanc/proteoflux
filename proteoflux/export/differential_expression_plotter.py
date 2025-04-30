import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from anndata import AnnData
from typing import Optional, List, Union, Dict
from proteoflux.utils.utils import logger, log_time
from proteoflux.export.plot_utils import plot_cluster_heatmap

class DifferentialExpressionPlotter:
    def __init__(
        self,
        adata: AnnData,
        analysis_config: Dict,
        protein_label_key: str = "FASTA_HEADERS",
    ):
        self.analysis_config = analysis_config
        self.export_config = analysis_config.get("exports")
        self.adata = adata
        self.protein_labels = adata.var.get(protein_label_key, adata.var_names)
        self.protein_names = adata.var["GENE_NAMES"] #TODO option ? 
        self.contrast_names = adata.uns.get("contrast_names", [f"C{i}" for i in range(adata.varm['log2fc'].shape[1])])

        # Cache stats for fast access
        self.log2fc = adata.varm["log2fc"]
        self.p = pd.DataFrame(adata.varm["p"],
                              columns=self.contrast_names,
                              index=adata.var.index)
        self.q = pd.DataFrame(adata.varm["q"],
                              columns=self.contrast_names,
                              index=adata.var.index)
        self.p_ebayes = pd.DataFrame(adata.varm["p_ebayes"],
                                     columns=self.contrast_names,
                                     index=adata.var.index)
        self.q_ebayes = pd.DataFrame(adata.varm["q_ebayes"],
                                     columns=self.contrast_names,
                                     index=adata.var.index)
        self.res_var = adata.uns.get("residual_variance", None)
        self.residuals = adata.uns.get("residuals", None)
        self.missingness = adata.uns.get("missingness", {})
        self.fasta_headers = adata.var['FASTA_HEADERS'].to_numpy()

        # Plotting options
        self.umap_n_neighbors = self.export_config.get("umap_n_neighbors")

    @log_time("Differential Expression - plotting")
    def plot_all(self):
        with PdfPages(self.export_config.get("path_plot")) as pdf:
            self.pdf = pdf
            self._plot_log2fc_distributions()
            self._plot_residual_variance_hist()
            self._plot_residual_heatmap()
            self._plot_stat_distribution("p", log_y=False)
            self._plot_stat_distribution("q", log_y=False)
            self._plot_ebayes_comparison()
            self._plot_volcano_plots()
            self._plot_cluster_heatmap()
            self._plot_pca()
            self._plot_umap()
            if "EC_high_vs_EC_low" in self.contrast_names:
                self._plot_spikein_separation("EC_high_vs_EC_low")


    def _plot_log2fc_distributions(self):
        fig, axs = plt.subplots(1, self.log2fc.shape[1], figsize=(4 * self.log2fc.shape[1], 4), sharey=True)

        for i, name in enumerate(self.contrast_names):
            ax = axs[i] if self.log2fc.shape[1] > 1 else axs
            ax.hist(self.log2fc[:, i], bins=40, alpha=0.8)
            ax.set_title(f"log2FC: {name}")
            ax.set_xlabel("log2FC")
            ax.set_ylabel("Count")

        fig.tight_layout()
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_residual_variance_hist(self):
        if self.res_var is None:
            print("No residual variance found — skipping.")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(self.res_var, bins=100, color="skyblue", edgecolor="black")
        ax.set_title("Residual Variance Distribution")
        ax.set_xlabel("Residual Variance")
        ax.set_ylabel("Protein Count")
        fig.tight_layout()
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_residual_heatmap(self):
        if self.residuals is None:
            print("No residuals found — skipping.")
            return

        if self.residuals.shape[0] == self.adata.n_vars:
            residuals = self.residuals.T
        else:
            residuals = self.residuals

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(residuals, cmap="vlag", xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title("Residual Heatmap")
        ax.set_xlabel("Proteins")
        ax.set_ylabel("Samples")
        fig.tight_layout()
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_stat_distribution(self, stat_key, log_y=False):
        contrasts = self.contrast_names
        n_contrasts = len(contrasts)

        raw = self.p if stat_key == "p" else self.q
        post = self.p_ebayes if stat_key == "p" else self.q_ebayes

        fig, axes = plt.subplots(1, n_contrasts, figsize=(4 * n_contrasts, 4))

        if n_contrasts == 1:
            axes = [axes]

        bins = np.linspace(0, 1, 50)
        colors = ["dodgerblue", "darkorange"]

        for i, contrast in enumerate(contrasts):
            ax = axes[i]
            sns.histplot(raw[contrast], bins=bins, color=colors[0], kde=True, alpha=0.5, ax=ax, stat="count", edgecolor=None)
            sns.histplot(post[contrast], bins=bins, color=colors[1], kde=True, alpha=0.5, ax=ax, stat="count", edgecolor=None)
            ax.set_xlim(0, 1)
            ax.set_title(f"{stat_key}-values: {contrast}")
            ax.legend([
                f"Raw {stat_key}",
                f"Empirical Bayes {stat_key}"
            ])
            if log_y:
                ax.set_yscale("log")
            ax.set_ylabel("count")

        fig.tight_layout()

        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_volcano_plots(self):
        sign_threshold = self.analysis_config.get("sign_threshold", 0.05)
        volcano_top_annotated = self.analysis_config.get("exports").get("volcano_top_annotated", 10)

        for i, name in enumerate(self.contrast_names):
            logfc = self.log2fc[:, i]
            pvals_raw = self.p.iloc[:, i]
            pvals_bayes = self.p_ebayes.iloc[:, i]
            qvals_raw = self.q.iloc[:, i]
            qvals_bayes = self.q_ebayes.iloc[:, i]

            group1, group2 = name.split("_vs_")

            # Map missingness from DataFrame
            miss_df = self.missingness
            missing_a = miss_df[group1].values >= 1.0
            missing_b = miss_df[group2].values >= 1.0
            both_missing = missing_a & missing_b

            color = np.full(logfc.shape[0], "gray", dtype=object)
            color[missing_a & ~missing_b] = "blue"
            color[missing_b & ~missing_a] = "green"

            def plot_panel(ax, pvals, qvals, title):
                mask = ~both_missing
                masked_missing_a = missing_a[mask]
                masked_missing_b = missing_b[mask]
                triangle_mask = masked_missing_a ^ masked_missing_b
                circle_mask = ~triangle_mask

                base_color = color[mask]
                base_logfc = logfc[mask]
                base_p = pvals.iloc[mask]

                ax.scatter(base_logfc[circle_mask], -np.log10(base_p.iloc[circle_mask]), c=base_color[circle_mask], alpha=0.7, s=10)
                ax.scatter(base_logfc[triangle_mask], -np.log10(base_p.iloc[triangle_mask]), c=base_color[triangle_mask], alpha=0.7, s=10, marker="^")

                ax.axhline(-np.log10(sign_threshold), color="black", linestyle="--", linewidth=0.8)
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

                for direction in ["up", "down"]:
                    sig_mask = (logfc > 0) if direction == "up" else (logfc < 0)
                    sig_mask &= (qvals < sign_threshold) & ~both_missing

                    top = np.argsort(qvals[sig_mask])[:volcano_top_annotated]
                    selected = np.where(sig_mask)[0][top]

                    for j in selected:
                        ax.text(
                            logfc[j],
                            -np.log10(pvals.iloc[j]),
                            self.protein_names.iloc[j],
                            fontsize=6,
                            ha="right" if logfc[j] > 0 else "left",
                            va="bottom",
                        )

                ax.set_xlabel("log2FC")
                ax.set_ylabel("-log10(q)")
                ax.set_title(title)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            plot_panel(ax, qvals_bayes, qvals_bayes, f"{name} (eBayes)")
            ax.legend(
                handles = [
                    plt.Line2D([0], [0], color="gray", marker="o", linestyle="None", label="Observed in both"),
                    plt.Line2D([0], [0], color="blue", marker="^", linestyle="None", label=f"missing from {group1}"),
                    plt.Line2D([0], [0], color="green", marker="^", linestyle="None", label=f"missing from {group2}"),
                ],
                title="Missingness",
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.)

            fig.tight_layout()
            self.pdf.savefig(fig)
            plt.close(fig)

    def _plot_ebayes_comparison(self):
        if "p_ebayes" not in self.adata.varm:
            print("No eBayes stats found — skipping.")
            return

        p_raw = self.p
        q_raw = self.q
        p_moderated = self.p_ebayes
        q_moderated = self.q_ebayes

        for i, name in enumerate(self.contrast_names):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            axs[0].scatter(-np.log10(p_raw.iloc[:, i]), -np.log10(p_moderated.iloc[:, i]), alpha=0.5, s=5)
            axs[0].plot([0, np.max(-np.log10(p_raw))], [0, np.max(-np.log10(p_raw))], "k--")
            #axs[0].plot([0, 20], [0, 20], "k--")
            axs[0].set_xlabel("–log10 p (raw)")
            axs[0].set_ylabel("–log10 p (eBayes)")
            axs[0].set_title(f"P-value shrinkage: {name}")

            axs[1].scatter(q_raw.iloc[:, i], q_moderated.iloc[:, i], alpha=0.5, s=5)
            axs[1].plot([0, 1], [0, 1], "k--")
            axs[1].set_xlabel("q (raw)")
            axs[1].set_ylabel("q (eBayes)")
            axs[1].set_title(f"FDR comparison: {name}")

            fig.tight_layout()
            self.pdf.savefig(fig)
            plt.close(fig)

    def _plot_cluster_heatmap(self):
        """
        Plot hierarchical clustering heatmap for samples based on protein expression.
        """
        # Get expression matrix (proteins × samples)
        expr_df = pd.DataFrame(
            self.adata.X.T,
            index=self.adata.var_names,
            columns=self.adata.obs_names
        )

        # Build sample condition mapping
        if "CONDITION" in self.adata.obs.columns:
            run_conditions = self.adata.obs["CONDITION"]
            unique_conditions = sorted(run_conditions.unique())
            condition_map = {cond: color for cond, color in zip(unique_conditions, sns.color_palette("tab10", n_colors=len(unique_conditions)))}
            col_colors = run_conditions.map(condition_map)
        else:
            col_colors = None

        # Perform clustering
        g = sns.clustermap(
            expr_df,
            cmap="vlag",
            row_cluster=True,
            col_cluster=True,
            col_colors=col_colors,
            xticklabels=True,
            yticklabels=False,
            method="ward",
            metric="euclidean",
            figsize=(10, 10)
        )

        g.fig.suptitle("Hierarchical Clustering of Samples by Expression", fontsize=16, y=0.955)
        g.ax_cbar.set_title("Expression", fontsize=9)

        # Add condition legend
        if col_colors is not None:
            legend_patches = [
                mpatches.Patch(color=condition_map[c], label=c) for c in unique_conditions
            ]
            g.ax_heatmap.legend(
                handles=legend_patches,
                title="Conditions",
                loc="upper right",
                fontsize=8,
                bbox_to_anchor=(1.20, 1.0)
            )

        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

        g.ax_cbar.set_position((0.10, 0.735, 0.03, 0.17))

        self.pdf.savefig(g.fig)
        plt.close(g.fig)

    def _plot_pca(self, color_key="CONDITION"):
        """
        Plot PCA of samples colored by condition.
        """
        if "X_pca" not in self.adata.obsm:
            sc.tl.pca(self.adata)

        pc_df = pd.DataFrame(
            self.adata.obsm["X_pca"][:, :2],
            columns=["PC1", "PC2"],
            index=self.adata.obs_names
        )
        pc_df[color_key] = self.adata.obs[color_key].values

        var_ratio = self.adata.uns["pca"]["variance_ratio"]
        pc1_var = var_ratio[0] * 100
        pc2_var = var_ratio[1] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=pc_df,
            x="PC1",
            y="PC2",
            hue=color_key,
            palette="tab10",
            edgecolor="black",
            s=50,
            alpha=0.8,
            ax=ax
        )

        # Annotate points
        palette = dict(zip(pc_df[color_key].unique(), sns.color_palette("tab10")))
        x_median = pc_df["PC1"].median()
        y_median = pc_df["PC2"].median()

        for sample, (x, y) in pc_df[["PC1", "PC2"]].iterrows():
            color = palette[pc_df.loc[sample, color_key]]

            ha = "left" if x <= x_median else "right"
            # annotate with a 5-point horizontal offset
            ax.annotate(
                sample,
                xy=(x, y),
                xytext=(5 if ha=="left" else -5, 0),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=7,
                color=color
            )

        ax.set_title("PCA")
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
        ax.legend(title=color_key, frameon=True,
                  loc="upper left",
                  bbox_to_anchor=(1.05, 1),
                  borderaxespad=0.)
        fig.tight_layout()

        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_umap(self, color_key="CONDITION"):
        """
        Plot UMAP of samples colored by condition.
        """
        if "X_umap" not in self.adata.obsm:
            if "neighbors" not in self.adata.uns:
                sc.pp.neighbors(self.adata,
                                n_neighbors=self.umap_n_neighbors)
            sc.tl.umap(self.adata)

        umap_df = pd.DataFrame(
            self.adata.obsm["X_umap"],
            columns=["UMAP1", "UMAP2"],
            index=self.adata.obs_names
        )
        umap_df[color_key] = self.adata.obs[color_key].values

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=umap_df,
            x="UMAP1",
            y="UMAP2",
            hue=color_key,
            palette="tab10",
            edgecolor="black",
            s=50,
            alpha=0.8,
            ax=ax
        )

        ## Annotate points
        palette = dict(zip(umap_df[color_key].unique(), sns.color_palette("tab10")))
        x_median = umap_df["UMAP1"].median()
        y_median = umap_df["UMAP2"].median()
        for sample, (x, y) in umap_df[["UMAP1", "UMAP2"]].iterrows():
            color = palette[umap_df.loc[sample, color_key]]

            ha = "left" if x <= x_median else "right"
            # annotate with a 5-point horizontal offset
            ax.annotate(
                sample,
                xy=(x, y),
                xytext=(5 if ha=="left" else -5, 0),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=7,
                color=color
            )

        ax.set_title("UMAP")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(title=color_key, frameon=True,
                  loc="upper left",
                  bbox_to_anchor=(1.05, 1),
                  borderaxespad=0.)
        fig.tight_layout()

        self.pdf.savefig(fig)

    def _plot_spikein_separation(self, contrast_name: str):
        if contrast_name not in self.contrast_names:
            print(f"Contrast '{contrast_name}' not found.")
            return

        i = self.contrast_names.index(contrast_name)
        logfc = self.log2fc[:, i]
        lognorm = self.adata.layers["lognorm"]  # shape: (samples x proteins)

        # Compute mean log2-intensity per protein
        mean_intensity = np.nanmean(lognorm, axis=0)

        # Classify species
        fasta_headers = self.adata.var["FASTA_HEADERS"].astype(str).fillna("")
        species = ["ECOLI" if re.search(r"ECOLI", header, re.IGNORECASE) else "HUMAN"
                   for header in fasta_headers]

        df = pd.DataFrame({
            "log2FC": logfc,
            "intensity": mean_intensity,
            "species": species,
            "protein": self.protein_names,
        })

        # Group stats
        median_human = df[df["species"] == "HUMAN"]["log2FC"].median()
        median_ecoli = df[df["species"] == "ECOLI"]["log2FC"].median()
        std_human = df[df["species"] == "HUMAN"]["log2FC"].std()
        std_ecoli = df[df["species"] == "ECOLI"]["log2FC"].std()
        delta_median = np.abs(np.round(median_human - median_ecoli, 2))
        delta_std = np.abs(np.round(std_human - std_ecoli, 2))

        # Layout: scatter + vertical histogram
        fig = plt.figure(figsize=(8, 6))
        gs = plt.GridSpec(1, 2, width_ratios=[4, 0.5], wspace=0.02)
        ax_main = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax_main)

        # Scatter
        sns.scatterplot(data=df, x="intensity", y="log2FC", hue="species", ax=ax_main, alpha=0.6, s=7)
        ax_main.axhline(median_human, color="blue", linestyle="--", linewidth=1)
        ax_main.axhline(median_ecoli, color="orange", linestyle="--", linewidth=1)
        ax_main.set_ylim(-2,4) #TODO not manual
        ax_main.set_title(f"{contrast_name}: spike-in separation")
        ax_main.set_xlabel("Mean log2 intensity")
        ax_main.set_ylabel("log2 Fold Change")
        ax_main.legend()

        ax_main.text(
            0.02, 0.95,
            f"Δ median = {delta_median}\nΔ SD = {delta_std}",
            transform=ax_main.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray")
        )

        # Vertical histograms
        sns.kdeplot(data=df[df["species"] == "HUMAN"], y="log2FC", ax=ax_hist, color="blue", fill=True, alpha=0.5, label="HUMAN")
        sns.kdeplot(data=df[df["species"] == "ECOLI"], y="log2FC", ax=ax_hist, color="orange", fill=True, alpha=0.5, label="ECOLI")
        ax_hist.set_xlabel("Density")
        ax_hist.set_ylabel("")
        #ax_hist.legend()

        # Clean up
        ax_hist.tick_params(axis="y", left=False, labelleft=False)

        self.pdf.savefig(fig)
        plt.close(fig)

