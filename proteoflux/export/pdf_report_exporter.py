"""PDF report exporter for Proteoflux results.

Generates a multi-page PDF containing:
  1) Title/summary page (pipeline settings, filtering counts, package versions)
  2) IDs per sample + condition-level variability metrics
  3) Volcano plots per contrast (annotated top hits)

All visuals rely on the contents of an AnnData produced by the pipeline.
"""

import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
import proteoflux
import inmoose
import polars
import scipy
import sklearn
import skmisc
import anndata
import platform
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
from anndata import AnnData
from typing import Optional, List, Union, Dict
from proteoflux.utils.utils import logger, log_time

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":       "STIXGeneral",
    "mathtext.fontset":  "stix",
})

def get_color_map(
    labels: List[str],
    palette: Optional[List[str]] = None,
    anchor: str = "Total",
    anchor_color: str = "red",
) -> Dict[str,str]:
    """Return a stable mapping label→color.

    - If `anchor` is present in labels, it gets `anchor_color`.
    - Remaining labels (original order) get colors from Matplotlib's cycle.
    """
    palette = palette or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    out: Dict[str,str] = {}
    # assign anchor first
    if anchor in labels:
        out[anchor] = anchor_color
    # assign the rest
    others = [l for l in labels if l != anchor]
    for i, lbl in enumerate(others):
        out[lbl] = palette[i % len(palette)]
    return out


def CV(values: np.ndarray, axis: int = 1, log_base: float = 2) -> np.ndarray:
    """Compute %CV from log-transformed values.

    Values are exponentiated back to linear space before computing std/mean.
    Args:
        values: log-transformed intensity matrix.
        axis:   axis to compute along (features by default).
        log_base: log base used for the original transform.
    Returns:
        Vector of %CV estimates (float array).
    """
    linear_values = log_base**values

    std = np.nanstd(linear_values, axis=axis)
    mean = np.clip(np.nanmean(linear_values, axis=axis), 1e-6, None)

    cv = 100 * std/mean

    return cv

def rMAD(values: np.ndarray, axis: int = 1, log_base: float = 2) -> np.ndarray:
    """Compute %rMAD from log-transformed values.

    Values are exponentiated back to linear space before computing rMAD.
    Args/Returns mirror `CV`.
    """
    linear = log_base**values

    med = np.nanmedian(linear, axis=axis)
    mad = np.nanmedian(np.abs(linear - med[:, None]), axis=axis)

    rmad = 100 * mad / np.clip(med, 1e-6, None)

    return rmad

def compute_metrics(mat: np.ndarray, metrics: List[str] = ["CV", "MAD"]) -> Dict[str, np.ndarray]:
    """Compute per-feature metrics across samples (features × samples).

    Supported keys:
      - "CV":   coefficient of variation (via `CV`, log-aware)
      - "RMAD": relative MAD (via `rMAD`, log-aware)
      - "Mean", "Median", "STD": standard statistics in log space
    Default behavior computes CV and RMAD (legacy default shows "MAD" but maps to RMAD).
    """
    if metrics is None:
        metrics = ["CV", "RMAD"]

    result = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        means   = np.nanmean(mat, axis=1)
        medians = np.nanmedian(mat, axis=1)
        stds    = np.nanstd(mat, axis=1, ddof=1)
        CVs = CV(mat, axis=1, log_base=2)
        rmads = rMAD(mat, axis=1, log_base=2)

        if "CV" in metrics:
            result["CV"] = CVs
        if "RMAD" in metrics:
            result["RMAD"] = rmads
        if "Mean" in metrics:
            result["Mean"] = means
        if "Median" in metrics:
            result["Median"] = medians
        if "STD" in metrics:
            result["STD"] = stds

    return result

class ReportPlotter:
    """Prepare plotting context from AnnData and config dict."""
    def __init__(
        self,
        adata: AnnData,
        config: Dict,
        protein_label_key: str = "FASTA_HEADERS",
    ):
        self.config = config
        self.dataset_config = config.get("dataset", {})
        self.analysis_config = config.get("analysis", {})

        self.export_config = self.analysis_config.get("exports")
        self.adata = adata
        self.protein_labels = adata.var.get(protein_label_key, adata.var_names)
        self.protein_names = adata.var["GENE_NAMES"] #TODO option ? 
        self.contrast_names = adata.uns.get("contrast_names", [f"C{i}" for i in range(adata.varm['log2fc'].shape[1])])

        # Cache stats for fast access
        self.log2fc = adata.varm["log2fc"]
        # eBayes may be absent in pilot mode
        self.q_ebayes = (pd.DataFrame(adata.varm["q_ebayes"],
                                      columns=self.contrast_names,
                                      index=adata.var.index)
                         if "q_ebayes" in adata.varm else None)
        self.p_ebayes = (pd.DataFrame(adata.varm["p_ebayes"],
                                      columns=self.contrast_names,
                                      index=adata.var.index)
                         if "p_ebayes" in adata.varm else None)
        # pilot mode flag (set by limma_pipeline)
        self.pilot_mode = bool(adata.uns.get("pilot_study_mode", False))

        self.missingness = adata.uns.get("missingness", {})
        self.fasta_headers = adata.var['FASTA_HEADERS'].to_numpy()

    @log_time("Preparing Pdf Report")
    def plot_all(self):
        """Create the full PDF report to the configured path."""
        with PdfPages(self.export_config.get("path_plot")) as pdf:
            self.pdf = pdf
            # first blank page
            self._plot_title_page()
            # combined IDs + metrics
            self._plot_IDs_and_metrics()
            # volcanoes
            self._plot_volcano_plots()

    def _plot_title_page(self):
        """Render a formatted title + intro + pipeline summary + package versions."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        x0, y = 0.05, 0.95
        fig.subplots_adjust(
            left   = x0,
            right  = 0.96,
            top    = 0.96,
            bottom = 0.04
        )

        # Fetch text from config
        title       = self.analysis_config.get("title", "")
        intro       = self.analysis_config.get("intro_text", "")
        footer_text = self.analysis_config.get("footer_text", "")
        preproc     = self.config.get("preprocessing", {})
        filtering   = preproc.get("filtering", {})
        quantification_method = preproc.get("pivot_signal_method", "sum")
        normalization = preproc.get("normalization", {})
        imputation  = preproc.get("imputation", {})
        de_method   = self.analysis_config.get("ebayes_method", "limma")
        input_layout = self.dataset_config.get("input_layout", "")
        xlsx_export = os.path.basename(self.analysis_config.get('exports').get('path_table'))
        h5ad_export = os.path.basename(self.analysis_config.get('exports').get('path_h5ad'))

        # Layout parameters
        line_height = 0.03

        # Title
        if title:
            fig.text(0.5, y, title,
                     ha="center", va="top", fontsize=20, weight="bold")
            y -= 1.5 * line_height

        # datetime
        timestamp = datetime.now().strftime("%Y-%m-%d")
        fig.text(0.5, y, timestamp,
                 ha="center", va="top",
                 fontsize=13, color="black")
        y -= 1.5 * line_height
        # Intro
        if intro:
            wrapped_lines = []
            for para in intro.split("\n"):
                wrapped = textwrap.fill(para, width=105)
                wrapped_lines.append(wrapped)
            for line in wrapped_lines:
                fig.text(x0, y, line, ha="left", va="top", fontsize=12)
                y -= 0.5 * line_height

        # Summary generated files
        fig.text(x0, y, "Input Layout: ",
                 ha="left", va="top", fontsize=12, weight="semibold")
        fig.text(x0 + 0.14, y, f"{input_layout}",
                 ha="left", va="top", fontsize=12)
        y -= line_height

        fig.text(x0, y, "Output files:", ha="left", va="top",
                 fontsize=12, weight="semibold")
        y -= 0.8 * line_height
        fig.text(x0+0.05, y, f"- {xlsx_export} (full data table)", ha="left", va="top", fontsize=12)
        y -= 0.8 * line_height
        fig.text(x0+0.05, y, f"- {h5ad_export} (compressed, viewable in ProteoViewer)", ha="left", va="top", fontsize=12)
        y -= line_height

        # Pipeline steps
        fig.text(x0, y, "Pipeline steps:", ha="left", va="top",
                 fontsize=14, weight="semibold")
        y -= line_height

        # Filtering
        fig.text(x0 + 0.02, y, "- Filtering:", ha="left", va="top",
                 fontsize=12)
        y -= line_height

        # Contaminants
        cont_cfg   = filtering.get("contaminants_files", [])
        flt_meta = self.adata.uns.get("preprocessing").get("filtering")
        base_names = [os.path.basename(f) for f in cont_cfg]
        removed_cont = flt_meta.get("cont").get("number_dropped", 0)
        n_cont = f"{removed_cont}".replace(",", "'")
        fig.text(x0 + 0.06, y,
                 f"- Contaminants ({', '.join(base_names)}): {n_cont} PSM removed",
                 ha="left", va="top", fontsize=11)
        y -= 0.8 * line_height

        # q-value threshold
        if "qvalue" in filtering:
            removed_q = flt_meta.get("qvalue").get("number_dropped", 0)
            n_q = f"{removed_q}".replace(",","'")
            fig.text(x0 + 0.06, y,
                     f"- q-value ≤ {filtering['qvalue']}: {n_q} PSM removed",
                     ha="left", va="top", fontsize=11)
            y -= 0.8 * line_height

        # PEP threshold
        if "pep" in filtering:
            removed_pep = flt_meta.get("pep").get("number_dropped", 0)
            op = "≥" if flt_meta.get("pep").get("direction").startswith("greater") else "≤"
            n_p = f"{removed_pep}".replace(",", "'")
            fig.text(x0 + 0.06, y,
                     f"- PEP {op} {filtering['pep']}: {n_p} PSM removed",
                     ha="left", va="top", fontsize=11)
            y -= 0.8 * line_height

        # Min precursors
        if "min_precursors" in filtering:
            removed_prec = flt_meta.get("prec").get("number_dropped", 0)
            n_r = f"{removed_prec}".replace(",", "'")
            fig.text(x0 + 0.06, y,
                     f"- Min. Num. Precursors = {filtering['min_precursors']}: {n_r} PSM removed",
                     ha="left", va="top", fontsize=11)
            y -= line_height

        # Quantification method
        fig.text(x0 + 0.02, y, f"- Quantification: {quantification_method}",
                 ha="left", va="top", fontsize=12)

        y -= line_height

        # Normalization
        norm_methods = normalization.get("method", [])
        line_height_mult = 1
        if isinstance(norm_methods, list):
            norm_methods = ", ".join(norm_methods)
        if "loess" in norm_methods:
            loess_span = preproc.get("normalization").get("loess_span")
            norm_methods += f" (loess_span={loess_span})"
        if "median_equalization_by_tag" in norm_methods:
            tags = preproc.get("normalization").get("reference_tag")
            tag_matches = preproc.get("normalization").get("tag_matches")
            norm_methods += f"\n   tags={tags}, matches={tag_matches}"
            line_height_mult = 1.5

        fig.text(x0 + 0.02, y, f"- Normalization: {norm_methods}",
                 ha="left", va="top", fontsize=12)
        y -= line_height_mult*line_height

        # Imputation
        imp = preproc.get("imputation", {})
        imp_method = imp.get("method", [])
        if "knn" in imp_method:
            extras = []
            if "knn_k" in imp:
                extras.append(f"k={imp['knn_k']}")
            if "tnknn" in imp_method and "knn_tn_perc" in imp:
                extras.append(f"tn_perc={imp['knn_tn_perc']}")
            if extras:
                imp_method += " (" + ", ".join(extras) + ")"

        if "rf" in imp_method:
            rf_max_iter = preproc.get("imputation").get("rf_max_iter")
            imp_method += f", rf_max_iter={rf_max_iter}"

        if "lc_conmed" in imp_method:
            lc_conmed_lod_k = preproc.get("imputation").get("lc_conmed_lod_k")
            imp_method += f", lod_k={lc_conmed_lod_k}"

        fig.text(x0 + 0.02, y, f"- Imputation: {imp_method} ",
                 ha="left", va="top", fontsize=12)
        y -= line_height

        # Differential Expression
        if self.pilot_mode:
            fig.text(x0 + 0.02, y,
                     f"- Pilot Study, LogFC only",
                     ha="left", va="top", fontsize=12)
        else:
            fig.text(x0 + 0.02, y,
                     f"- Differential expression: eBayes via {de_method}",
                     ha="left", va="top", fontsize=12)
        y -= line_height

        # Package versions
        fig.text(x0, y, "Key package versions:", ha="left", va="top",
                 fontsize=14, weight="semibold")
        y -= line_height

        pkgs = {
            "python":       platform.python_version(),
            "proteoflux":   proteoflux.__version__,
            "inmoose":      inmoose.__version__,
            "numpy":        np.__version__,
            "pandas":       pd.__version__,
            "polars":       polars.__version__,
            "anndata":      anndata.__version__,
            "scanpy":       sc.__version__,
            "scipy":        scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "skmisc":       skmisc.__version__,
        }
        for name, ver in pkgs.items():
            fig.text(x0 + 0.02, y, f"- {name}: {ver}",
                     ha="left", va="top", fontsize=12)
            y -= 0.8*line_height

        y -= line_height

        fig.text(x0, y,
            "This analysis was performed using Proteoflux. \nFor full pipeline details, definitions, and citation instructions, see: [doi.org/xxx]",
            ha="left", va="bottom", fontsize=12)

        # footer
        fig.text(0.5, 0.02,
            f"{footer_text}",
            ha="center", va="bottom",
            fontsize=12, style="italic", color="gray")

        # Save and close
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_IDs_and_metrics(self):
        """Bar: IDs per sample; Violin: %CV/%rMAD per condition (single page)."""

        # prepare data
        df_raw = pd.DataFrame(self.adata.layers['raw'], index=self.adata.obs_names, columns=self.adata.var_names)
        counts = df_raw.notna().sum(axis=1)
        raw_conds = (self.adata.obs['CONDITION'].cat.categories if hasattr(self.adata.obs['CONDITION'], 'cat') else self.adata.obs['CONDITION'].unique())
        conds = ['Total'] + list(raw_conds)
        # color map
        color_map = get_color_map(labels=conds)
        # metrics
        mat_all = self.adata.X.T
        res_all = compute_metrics(mat_all, metrics=["CV","RMAD"])
        data_cv = [res_all["CV"]]
        data_rmad = [res_all["RMAD"]]
        for cond in raw_conds:
            mask = self.adata.obs['CONDITION'] == cond
            mat = self.adata.X[mask.values,:].T
            res = compute_metrics(mat, metrics=["CV","RMAD"])
            data_cv.append(res["CV"])
            data_rmad.append(res["RMAD"])
        # plotting
        fig = plt.figure(figsize=(12,12))
        fig.subplots_adjust(top=0.95, bottom=0.10, left=0.10, right=0.85)
        outer = GridSpec(2,1,height_ratios=[1,1],hspace=0.5)
        ax_bar = fig.add_subplot(outer[0])
        inner = GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.3)
        ax_rm = fig.add_subplot(inner[0])
        ax_cv = fig.add_subplot(inner[1])
        # barplot
        sample_colors = [color_map[c] for c in self.adata.obs['CONDITION']]
        ax_bar.grid(axis='y', which='both', visible=True)
        ax_bar.bar(counts.index, counts.values, color=sample_colors)
        ax_bar.set_xticks(range(len(counts.index)))
        ax_bar.set_xticklabels(counts.index,rotation=45, ha='right')
        ax_bar.set_ylabel('Number of IDs')
        ax_bar.set_ylim([0, np.max(counts.values)+700]) #add some padding for annotation
        ax_bar.set_title('Protein IDs per Sample')
        for i, val in enumerate(counts.values):
            ax_bar.text(i, val+50, str(int(val)),
                        ha='center', va='bottom', fontsize=8, rotation=45)

        # violins
        parts_rm = ax_rm.violinplot(data_rmad,positions=range(len(conds)),showmeans=False,showextrema=False,showmedians=True)
        parts_cv = ax_cv.violinplot(data_cv,positions=range(len(conds)),showmeans=False,showextrema=False,showmedians=True)
        for i,body in enumerate(parts_rm['bodies']):
            body.set_facecolor(color_map[conds[i]]);
            body.set_edgecolor('black'); body.set_alpha(0.7)
        for i,body in enumerate(parts_cv['bodies']):
            body.set_facecolor(color_map[conds[i]]);
            body.set_edgecolor('black'); body.set_alpha(0.7)
        ax_rm.set_xticks(range(len(conds))); ax_rm.set_xticklabels(conds,rotation=45,ha='right')
        ax_rm.set_ylabel('%rMAD'); ax_rm.set_title('%rMAD per Condition')
        ax_rm.grid(axis='y', which='both', visible=True)
        ax_cv.set_xticks(range(len(conds))); ax_cv.set_xticklabels(conds,rotation=45,ha='right')
        ax_cv.set_ylabel('%CV'); ax_cv.set_title('%CV per Condition')
        ax_cv.grid(axis='y', which='both', visible=True)
        for i, arr in enumerate(data_rmad):
            med = np.nanmedian(arr)
            ax_rm.text(i, med, f'{med:.1f}',
                       ha='center', va='bottom', fontsize=8)

        ax_cv.set_xticks(range(len(conds)))
        ax_cv.set_xticklabels(conds, rotation=45, ha='right')
        ax_cv.set_ylabel('%CV'); ax_cv.set_title('%CV per Condition')
        for i, arr in enumerate(data_cv):
            med = np.nanmedian(arr)
            ax_cv.text(i, med, f'{med:.1f}',
                       ha='center', va='bottom', fontsize=8)

        # shared legend
        handles = [mpatches.Patch(color=color_map[c],label=c) for c in conds]
        ax_bar.legend(handles=handles,title='Condition',bbox_to_anchor=(1.05,1),loc='upper left')
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_volcano_plots(self):
        """Volcano plots (one per contrast). eBayes-only; skip in pilot mode."""

        if (self.q_ebayes is None):
            return

        sign_threshold = self.analysis_config.get("sign_threshold", 0.05)
        volcano_top_annotated = self.analysis_config.get("exports").get("volcano_top_annotated", 10)
        for i, name in enumerate(self.contrast_names):
            logfc = self.log2fc[:, i]
            # choose source: eBayes if available, else raw
            qvals_src = self.q_ebayes.iloc[:, i]
            pvals_src = self.p_ebayes.iloc[:, i]

            group1, group2 = name.split("_vs_")

            # Map missingness from DataFrame
            miss_df = self.missingness
            missing_a = miss_df[group1].values >= 1.0
            missing_b = miss_df[group2].values >= 1.0
            both_missing = missing_a & missing_b

            color = np.full(logfc.shape[0], "gray", dtype=object)

            def plot_panel(ax, qvals_for_y, qvals_for_pick, title):
                mask = ~(missing_a | missing_b)

                base_color = color[mask]
                base_logfc = logfc[mask]
                base_q = qvals_for_y.iloc[mask]
                ax.scatter(base_logfc, -np.log10(base_q), c=base_color, alpha=0.7, s=10)


                ax.axhline(-np.log10(sign_threshold), color="black", linestyle="--", linewidth=0.8)
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

                # annotate top hits, excluding any protein missing in either group
                for direction in ["up", "down"]:
                    dir_mask = (logfc > 0) if direction == "up" else (logfc < 0)
                    sig_mask = mask & dir_mask & (qvals_for_pick < sign_threshold)

                    # pick top by smallest q-value
                    top_idx = np.argsort(qvals_for_pick[sig_mask])[:volcano_top_annotated]
                    selected = np.where(sig_mask)[0][top_idx]

                    for j in selected:
                        ax.text(
                            logfc[j],
                            -np.log10(qvals_for_y.iloc[j]),
                            self.protein_names.iloc[j],
                            fontsize=6,
                            ha="right" if logfc[j] > 0 else "left",
                            va="bottom",
                        )

                ax.set_xlabel("log2FC")
                ax.set_ylabel("-log10(q)")
                ax.set_title(title)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # Use q on both axes/selection (consistent with ylabel)
            plot_panel(ax, qvals_src, qvals_src, f"{name}")

            fig.tight_layout()
            self.pdf.savefig(fig)
            plt.close(fig)
