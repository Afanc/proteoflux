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
from scipy.cluster import hierarchy as sch
import matplotlib.patches as mpatches
from anndata import AnnData
from typing import Optional, List, Union, Dict
from proteoflux.utils.utils import logger, log_time

MAX_SAMPLES_BARPLOT = 30

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

def _shorten_labels_smart(names: List[str], max_len: int = 28, min_common: int = 5) -> List[str]:
    """
    Produce compact labels that preserve the differing part of similar sample names.

    Logic:
    - Find longest common prefix and suffix across all names.
    - If both exist, keep prefix[:p] + '...' + suffix[-s:], where p+s <= max_len.
    - Always keep the differing middle segment visible if it's short (<= 10).
    - If no clear common parts, fall back to head/tail trim.

    Examples
    --------
    ['1_fileA', '2_fileA']      -> ['1_...fileA', '2_...fileA']
    ['fileA_1', 'fileA_2']      -> ['fileA_1', 'fileA_2']
    ['longprefix_A', 'longprefix_B'] -> ['...A', '...B']
    """
    if not names:
        return names
    if len(names) == 1:
        s = names[0]
        if len(s) <= max_len:
            return names
        return [s[: max_len // 2 - 1] + "..." + s[-(max_len // 2 - 2):]]

    pref = os.path.commonprefix(names)
    rev = [s[::-1] for s in names]
    suff = os.path.commonprefix(rev)[::-1]

    if len(pref) < min_common:
        pref = ""
    if len(suff) < min_common:
        suff = ""

    out = []
    for s in names:
        if len(s) <= max_len:
            out.append(s)
            continue
        # compute the part that differs
        diff_start = len(pref)
        diff_end = len(s) - len(suff)
        diff = s[diff_start:diff_end]
        if len(diff) > 10:  # cap visible diff size
            diff = diff[:5] + "..."
        # compose: prefix (trimmed), diff, suffix (trimmed)
        keep_p = min(len(pref), max_len // 3)
        keep_s = min(len(suff), max_len - keep_p - len(diff) - 3)
        short = s[:keep_p] + diff + "..." + s[len(s) - keep_s:]
        if len(short) > max_len:
            short = short[: max_len // 2 - 1] + "..." + short[-(max_len // 2 - 2):]
        out.append(short)
    return out

def _plot_ids_table_multi_column(ax, df_ids, *, cond_color_map=None):
    """
    Render a compact multi-column table of sample IDs.
    Used when number of samples is too large for a bar plot.
    """
    ax.axis("off")

    # --- for testing
    #target_n = 100
    #df_ids = pd.concat([df_ids] * int(np.ceil(target_n / len(df_ids))), ignore_index=True).iloc[:target_n]
    # ---

    n = len(df_ids)
    max_rows_per_col = 45
    max_cols = 4

    n_cols = min(max_cols, int(np.ceil(n / max_rows_per_col)))
    n_rows = int(np.ceil(n / n_cols))

    # Font size heuristic
    if n_rows <= 50:
        fontsize = 9
    elif n_rows <= 75:
        fontsize = 8
    elif n_rows <= 100:
        fontsize = 7
    elif n_rows <= 150:
        fontsize = 6
    else:
        fontsize = 5

    cols = []
    for i in range(n_cols):
        start = i * n_rows
        end = min((i + 1) * n_rows, n)
        block = df_ids.iloc[start:end]
        cols.append(block)

    gs = GridSpecFromSubplotSpec(
        1,
        n_cols,
        subplot_spec=ax.get_subplotspec(),
        wspace=0.05,
    )

    for i, block in enumerate(cols):
        ax_t = ax.figure.add_subplot(gs[0, i])
        ax_t.axis("off")

        cell_text = block[["Sample", "Condition", "IDs"]].values.tolist()
        table = ax_t.table(
            cellText=cell_text,
            colLabels=["Sample", "Condition", "#IDs"],
            loc="upper left",
            cellLoc="left",
            bbox=[-0.1, 0.0, 1, 0.96],
            colWidths=[0.65, 0.20, 0.15],
        )

        table.set_clip_on(False)
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)

        # Table indices: (row, col) where row=0 is header.
        for (r, c), cell in table.get_celld().items():
            txt = cell.get_text()

            # Body alignment
            cell._loc = "left"
            txt.set_ha("left")
            if c != 0:
                txt.set_ha("center")
            else:
                cell._loc = "left"

            # Color only the Condition text (body rows only)
            if (r > 0) and (c == 1) and cond_color_map is not None:
                label = txt.get_text()
                if label in cond_color_map:
                    txt.set_color(cond_color_map[label])

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

        self.analysis_type = self.dataset_config.get("analysis_type", "")

        self.export_config = self.analysis_config.get("exports")
        self.adata = adata
        self.protein_labels = adata.var.get(protein_label_key, adata.var_names)
        self.protein_names = adata.var["GENE_NAMES"] #TODO option ? 
        self.contrast_names = adata.uns.get("contrast_names", [])

        # Cache stats for fast access
        self.log2fc = adata.varm["log2fc"] if "log2fc" in adata.varm else None

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
        self._has_contrasts = bool(self.contrast_names) and (self.log2fc is not None)

        self.missingness = adata.uns.get("missingness", {})
        #self.fasta_headers = adata.var['FASTA_HEADERS'].to_numpy()

    @log_time("Preparing Pdf Report")
    def plot_all(self):
        """Create the full PDF report to the configured path."""
        with PdfPages(self.export_config.get("path_plot")) as pdf:
            self.pdf = pdf

            # first blank page
            self._plot_title_page()

            # combined IDs + metrics
            self._plot_IDs_and_metrics()

            # hierch. clustering
            self._plot_hclust_heatmap_matplotlib(tag = "intensity")

            # volcanoes
            if self._has_contrasts and (self.q_ebayes is not None):
                self._plot_volcano_plots()

    def _plot_title_page(self):
        """Render a formatted title + intro + pipeline summary + package versions."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        x0, y = 0.05, 0.95
        fig.subplots_adjust(
            left   = x0,
            right  = 0.96,
            top    = 0.98,
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
        analysis_type = self.dataset_config.get("analysis_type", "proteomics")
        xlsx_export = os.path.basename(self.analysis_config.get('exports').get('path_table'))
        h5ad_export = os.path.basename(self.analysis_config.get('exports').get('path_h5ad'))

        # Layout parameters
        line_height = 0.03

        # Title
        if title:
            fig.text(0.5, y, title,
                     ha="center", va="top", fontsize=20, weight="bold")
            y -= 1.3 * line_height

        # datetime
        timestamp = datetime.now().strftime("%Y-%m-%d")
        fig.text(0.5, y, timestamp,
                 ha="center", va="top",
                 fontsize=13, color="black")
        y -= 1.2 * line_height
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

        fig.text(x0, y, "Analysis Type: ",
                 ha="left", va="top", fontsize=12, weight="semibold")
        fig.text(x0 + 0.14, y, f"{analysis_type}",
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
        cont_cfg   = filtering.get("contaminants_files") or []
        flt_meta = self.adata.uns.get("preprocessing").get("filtering")
        quant_meta = self.adata.uns.get("preprocessing").get("quantification")
        base_names = [os.path.basename(f) for f in cont_cfg]
        cont_meta = flt_meta.get("cont") or {}
        removed_cont = cont_meta.get("number_dropped", 0)
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
            y -= 0.8 * line_height

        # Left censoring
        if "min_linear_intensity" in filtering:
            removed_censor = flt_meta.get("censor").get("number_dropped", 0)
            n_r = f"{removed_censor}".replace(",", "'")
            fig.text(x0 + 0.06, y,
                     f"- Left Censoring ≤ {filtering['min_linear_intensity']}: {n_r} PSM removed",
                     ha="left", va="top", fontsize=11)
            y -= 0.8 * line_height

        # Phospho localization filter
        loc_meta = flt_meta.get("loc", {})
        if loc_meta and not loc_meta.get("skipped"):
            removed_loc = loc_meta.get("number_dropped", 0)
            n_r = f"{removed_loc}".replace(",", "'")
            mode = str(loc_meta.get("mode", "")).replace("filter_", "")
            thr = loc_meta.get("threshold", "n/a")
            fig.text(
                x0 + 0.06,
                y,
                f"- Phospho Localization Score ({mode}, thr={thr}): {n_r} sites removed",
                ha="left",
                va="top",
                fontsize=11,
            )
            y -= 0.8 * line_height


        # Quantification method
        quant_txt = quantification_method
        if quantification_method == "directlfq":
            directlfq_min_nonan = quant_meta.get("directlfq_min_nonan", 1)
            quant_txt += f", min nonan={directlfq_min_nonan}"
        fig.text(x0 + 0.02, y, f"- Quantification: {quant_txt}",
                 ha="left", va="top", fontsize=12)

        y -= line_height

        # Normalization
        norm_methods = normalization.get("method", [])
        norm_meta = self.adata.uns.get("preprocessing").get("normalization")
        line_height_mult = 1
        if isinstance(norm_methods, list):
            norm_methods = ", ".join(norm_methods)
        if "loess" in norm_methods:
            loess_span = preproc.get("normalization").get("loess_span")
            norm_methods += f" (loess_span={loess_span})"
        if "median_equalization_by_tag" in norm_methods:
            tags = preproc.get("normalization").get("reference_tag")
            tag_matches = norm_meta.get("tag_matches")
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
            lc_conmed_lod_k = preproc.get("imputation").get("lc_conmed_lod_k", "NA")
            lc_conmed_min_obs = preproc.get("imputation").get("lc_conmed_in_min_obs", "1")
            imp_method += f", lod_k={lc_conmed_lod_k}, min_obs={lc_conmed_min_obs}"

        fig.text(x0 + 0.02, y, f"- Imputation: {imp_method} ",
                 ha="left", va="top", fontsize=12)
        y -= line_height

        # Differential Expression
        if self.pilot_mode and not self._has_contrasts:
            fig.text(x0 + 0.02, y,
                     f"- Pilot study: single-condition run, statistical testing skipped",
                     ha="left", va="top", fontsize=12)
        elif self.pilot_mode:
            fig.text(x0 + 0.02, y, f"- Pilot study: LogFC only", ha="left", va="top", fontsize=12)

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
            y -= 0.6*line_height

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
        raw_conds = (self.adata.obs['CONDITION'].cat.categories
                     if hasattr(self.adata.obs['CONDITION'], 'cat')
                     else self.adata.obs['CONDITION'].unique())
        # If only one condition, show Total-only violins
        single_condition = (len(list(raw_conds)) == 1)
        conds = ['Total'] if single_condition else (['Total'] + list(raw_conds))

        # color map
        color_map = get_color_map(labels=conds)

        # metrics
        mat_all = self.adata.X.T
        res_all = compute_metrics(mat_all, metrics=["CV","RMAD"])
        data_cv = [res_all["CV"]]
        data_rmad = [res_all["RMAD"]]
        if not single_condition:
            for cond in raw_conds:
                mask = self.adata.obs['CONDITION'] == cond
                mat = self.adata.X[mask.values,:].T
                res = compute_metrics(mat, metrics=["CV","RMAD"])
                data_cv.append(res["CV"])
                data_rmad.append(res["RMAD"])

        # plotting
        n_samples = int(len(counts.index))
        use_table = (n_samples > MAX_SAMPLES_BARPLOT)

        fig = plt.figure(figsize=(12,12))

        if use_table:
            # Table needs more width; barplot benefits from extra right margin for legend.
            fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.98)
            outer_hspace = 0.10
            # Allocate progressively more vertical space to the (table) top panel as n grows. up to 400 +- readable
            top_ratio = float(np.clip(3.0 + (n_samples - 250) / 150.0, 2.0, 5.0))

            #top_ratio = 2.0
            bottom_ratio = 1.0
        else:
            fig.subplots_adjust(top=0.95, bottom=0.10, left=0.10, right=0.85)
            outer_hspace = 0.40
            top_ratio = 1.0
            bottom_ratio = 1.0

        outer = GridSpec(2, 1, height_ratios=[top_ratio, bottom_ratio], hspace=outer_hspace)

        ax_bar = fig.add_subplot(outer[0])
        inner = GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],wspace=0.3)
        ax_rm = fig.add_subplot(inner[0])
        ax_cv = fig.add_subplot(inner[1])

        # barplot or table (keep the same space on page)
        if not use_table:
            sample_colors = [color_map.get(c, color_map["Total"]) for c in self.adata.obs["CONDITION"]]
            ax_bar.grid(axis="y", which="both", visible=True)
            ax_bar.bar(counts.index, counts.values, color=sample_colors)
            ax_bar.set_xticks(range(len(counts.index)))

            # short sample names
            sample_names = list(counts.index)
            sample_names_short = _shorten_labels_smart(sample_names, max_len=28, min_common=5)
            ax_bar.set_xticklabels(sample_names_short, rotation=45, ha="right")

            id_txt = "Protein IDs per Sample"
            if self.analysis_type == "peptidomics":
                id_txt = "Peptide IDs per Sample"
            elif self.analysis_type == "phosphoproteomics":
                id_txt = "Phosphosite IDs per Sample"

            ax_bar.set_ylabel("Number of IDs")
            ax_bar.set_ylim([0, np.max(counts.values) + 850])  # padding for annotations
            ax_bar.set_title(id_txt, pad=5)
            for i, val in enumerate(counts.values):
                ax_bar.text(
                    i, val + 50, str(int(val)),
                    ha="center", va="bottom", fontsize=8, rotation=45
                )
        else:
            # Black/white table: Sample, Condition, #IDs
            sample_names = list(counts.index)
            sample_names_short = _shorten_labels_smart(sample_names, max_len=28, min_common=5)
            cond_series = self.adata.obs.loc[sample_names, "CONDITION"].astype(str).to_numpy()
            df_ids = pd.DataFrame(
                {
                    "Sample": sample_names_short,
                    "Condition": cond_series,
                    "IDs": [int(x) for x in counts.values],
                }
            )
            ax_bar.set_title("Protein IDs per Sample")
            _plot_ids_table_multi_column(ax_bar, df_ids, cond_color_map=color_map)

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
        if not use_table:
            handles = [mpatches.Patch(color=color_map[c],label=c) for c in conds]
            ax_bar.legend(handles=handles,title='Condition',bbox_to_anchor=(1.05,1),loc='upper left')
        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_hclust_heatmap_matplotlib(self, tag: str = "intensity"):
        """
        Hierarchical clustering clustergram (dendrograms + heatmap) using
        the precomputed linkages and orders stored in `adata.uns`.
        """
        # --- 1) pick the matrix (samples × features) and convert to DataFrame (features × samples)
        X = self.adata.X
        M = X.toarray() if hasattr(X, "toarray") else X
        title = "Hierarchical clustering (LC imputed)"

        df = pd.DataFrame(M.T, index=self.adata.var_names, columns=self.adata.obs_names)

        # --- 2) apply precomputed order if present
        feat_order   = self.adata.uns.get(f"{tag}_feature_order")
        sample_order = self.adata.uns.get(f"{tag}_sample_order")
        if feat_order is not None:
            df = df.reindex(index=list(feat_order))
        if sample_order is not None:
            df = df.reindex(columns=list(sample_order))

        # --- 3) linkage trees: use precomputed linkages if available, else compute
        row_link = self.adata.uns.get(f"{tag}_feature_linkage")
        col_link = self.adata.uns.get(f"{tag}_sample_linkage")

        # If orders were not supplied, derive from the linkages now
        if feat_order is None:
            feat_order = [df.index[i] for i in sch.leaves_list(row_link)]
            df = df.reindex(index=feat_order)
        if sample_order is None:
            sample_order = [df.columns[i] for i in sch.leaves_list(col_link)]
            df = df.reindex(columns=sample_order)

        # --- 4) determine heatmap scale (diverging if centered spans +/-)
        vmin = float(np.nanmin(df.values))
        vmax = float(np.nanmax(df.values))
        cmap = "viridis"

        # --- 5) layout: top dendrogram, condition strip, heatmap + dedicated colorbar column
        fig = plt.figure(figsize=(10.5, 9.0))
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.15, top=0.94, wspace=0.0, hspace=0.0)
        gs = GridSpec(
            3, 4,
            width_ratios=[1.0, 4.0, 0.15, 0.12],
            height_ratios=[1.0, 0.12, 4.0],
        )

        ax_dend_top  = fig.add_subplot(gs[0, 1])
        ax_dend_left = fig.add_subplot(gs[2, 0])
        ax_ann       = fig.add_subplot(gs[1, 1])
        ax_heat      = fig.add_subplot(gs[2, 1])
        ax_cbar      = fig.add_subplot(gs[2, 3])

        # top dendrogram (samples)
        sch.dendrogram(
            col_link, ax=ax_dend_top, orientation="top", no_labels=True, color_threshold=None
        )
        ax_dend_top.set_xticks([]); ax_dend_top.set_yticks([])
        for spine in ax_dend_top.spines.values():
            spine.set_visible(False)

        # left dendrogram (features)
        sch.dendrogram(
            row_link, ax=ax_dend_left, orientation="left", no_labels=True, color_threshold=None
        )
        ax_dend_left.set_xticks([]); ax_dend_left.set_yticks([])
        for spine in ax_dend_left.spines.values():
            spine.set_visible(False)

        # Condition annotation strip (aligned with clustered sample order).
        # IMPORTANT: color mapping must match the rest of the report (order-dependent).
        cond_series = self.adata.obs.loc[list(df.columns), "CONDITION"].astype(str).to_numpy()

        all_conds = (
            list(self.adata.obs["CONDITION"].cat.categories.astype(str))
            if hasattr(self.adata.obs["CONDITION"], "cat")
            else pd.unique(self.adata.obs["CONDITION"].astype(str)).tolist()
        )
        # Match _plot_IDs_and_metrics(): anchor 'Total' first, then conditions in stable order.
        cond_color_map = get_color_map(labels=(["Total"] + list(all_conds)))

        rgb = np.array(
            [matplotlib.colors.to_rgb(cond_color_map.get(c, cond_color_map["Total"])) for c in cond_series],
            dtype=float,
        )[None, :, :]

        ax_ann.imshow(rgb, aspect="auto", interpolation="nearest")
        ax_ann.set_xticks([])
        ax_ann.set_yticks([])
        for spine in ax_ann.spines.values():
            spine.set_visible(False)

        # heatmap (already reordered)
        im = ax_heat.imshow(
            df.values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        # Ensure strip and heatmap share identical horizontal geometry
        fig.suptitle(title, fontsize=13, y=0.98)

        # neat axes: y is features (no tick labels for speed), x shows shortened sample names
        names_short = _shorten_labels_smart(list(df.columns), max_len=28, min_common=5)
        ax_heat.set_xticks(range(len(names_short)))
        ax_heat.set_xticklabels(names_short, rotation=45, ha="right", fontsize=8)
        ax_heat.set_yticks([])

        fig.colorbar(im, cax=ax_cbar)
        ax_cbar.set_ylabel("Log Intensity")

        self.pdf.savefig(fig)
        plt.close(fig)

    def _plot_volcano_plots(self):
        """Volcano plots (one per contrast). eBayes-only; skip in pilot mode."""

        exports_cfg = self.analysis_config.get("exports", {}) or {}
        sign_threshold = float(
            exports_cfg.get(
                "volcano_sign_threshold",
                self.analysis_config.get("sign_threshold", 0.05),
            )
        )
        volcano_top_annotated = int(exports_cfg.get("volcano_top_annotated", 10))

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
