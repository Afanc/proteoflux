"""
Low-level Plotly utilities for ProteoFlux panel app.
"""

import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
from typing import List, Sequence, Optional, Dict, Tuple, Union, Literal
from anndata import AnnData
import scanpy as sc
from proteoflux.utils.utils import logger, log_time
from functools import lru_cache

def get_color_map(
    labels: List[str],
    palette: List[str] = None,
    anchor: str = "Total",
    anchor_color: str = "black",
) -> Dict[str,str]:
    """
    Returns a stable mapping label→color.

    - anchor: if present in labels, gets anchor_color.
    - the remaining labels (sorted) get colors from palette in order.
    """
    palette = palette or px.colors.qualitative.Plotly
    out: Dict[str,str] = {}
    # assign anchor first (so it's always the same)
    if anchor in labels:
        out[anchor] = anchor_color
    # assign the rest in sorted order
    others = sorted(l for l in labels if l != anchor)
    for i, lbl in enumerate(others):
        out[lbl] = palette[i % len(palette)]
    return out

def categorize_proteins_by_run_count(df: pd.DataFrame) -> pd.Series:
    run_counts = df.notna().sum(axis=1)
    total_runs = df.shape[1]
    half_runs  = total_runs * 0.5

    def _cat(n):
        if n == total_runs: return "complete"
        if n == 1:          return "unique"
        if n <= half_runs:  return "sparse"
        return "shared"

    return run_counts.map(_cat)


def plot_stacked_proteins_by_category(
    im,
    matrix_key: str = "normalized",
    category_colors: Dict[str,str] = None,
    width: int = 1200,
    height: int = 500,
    title: str = "Protein IDs by Sample and Category",
) -> go.Figure:
    # 1) proteins×samples
    df = pd.DataFrame(im.matrices[matrix_key], columns=im.columns)

    # 2) categorize proteins
    prot_cat = categorize_proteins_by_run_count(df)

    # 3) pivot sample×category counts
    samples = df.columns.tolist()
    cats    = ["complete","shared","sparse","unique"]
    pivot   = pd.DataFrame(
        {cat: df.loc[prot_cat==cat].notna().sum(axis=0) for cat in cats},
        index=samples
    )

    # 4) condition colors for bar borders
    cond_df = (
        im.dfs["condition_pivot"]
          .to_pandas()
          .rename(columns={"CONDITION":"Condition"})
          .set_index("Sample")
    )
    sample_conditions = cond_df["Condition"].reindex(samples)
    unique_conds      = sample_conditions.unique().tolist()
    cond_palette      = px.colors.qualitative.Plotly
    cond_color_map    = get_color_map(unique_conds, px.colors.qualitative.Plotly)
    edge_colors       = [cond_color_map[sample_conditions[s]] for s in samples]

    # default fills
    if category_colors is None:
        category_colors = {
            "complete": "darkgray",
            "shared"  : "lightgray",
            "sparse"  : "white",
            "unique"  : "red",
        }

    fig = go.Figure()

    # 5) actual bar traces (no legend entries)
    for cat in cats:
        fig.add_trace(go.Bar(
            x=samples, y=pivot[cat],
            marker_color=category_colors[cat],
            marker_line_color=edge_colors,
            marker_line_width=2,
            showlegend=False,
            hovertemplate=(
                f"Category: {cat.capitalize()}<br>"+
                "Count: %{y}<extra></extra>"
            ),
        ))

    # 6) total annotations
    totals = pivot.sum(axis=1)
    for s, tot in totals.items():
        fig.add_annotation(
            x=s, y=tot,
            text=str(tot),
            showarrow=False,
            yanchor="bottom"
        )

    # 7) Protein Category legend (left)
    for i, cat in enumerate(cats):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=cat.capitalize(),
            marker=dict(
                symbol="square",
                size=12,
                color=category_colors[cat],
                line=dict(color="black", width=2),
            ),
            legend="legend1",
            legendgrouptitle_text="Protein Category",
        ))

    # 8) Sample Condition legend (right)
    for j, cond in enumerate(unique_conds):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=cond,
            marker=dict(
                symbol="square",
                size=12,
                color="white",
                line=dict(color=cond_color_map[cond], width=2),
            ),
            legend="legend2",
            legendgrouptitle_text="Sample Condition",
        ))

    # 9) layout with two legends
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5),
        xaxis_title="Sample",
        yaxis_title="Number of Proteins",
        template="plotly_white",
        width=width, height=height,
        # first legend (categories) on the left
        legend=dict(
            x=1.40, y=1,
            xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1
        ),
        # second legend (conditions) on the right
        legend2=dict(
            x=1.20, y=1,
            xanchor="right", yanchor="top",
            bordercolor="black", borderwidth=1
        ),
        legend_itemclick=False,
        legend_itemdoubleclick=False,
        #margin=dict(l=120, r=120),
    )
    fig.update_xaxes(tickangle=45)
    return fig

def plot_bar_plotly(
    x: Sequence,
    y: Sequence,
    colors: Union[str, Sequence[str]] = "steelblue",
    orientation: str = "v",             # "v" or "h"
    width: int = 900,
    height: int = 500,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    template: str = "plotly_white",
    **bar_kwargs,                       # passed to go.Bar
) -> go.Figure:
    """
    Generic bar plot.

    Parameters
    ----------
    x : sequence of category labels
    y : sequence of numeric values
    colors : single color or sequence matching len(x)
    orientation : 'v' or 'h'
    **bar_kwargs : any go.Bar args (e.g. opacity, marker_line)
    """
    fig = go.Figure(go.Bar(
        x=x if orientation=="v" else y,
        y=y if orientation=="v" else x,
        marker_color=colors,
        orientation=orientation,
        **bar_kwargs
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=template,
        width=width,
        height=height,
        barmode="group",
    )
    return fig

def compute_metric_by_condition(
    df_mat: pd.DataFrame,
    cond_map: pd.Series,
    metric: Literal["CV", "rMAD"] = "CV",
) -> Dict[str, np.ndarray]:
    """
    Fast, safe, truly warning-free computation of %CV or rMAD per condition.
    """
    out: Dict[str, np.ndarray] = {}
    arr = df_mat.values

    def compute_cv(x: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(x)
        n_valid = valid.sum(axis=1)
        # Mask rows with <= 1 valid values
        mask = n_valid > 1
        mean = np.full(x.shape[0], np.nan)
        std = np.full(x.shape[0], np.nan)

        if np.any(mask):
            mean[mask] = np.nanmean(x[mask], axis=1)
            std[mask] = np.nanstd(x[mask], axis=1, ddof=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(mean != 0, std / mean * 100, np.nan)
        return result

    def compute_rmad(x: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(x)
        n_valid = valid.sum(axis=1)
        mask = n_valid > 0
        med = np.full(x.shape[0], np.nan)
        mad = np.full(x.shape[0], np.nan)

        if np.any(mask):
            med[mask] = np.nanmedian(x[mask], axis=1)
            mad[mask] = np.nanmedian(np.abs(x[mask] - med[mask][:, None]), axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(med != 0, mad / med * 100, np.nan)
        return result

    compute_fn = compute_cv if metric == "CV" else compute_rmad

    # Global ("Total")
    out["Total"] = compute_fn(arr)

    # Per-condition
    for cond in cond_map.unique():
        cols = cond_map[cond_map == cond].index
        sub_arr = df_mat[cols].values
        out[cond] = compute_fn(sub_arr)

    return out

def compute_cv_by_condition(
    df_mat: pd.DataFrame,
    cond_map: pd.Series
) -> Dict[str, np.ndarray]:
    cvs: Dict[str, np.ndarray] = {}
    # 0) global CV across all samples
    mean_all = df_mat.mean(axis=1)
    sd_all   = df_mat.std(axis=1, ddof=1)
    cvs["Total"] = (sd_all / mean_all * 100).values

    # 1) per‐condition CV
    for cond in cond_map.unique():
        cols = cond_map[cond_map == cond].index
        sub  = df_mat[cols]
        mean = sub.mean(axis=1)
        sd   = sub.std(axis=1, ddof=1)
        cvs[cond] = (sd / mean * 100).values

    return cvs

def plot_violins(
    data: Dict[str, Sequence],
    colors: Dict[str,str] = None,
    title: str = None,
    width: int = 900,
    height: int = 500,
    y_title: str = None,
    x_title: str = None,
    showlegend: bool = True,
) -> go.Figure:
    """
    Draw one full violin per key in `data`.
    - data: {label: array_of_values}
    - colors: optional mapping label→color (falls back to Plotly palette)
    """
    labels = list(data.keys())

    # default palette if not provided
    default_colors = px.colors.qualitative.Plotly
    color_map = colors or dict(zip(labels, default_colors))

    fig = go.Figure()
    for lbl in labels:
        arr = np.asarray(data[lbl])
        fig.add_trace(go.Violin(
            x=[lbl]*len(arr),
            y=arr,
            name=lbl,
            legendgroup=lbl,
            line_color=color_map[lbl],
            opacity=0.6,
            width=0.7,
            box_visible=True,
            meanline_visible=True,
            points=False,
            hoverinfo="skip",
        ))

    y_all = np.concatenate(list(data.values()))
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    offset = (y_max - y_min) * 0.1
    for lbl in labels:
        med = np.nanmedian(data[lbl])
        fig.add_annotation(
            x=lbl, y=med+offset,
            text=f"{med:.2f}",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="black", weight="bold", size=11)
        )

    fig.update_layout(
        violinmode="group",
        template="plotly_white",
        title=dict(text=title, x=0.5) or "",
        width=width, height=height,
        xaxis=dict(title=x_title or "", showgrid=True),
        yaxis=dict(title=y_title or "", showgrid=True),
        showlegend=showlegend,
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    return fig


@log_time("Plotting PCA")
def plot_pca_2d(
    adata: AnnData,
    pc: tuple[int,int] = (1,2),
    color_key: str = "CONDITION",
    colors: dict[str,str] = None,
    title: str = "PCA",
    width: int = 900,
    height: int = 500,
    annotate: bool = False,
) -> go.Figure:
    """
    2D PCA scatter of samples, colored by adata.obs[color_key].
    """
    # ensure PCA is present
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata)
    # build DataFrame
    pcs = adata.obsm["X_pca"][:, [pc[0]-1, pc[1]-1]]
    df = pd.DataFrame(pcs, columns=[f"PC{pc[0]}", f"PC{pc[1]}"],
                      index=adata.obs_names)
    df[color_key] = adata.obs[color_key].values

    # pick colors
    levels = df[color_key].unique().tolist()
    palette = get_color_map(levels, px.colors.qualitative.Plotly)

    fig = go.Figure()
    for lv in levels:
        sub = df[df[color_key] == lv]
        fig.add_trace(go.Scatter(
            x=sub[f"PC{pc[0]}"],
            y=sub[f"PC{pc[1]}"],
            mode="markers+text" if annotate else "markers",
            text=sub.index.to_list(),
            textposition="top center",
            name=lv,
            marker=dict(color=palette[lv], size=8, line=dict(width=1, color="black")),
            hovertemplate = (f"Sample: %{{text}}")
        ))
    # axis labels with explained variance
    var = adata.uns["pca"]["variance_ratio"]
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width, height=height,
        template="plotly_white",
        xaxis=dict(title=f"PC{pc[0]} ({var[pc[0]-1]*100:.1f}% var)"),
        yaxis=dict(title=f"PC{pc[1]} ({var[pc[1]-1]*100:.1f}% var)"),
    )
    fig.update_xaxes(showline=True, mirror=True, linecolor="black")
    fig.update_yaxes(showline=True, mirror=True, linecolor="black")
    return fig


@log_time("UMAP")
def plot_umap_2d(
    adata: AnnData,
    color_key: str = "CONDITION",
    colors: dict[str,str] = None,
    title: str = "UMAP",
    width: int = 900,
    height: int = 500,
    annotate: bool = False,
) -> go.Figure:
    """
    2D UMAP scatter of samples, colored by adata.obs[color_key].
    """
    # ensure neighbors + UMAP are present
    if "X_umap" not in adata.obsm:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    df = pd.DataFrame(
        adata.obsm["X_umap"][:, :2],
        columns=["UMAP1","UMAP2"],
        index=adata.obs_names
    )
    df[color_key] = adata.obs[color_key].values

    levels = df[color_key].unique().tolist()
    palette = get_color_map(levels, px.colors.qualitative.Plotly)

    fig = go.Figure()
    for lv in levels:
        sub = df[df[color_key] == lv]
        fig.add_trace(go.Scatter(
            x=sub["UMAP1"],
            y=sub["UMAP2"],
            mode="markers+text" if annotate else "markers",
            text=sub.index.to_list(),
            textposition="top center",
            name=lv,
            marker=dict(color=palette[lv], size=8, line=dict(width=1, color="black")),
            hovertemplate="Sample: %{text}")
        )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=width, height=height,
        template="plotly_white",
        xaxis=dict(title="UMAP1"),
        yaxis=dict(title="UMAP2"),
    )
    fig.update_xaxes(showline=True, mirror=True, linecolor="black")
    fig.update_yaxes(showline=True, mirror=True, linecolor="black")
    return fig

# tried to use this for caching, doesn't work !
def _compute_orders_and_dendros(
    mat_bytes: bytes,
    shape: tuple,
    row_labels: tuple,
    col_labels: tuple,
    method: str,
    metric: str
):
    """
    mat_bytes : raw bytes of your data matrix, dtype float64
    shape     : (n_rows, n_cols) of that matrix
    """
    n_rows, n_cols = shape
    arr = np.frombuffer(mat_bytes, dtype=np.float64).reshape(n_rows, n_cols)

    # SciPy linkage exactly once
    row_link = sch.linkage(arr,       method=method, metric=metric)
    col_link = sch.linkage(arr.T,     method=method, metric=metric)

    # Produce the two dendrogram objects
    dcol = ff.create_dendrogram(
        arr.T,
        orientation="bottom",
        labels=list(col_labels),       # ← pass your sample names
        linkagefun=lambda _: col_link
    )
    # tag the col‐dendrogram traces so they draw in the right subplot
    for tr in dcol.data:
        tr["yaxis"] = "y2"

    drow = ff.create_dendrogram(
        arr,
        orientation="right",
        labels=list(row_labels),       # ← pass your protein IDs
        linkagefun=lambda _: row_link
    )
    # similarly tag the row‐dendrogram
    for tr in drow.data:
        tr["xaxis"] = "x2"

    return dcol, drow

def plot_cluster_heatmap_plotly(
    data: pd.DataFrame,
    y_labels: Optional[pd.DataFrame] = None,
    cond_series: Optional[pd.Series] = None,
    method: str = "ward",
    metric: str = "euclidean",
    colorscale: str = "vlag",
    width: int = 800,
    height: int = 800,
    title: str = "Hierarchical Clustering Heatmap",
) -> go.Figure:
    """
    Produces a “clustergram” of sample–sample distances:
      - top & left dendrograms via ff.create_dendrogram
      - heatmap = distance matrix (pdist → squareform), reordered
      - no hoverinfo for speed
      - nice axis styling exactly like the Plotly example
    """
    # 1) Clean up: fill any NaNs, but keep all rows
    df = data

    # 2) Column dendrogram (samples)
    dendro_col = ff.create_dendrogram(
        df.values.T,
        orientation="bottom",
        labels=list(df.columns),
        linkagefun=lambda x: sch.linkage(x, method=method, metric=metric)
    )
    for trace in dendro_col.data:
        trace["yaxis"] = "y2"

    # 3) Row dendrogram (proteins)
    dendro_row = ff.create_dendrogram(
        df.values,
        orientation="right",
        labels=list(df.index),
        linkagefun=lambda x: sch.linkage(x, method=method, metric=metric)
    )
    for trace in dendro_row.data:
        trace["xaxis"] = "x2"
        dendro_col.add_trace(trace)

    fig = dendro_col  # start from the col‐dendrogram figure

    # 4) Extract the leaf order by label
    col_leaves = fig.layout["xaxis"]["ticktext"]
    row_leaves = dendro_row.layout["yaxis"]["ticktext"]
    col_order = [df.columns.get_loc(lbl) for lbl in col_leaves]
    row_order = [df.index.get_loc(lbl)   for lbl in row_leaves]

    # 5) Reorder the DataFrame
    df = df.iloc[row_order, :].iloc[:, col_order]

    # 6) Add the heatmap of the raw (or z-scored) values
    min_val, max_val = np.nanmin(df.values), np.nanmax(df.values)
    abs_max = max(abs(min_val), abs(max_val))
    zmin, zmax = -abs_max, abs_max

    heat = go.Heatmap(
        z=df.values,
        x=fig.layout.xaxis["tickvals"],
        y=dendro_row.layout.yaxis["tickvals"],
        colorscale=colorscale,
        reversescale=True,
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        #hoverinfo="none",
        hoverinfo="text",
        hovertemplate=
            "Protein: %{y}<br>"
            "Sample: %{x}<br>"
            "Value: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Value"),
    )
    fig.add_trace(heat)

    # gene names
    if y_labels is not None:
        # get the protein IDs in the clustered order
        row_leaves = dendro_row.layout["yaxis"]["ticktext"]
        # map each protein ID → gene name
        ticktexts = [y_labels[df.index.get_loc(pid)] for pid in row_leaves]
        # overwrite the main y-axis (heatmap) ticks
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=dendro_row.layout["yaxis"]["tickvals"],
                ticktext=ticktexts
            )
        )

    levels  = pd.Categorical(cond_series).categories
    cmap    = get_color_map(levels, px.colors.qualitative.Plotly)
    colours = [cmap[cond_series[s]] for s in df.columns]

    # inject coloured tick labels via simple HTML <span>
    fig.update_xaxes(
        tickvals = fig.layout.xaxis.tickvals,
        ticktext = [
            f"<span style='color:{col};'>{lbl}</span>"
            for lbl,col in zip(df.columns, colours)
        ]
    )

    for lvl in levels:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=cmap[lvl], size=10),
            name=str(lvl),
            showlegend=True,
            hoverinfo="none"
        ))

    for tr in fig.data:
        # only show legend for our dummy condition traces
        if not (tr.type == "scatter" and tr.name in levels):
            tr.showlegend = False

    # position the legend to the right
    fig.update_layout(legend=dict(
        title="Condition",
        orientation="v",
        x=-0.0, y=1,
        xanchor="left", yanchor="top"
    ))

    # 7) Tidy up layout
    fig.update_layout({
        "width": width, "height": height,
        "showlegend": True,
        "hovermode": "closest",
        "title": {"text": title, "x": 0.5},
    })
    fig.update_layout(xaxis={"domain":[0.15,1], "mirror":False,
                             "showgrid":False, "showline":False,
                             "zeroline":False, "ticks":""})
    fig.update_layout(xaxis2={"domain":[0,0.15], "mirror":False,
                              "showgrid":False, "showline":False,
                              "zeroline":False, "showticklabels":False,
                              "ticks":""})
    fig.update_layout(yaxis={"domain":[0,0.85], "mirror":False,
                             "showgrid":False, "showline":False,
                             "zeroline":False, "showticklabels":False,
                             "ticks":""})
    fig.update_layout(yaxis2={"domain":[0.825,0.975], "mirror":False,
                              "showgrid":False, "showline":False,
                              "zeroline":False, "showticklabels":False,
                              "ticks":""})

    return fig

def plot_histogram_plotly(
    df: Optional[pd.DataFrame] = None,
    value_col: str = "Intensity",
    group_col: str = "Normalization",
    labels: List[str] = ["Before", "After"],
    colors: List[str] = ["blue", "red"],
    nbins: int = 50,
    stat: str = "probability",           # "count" or "probability"
    log_x: bool = True,                  # whether to log-transform
    log_base: int = 10,                   # 2 or 10
    opacity: Union[float, Sequence[float]] = 0.5,
    x_range: Optional[Tuple[float,float]] = None,
    y_range: Optional[Tuple[float,float]] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    width: int = 900,
    height: int = 500,
    title: Optional[str] = None,
    data: Optional[Dict[str, Sequence]] = None,
) -> go.Figure:
    """
    Generic overlaid histogram for one or more distributions.
    Supports optional log-base-2 or log-base-10 transform with correct tick labels.
    """
    # ------------------------------------------------
    # 1) Prepare raw arrays
    # ------------------------------------------------
    if data is not None:
        labels = list(data.keys())
        values = {lbl: np.asarray(data[lbl]) for lbl in labels}
    else:
        if df is None:
            raise ValueError("Need either 'data' or 'df'")
        df2 = df.copy()
        if log_x:
            df2 = df2[df2[value_col] > 0]
            df2["_val"] = np.log(df2[value_col]) / np.log(log_base)
        else:
            df2["_val"] = df2[value_col]
        values = {
            lbl: df2.loc[df2[group_col] == lbl, "_val"].values
            for lbl in labels
        }

    # ------------------------------------------------
    # 2) Compute bins on the transformed scale
    # ------------------------------------------------
    all_vals = np.concatenate(list(values.values()))
    mn = x_range[0] if (log_x and x_range) else (np.min(all_vals))
    mx = x_range[1] if (log_x and x_range) else (np.max(all_vals))
    bins = np.linspace(mn, mx, nbins + 1)

    # ------------------------------------------------
    # 3) Draw overlaid bars
    # ------------------------------------------------
    fig = go.Figure()
    ops = (list(opacity) if isinstance(opacity, (list,tuple,np.ndarray))
           else [opacity]*len(labels))

    for i, lbl in enumerate(labels):
        arr = values[lbl]
        counts, edges = np.histogram(arr, bins=bins)
        if stat == "probability":
            counts = counts / counts.sum()
        mids   = 0.5*(edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        fig.add_trace(go.Bar(
            x=mids, y=counts, width=widths,
            name=lbl,
            marker_color=colors[i] if i < len(colors) else None,
            opacity=ops[i],
            hoverinfo="skip",
        ))

    # ------------------------------------------------
    # 4) Axis formatting
    # ------------------------------------------------
    if log_x:
        # tick positions are integers on the log-scale
        lo = int(np.floor(bins[0]))
        hi = int(np.ceil (bins[-1]))
        units = list(range(lo, hi+1))
        # labels like 2ⁿ or 10ⁿ
        if log_base == 10:
            ticktext = [f"10<sup>{u}</sup>" for u in units]
            xlabel   = x_title or f"log₁₀({value_col})"
        else:
            ticktext = [f"2<sup>{u}</sup>"  for u in units]
            xlabel   = x_title or f"log₂({value_col})"
        fig.update_xaxes(
            type="linear",
            autorange=False,
            range=[lo, hi],
            tickmode="array",
            tickvals=units,
            ticktext=ticktext,
            showgrid=True,
            title_text=xlabel,
        )
        fig.update_yaxes(
            title_text=y_title or stat.title(),
            showgrid=True,
        )
    else:
        fig.update_xaxes(
            range=x_range,
            title_text=x_title or value_col,
            showgrid=True,
        )
        fig.update_yaxes(
            range=y_range,
            title_text=y_title or stat.title(),
            showgrid=True,
        )

    # ------------------------------------------------
    # 5) Final layout
    # ------------------------------------------------
    fig.update_layout(
        title=dict(text=title or "", x=0.5),
        barmode="overlay",
        template="plotly_white",
        width=width, height=height,
    )

    return fig

def add_violin_traces(
    fig: go.Figure,
    df,
    x: str,
    y: str,
    color: str,
    name_suffix: str,
    row: int,
    col: int,
    showlegend: bool = False,
    opacity: float = 0.7,
    width: float = 0.8,
) -> None:
    """
    Add violin traces to the given (row, col) subplot of a Figure.

    Parameters
    ----------
    fig : go.Figure
        Figure created with make_subplots.
    df : pd.DataFrame
        Long-format DataFrame containing columns [x, y].
    x : str
        Column name for categorical axis.
    y : str
        Column name for numeric values.
    color : str
        Color for the violin outline.
    name_suffix : str
        Legend group identifier (if showlegend=True).
    row : int
        Subplot row index (1-based).
    col : int
        Subplot column index (1-based).
    showlegend : bool
        Whether to add a legend entry for these traces.
    opacity : float
        Fill opacity of the violin.
    width : float
        Width of each violin in category units.
    """
    cats = df[x].unique()
    for cat in cats:
        sub = df[df[x] == cat]
        fig.add_trace(
            go.Violin(
                x=sub[x],
                y=sub[y],
                showlegend=showlegend,
                legendgroup=name_suffix,
                scalegroup=cat,
                line_color=color,
                opacity=opacity,
                width=width,

                # internal quartile+median lines only (transparent box)
                box_visible=True,
                box_fillcolor="rgba(0,0,0,0)",
                box_line_color="black",
                box_line_width=1,
                meanline_visible=False,

                # no outlier markers
                points=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

