import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from proteoflux.evaluation.evaluation_utils import prepare_long_df, aggregate_metrics
from proteoflux.panel_app.components.plot_utils import plot_stacked_proteins_by_category, plot_violins, compute_metric_by_condition, get_color_map, plot_cluster_heatmap_plotly
from proteoflux.panel_app.components.normalization_plots import plot_cv_by_condition #better, other place

from proteoflux.utils.utils import logger, log_time


@log_time("Plotting barplot proteins per sample")
def plot_barplot_proteins_per_sample(
    im,
    matrix_key: str = "normalized",
    bar_color: str = "teal",
    title: str = "Proteins Detected per Sample",
    width: int = 900,
    height: int = 500,
) -> go.Figure:
    """
    Count proteins per sample and draw as a bar plot using the generic helper.
    """
    # call the generic bar helper

    fig = plot_stacked_proteins_by_category(im)
    return fig

@log_time("Plotting violins metrics per sample")
def plot_violin_cv_rmad_per_condition(
    im,
    matrix_key: str = "normalized",
    title: str = "%CV / rMAD per Condition",
    width: int = 900,
    height: int = 800,
) -> list[go.Figure]:
    # 1) grab the raw matrix and condition array
    arr = im.matrices[matrix_key]            # (n_proteins x n_samples) numpy array
    samples = im.columns                     # list of sample names
    cond_s = (
        im.dfs["condition_pivot"]
          .to_pandas()
          .rename(columns={"CONDITION":"Condition"})
          .set_index("Sample")["Condition"]
    )
    cond_array = cond_s.reindex(samples).values  # aligned to arr’s columns

    # 2) build the intensity DataFrame and compute CV / rMAD metrics
    df_mat = pd.DataFrame(arr, columns=samples)
    cond_map = pd.Series(cond_array, index=samples)

    # Vectorized + safe computation
    cv_dict = compute_metric_by_condition(df_mat, cond_map, metric="CV")
    rmad_dict = compute_metric_by_condition(df_mat, cond_map, metric="rMAD")

    # 3) draw grouped violins

    labels = list(cv_dict.keys())
    color_map = get_color_map(labels,
                              palette=px.colors.qualitative.Plotly,
                              anchor="Total",
                              anchor_color="mediumblue")

    cv_fig = plot_violins(
                 data=cv_dict,
                 colors=color_map,
                 title="%CV per Condition",
                 width=width,
                 height=height,
                 x_title="Condition",
                 y_title="%CV",
                 showlegend=False,
                 )

    rmad_fig = plot_violins(
                   data=rmad_dict,
                   colors=color_map,
                   title="%rMAD per Condition",
                   width=width,
                   height=height,
                   x_title="Condition",
                   y_title="%rMAD",
                   showlegend=False,
                   )
    return [cv_fig, rmad_fig]


@log_time("Plotting Hierarchical Clustering")
def plot_h_clustering_heatmap(im):

    meta = im.dfs["protein_metadata"].to_pandas()
    df = pd.DataFrame(im.matrices["imputed"],
                      index=meta["INDEX"].tolist(),
                      columns=im.columns)

    df_z = df.sub(df.mean(axis=1), axis=0)

    gene_map = meta.set_index("INDEX")["GENE_NAMES"]
    y_labels = gene_map.reindex(df_z.index).tolist()

    cond_ser = (
        im.dfs["condition_pivot"]
          .to_pandas()
          .set_index("Sample")["CONDITION"]
    )

    fig = plot_cluster_heatmap_plotly(
        data=df_z,
        y_labels=y_labels,
        method="ward",
        metric="euclidean",
        colorscale="RdBu",
        cond_series=cond_ser.reindex(df_z.columns),
        title="Clustergram of All Samples"
    )
    return fig
