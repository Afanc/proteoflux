import pandas as pd
from typing import Tuple, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from proteoflux.evaluation.evaluation_utils import prepare_long_df, aggregate_metrics
from proteoflux.panel_app.components.plot_utils import plot_histogram_plotly, add_violin_traces
from proteoflux.utils.utils import logger, log_time

@log_time("Converting to long df")
def get_intensity_long_df(
    im,
    before_key: str = "postlog",
    after_key:  str = "normalized",
    cond_map:    Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build one long-format DataFrame combining Before/After intensities.

    Columns: [Sample, Intensity, Normalization, Condition].

    - im: IntermediateResults
    - before_key/after_key: keys into im.matrices
    - cond_map: optional pandas DataFrame with columns [Sample, Condition].
                If None, pulled from im.dfs["condition_pivot"].
    """
    cols = im.columns

    # 1) prepare condition mapping
    if cond_map is None:
        cond_map = (
            im.dfs["condition_pivot"]
              .to_pandas()
              .rename(columns={"Sample":"Sample", "CONDITION":"Condition"})
              .loc[:, ["Sample","Condition"]]
        )

    # 2) melt & tag Before
    df_before = prepare_long_df(
        df=pd.DataFrame(im.matrices[before_key], columns=cols),
        label="Before",
        label_col="Normalization",
        condition_mapping=cond_map
    )

    # 3) melt & tag After
    df_after = prepare_long_df(
        df=pd.DataFrame(im.matrices[after_key], columns=cols),
        label="After",
        label_col="Normalization",
        condition_mapping=cond_map
    )

    # 4) concatenate and drop any NA
    return pd.concat([df_before, df_after], ignore_index=True).dropna()


@log_time("Plotting violins before/after norm")
def plot_violin_by_group_go(
    df_long:       pd.DataFrame,
    group_col:     str,
    colors:        Tuple[str, str] = ("blue","red"),
    subplot_titles:Tuple[str, str] = ("Before","After"),
    title:         Optional[str] = None,
    width:         int = 900,
    height:        int = 500,
    opacity:       float = 0.7,
) -> go.Figure:
    """
    Generic side-by-side violin plot.

    - df_long: output of get_intensity_long_df()
    - group_col: which column to group on ("Condition" or "Sample")
    - colors: (before_color, after_color)
    - subplot_titles: titles for the two panels
    - title: overall figure title
    """
    df_b = df_long[df_long["Normalization"]=="Before"]
    df_a = df_long[df_long["Normalization"]=="After"]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        subplot_titles=list(subplot_titles),
        horizontal_spacing=0.08,
    )

    # add violins
    add_violin_traces(fig, df_b, x=group_col, y="Intensity",
                      color=colors[0], name_suffix="Before",
                      row=1, col=1, showlegend=False, opacity=opacity)
    add_violin_traces(fig, df_a, x=group_col, y="Intensity",
                      color=colors[1], name_suffix="After",
                      row=1, col=2, showlegend=False, opacity=opacity)

    # draw median line across each panel
    for col_id, df_sub in [(1, df_b), (2, df_a)]:
        med = df_sub["Intensity"].median()
        fig.add_hline(
            y=med,
            row=1, col=col_id,
            line=dict(color="gray", dash="dash", width=1),
            layer="below"
        )

    # add vertical grid‐lines at each category center
    cats = df_b[group_col].unique()
    for i, _ in enumerate(cats):
        for xref in ("x1", "x2"):
            fig.add_shape(
                type="line",
                xref=xref, yref="paper",
                x0=i, x1=i,
                y0=0, y1=1,
                line=dict(color="gray", dash="dot", width=1),
                layer="above",
            )

    # finalize layout
    fig.update_layout(
        title=dict(text=title, x=0.5) if title else {},
        width=width,
        height=height,
        template="plotly_white",
        violinmode="group",
        showlegend=False
    )
    # draw boxed axes
    for rid, cid in [(1,1),(1,2)]:
        fig.update_xaxes(showline=True, mirror=True, linecolor="black",
                         row=rid, col=cid, title_text=group_col)
        fig.update_yaxes(showline=True, mirror=True, linecolor="black",
                         row=rid, col=cid, title_text="Intensity")
    fig.update_yaxes(row=1, col=2, side="right")

    return fig

@log_time("Plotting histogram before/after log")
def plot_intensities_histogram(im) -> go.Figure:
    """
    Plot overlaid Before/After intensity histograms.
    """
    df = get_intensity_long_df(im,
                               before_key="raw_mat",
                               after_key="postlog",
                               cond_map=None)

    fig = plot_histogram_plotly(
        df=df,
        value_col="Intensity",
        group_col="Normalization",
        labels=["Before","After"],
        colors=["blue","red"],
        nbins=50,
        stat="probability",
        log_x=True,
        opacity=0.5,
        width=900,
        height=500,
        title="Distribution of intensities"
    )
    return fig

def plot_intensities_histogram_works(im) -> go.Figure:
    """
    Plot overlaid Before/After intensity histograms.
    """
    df = get_intensity_long_df(im,
                               before_key="raw_mat",
                               after_key="postlog",
                               cond_map=None)
    return plot_histogram_plotly(
        df=df,
        title="Distribution of intensities (before and after log-transformation)",
        value_col="Intensity",
        group_col="Normalization",
        labels=["Before","After"],
        colors=["blue","red"],
        nbins=50,
        stat="probability",
        log_x=True,
        opacity=0.5,
        width=900,
        height=500,
    )


@log_time("Plotting violins before/after by condition")
def plot_violin_intensity_by_condition(im) -> go.Figure:
    """
    Side-by-side violins grouped by Condition.
    """
    df = get_intensity_long_df(im, cond_map=None)
    return plot_violin_by_group_go(
        df_long=df,
        group_col="Condition",
        colors=("blue","red"),
        subplot_titles=("Before Normalization","After Normalization"),
        title="Distribution by Condition (log-transformed data)",
    )


@log_time("Plotting violins before/after by sample")
def plot_violin_intensity_by_sample(im) -> go.Figure:
    """
    Side-by-side violins grouped by Sample.
    """
    df = get_intensity_long_df(im, cond_map=None)
    return plot_violin_by_group_go(
        df_long=df,
        group_col="Sample",
        colors=("blue","red"),
        subplot_titles=("Before Normalization","After Normalization"),
        title="Distribution by Sample (log-transformed data)",
        width=max(1500, len(im.columns)*20),
        height=600,
    )

def plot_grouped_violin_metric_by_condition(
    im,
    metric: str,
    colors=('blue', 'red'),
    title=None,
    width=900,
    height=600,
) -> go.Figure:
    # 1) build the metric DataFrame
    cond_map = (
        im.dfs["condition_pivot"]
          .to_pandas()
          .rename(columns={"CONDITION": "Condition"})
          .loc[:, ["Sample", "Condition"]]
    )
    bdf = pd.DataFrame(im.matrices["postlog"],    columns=im.columns)
    adf = pd.DataFrame(im.matrices["normalized"], columns=im.columns)
    agg = aggregate_metrics(
        before_df=bdf, after_df=adf,
        condition_mapping=cond_map,
        metrics=[metric],
        label_col="Normalization",
        before_label="Before", after_label="After",
        condition_key="Condition",
        sample_key="Sample",
    )
    dfm = agg[metric]

    # 2) map each condition to an integer slot
    conditions = dfm["Condition"].unique()
    idx_map    = {cond:i for i,cond in enumerate(conditions)}
    offset     = 0.2

    # 3) build the violin traces
    fig = go.Figure()
    for norm, color in zip(["Before", "After"], colors):
        sub = dfm[dfm["Normalization"] == norm]
        # numeric x with ±offset
        x_vals = sub["Condition"].map(idx_map) + ( -offset if norm=="Before" else +offset )
        fig.add_trace(go.Violin(
            x=x_vals,
            y=sub["Value"],
            name=norm,
            legendgroup=norm,
            line_color=color,
            box_visible=True,
            box_line_color="black",
            box_line_width=1,
            opacity=0.6,
            width=offset * 1.8,   # half-width ≈ offset
            meanline_visible=True,
            points=False,
            hoverinfo="skip",
            spanmode="hard",
        ))


    for i in list(idx_map.values()):
        fig.add_vline(
            x=i,
            line=dict(color="gray", dash="dot", width=1),
            layer="above",
        )

    # 4) finalize layout with one tick per condition
    title_text=f"{metric} per Condition (Log-transformed data)"
    fig.update_layout(
        template="plotly_white",
        violinmode="overlay",  # works with our manual offsets
        title=dict(text=title_text, x=0.5),
        width=width,
        height=height,
        xaxis=dict(
            title="Condition",
            tickmode="array",
            tickvals=list(idx_map.values()),
            ticktext=conditions,
        ),
        yaxis=dict(title=metric, showgrid=True),
        showlegend=True,
    )

    # 6) draw boxed axes
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig

@log_time("Plotting violins cv by condition")
def plot_cv_by_condition(im) -> go.Figure:
    """
    Convenience wrapper to plot %CV per condition.
    """
    return plot_grouped_violin_metric_by_condition(
        im,
        metric="CV",
        colors=("blue","red"),
        title="Coefficient of Variation (CV) per Condition"
    )


@log_time("Plotting violins rmad by condition")
def plot_rmad_by_condition(im) -> go.Figure:
    """
    Convenience wrapper to plot RMAD per condition.
    """
    return plot_grouped_violin_metric_by_condition(
        im,
        metric="RMAD",
        colors=("blue","red"),
        title="Robust MAD (RMAD) per Condition"
    )

