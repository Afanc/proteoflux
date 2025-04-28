import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from skmisc.loess import loess
from sklearn.linear_model import RANSACRegressor
from typing import Optional, Union, List, Tuple, Dict
import missingno as msno

def plot_histogram(
    data: pd.DataFrame,
    pdf: PdfPages,
    value_col: str = "Intensity",
    group_col: str = "Normalization",
    labels: List[str] = ["Before", "After"],
    colors: List[str] = ["blue", "red"],
    title: str = "Distribution of Intensities",
    log_scale: bool = True,
    stat: str = "probability"
) -> None:
    """
    Plot histogram of intensity distributions grouped by a categorical column (default: 'Normalization').
    """
    plt.figure(figsize=(12, 6))
    min_intensity = max(1e-10, data[value_col].min())
    max_intensity = data[value_col].max()
    bins = np.logspace(np.log10(min_intensity), np.log10(max_intensity), num=50)

    for label, color in zip(labels, colors):
        sns.histplot(
            data=data[data[group_col] == label],
            x=value_col,
            bins=bins,
            stat=stat,
            multiple="dodge",
            #element="step",
            color=color,
            label=label,
            alpha=0.5
        )

    plt.legend(title=group_col)
    plt.title(title)
    if log_scale:
        plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    pdf.savefig()
    plt.close()

def plot_violin_on_axis(
    ax: Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    title: str,
    xlabel: str,
    ylabel: str,
    palette: dict = {"Before": "blue", "After": "red"},
    draw_median_line: bool = True,
    density_norm: str = "count",
    xtick_rotation: Optional[int] = None,
    xtick_fontsize: int = 8,
) -> None:
    """
    Plots a violin plot on the provided axis using the specified parameters.
    """
    sns.violinplot(
        x=x, y=y, hue=hue, data=data,
        inner="quart", dodge=True, alpha=0.5,
        split=False,
        palette=palette,
        ax=ax,
        density_norm=density_norm
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_median_line:
        median_val = np.median(data[y])
        ax.axhline(y=median_val, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)

    if hue:
        ax.legend(title=hue, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    if xtick_rotation is not None:
        ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=xtick_fontsize)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

def plot_aggregated_violin(
    ax: Axes,
    data: pd.DataFrame,
    x: str = "Condition",
    y: str = "Value",
    hue: str = "Normalization",
    title: str = "",
    xlabel: str = None,
    ylabel: str = None,
    palette: dict = {"Before": "blue", "After": "red"},
    density_norm: str = "width",
    log_scale: bool = False,
) -> None:
    """
    General helper to plot aggregated violin plots from a DataFrame that contains:
      - 'Condition'
      - 'Value'
      - 'Normalization'

    Parameters:
        ax: The matplotlib axis on which to plot.
        data: The long-format DataFrame containing the data.
        title: The title for the plot.
        ylabel: The label for the y-axis.
        density_norm: The density normalization setting (e.g., "width" or "area").
    """

    sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        ax=ax,
        inner="quart",
        dodge=True,
        density_norm=density_norm,
        split=False,
        palette=palette,
        alpha=0.5,
    )

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.legend(title=hue, loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

def plot_MA(
    ax: Axes,
    M: np.ndarray,
    A: np.ndarray,
    label: str = "Sample",
    reg_label: str = "Sample",
    color: str = "black",
    model: Optional[Union[loess, RANSACRegressor]] = None,
    span: float = 0.9,
    regression_type: Optional[str] = None,
    refit: bool = False,
    random_state: Optional[int] = 42,
    title: Optional[str] = None,
    xlabel: Optional[str] = "M (Mean Intensity)",
    ylabel: Optional[str] = "A (Log-ratio or Difference)",
    show_legend: bool = True,
    ylim: tuple = (-1.5, 1.5),
) -> None:
    """
    Plot an MA scatter plot with an optional regression line.
    """

    mask = ~np.isnan(A)
    M_clean = M[mask]
    A_clean = A[mask]

    ax.scatter(M_clean, A_clean, color=color, alpha=0.2, label=label, s=8, zorder=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylim(*ylim)

    x_vals, y_vals = None, None
    if regression_type:
        if refit:
            if regression_type == "loess":
                loess_fit = loess(M_clean, A_clean, span=span)
                loess_fit.fit()
                x_vals = M_clean
                y_vals = loess_fit.outputs.fitted_values
            elif regression_type == "linear":
                reg = RANSACRegressor(random_state=random_state)
                reg.fit(M_clean.reshape(-1, 1), A_clean)
                x_vals = M_clean
                y_vals = reg.predict(M_clean.reshape(-1, 1))
            else:
                raise ValueError("Invalid regression_type. Choose 'loess' or 'linear'.")
        else:
            if isinstance(model, loess):
                x_vals = model.inputs.x
                y_vals = model.outputs.fitted_values
            elif isinstance(model, RANSACRegressor):
                x_vals = M_clean
                y_vals = model.predict(M_clean.reshape(-1, 1))

        if x_vals is not None and y_vals is not None and x_vals.shape == y_vals.shape:
            sorted_idx = np.argsort(x_vals)
            ax.plot(
                x_vals[sorted_idx],
                y_vals[sorted_idx],
                linestyle='--',
                color=color,
                linewidth=1.5,
                zorder=3,
                label=f"{regression_type.capitalize()} fit ({reg_label})"
            )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend(loc='upper right')

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

def plot_bar_on_axis(
    ax: Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    palette: Optional[Union[str, List[str], dict]] = "Blues_d",
    xtick_rotation: Optional[int] = None,
    xtick_fontsize: int = 8,
    log_scale: bool = False,
    draw_grid: bool = True,
    annotate_values: bool = False,
) -> None:
    """
    Plots a barplot on the provided axis using Seaborn and consistent formatting.

    Parameters:
        ax (Axes): Matplotlib axis to plot on.
        data (pd.DataFrame): DataFrame with data to plot.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis (bar height).
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        palette: Color palette (optional).
        xtick_rotation (int): Rotate x-tick labels if specified.
        xtick_fontsize (int): Font size for x-tick labels.
        draw_grid (bool): Whether to draw grid on the plot.
    """
    sns.barplot(x=x,
                y=y,
                data=data,
                hue=x,
                palette=palette,
                ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    if log_scale:
        ax.set_yscale("log")

    if xtick_rotation is not None:
        ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=xtick_fontsize)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    if annotate_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color='black'
                    )


def plot_missing_corr_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    fontsize: int = 8,
    title: str = "Missing Values Correlation Heatmap",
    adjust_params: Optional[Dict[str, float]] = None,
    suptitle: Optional[str] = None,
    additional_title: Optional[str] = None
):
    """
    Plots a correlation heatmap of missing values using msno.heatmap.

    Args:
        data (pd.DataFrame): DataFrame containing missing values.
        figsize (tuple): Figure size.
        fontsize (int): Font size for the plot.
        title (str): Title for the heatmap.
        adjust_params (dict, optional): Parameters for plt.subplots_adjust.
        suptitle (str, optional): Overall figure title.
        additional_title (str, optional): Additional title details.
    """
    plt.figure(figsize=figsize)
    msno.heatmap(data, figsize=figsize, fontsize=fontsize)
    if adjust_params is None:
        adjust_params = {"left": 0.25, "bottom": 0.25, "top": 0.85}
    plt.subplots_adjust(**adjust_params)
    plt.suptitle(suptitle if suptitle is not None else title, fontsize=20, y=0.955)
    if additional_title:
        plt.title(additional_title, fontsize=10, y=1.05)
    else:
        plt.title("Nullity correlation: -1 (presence guarantees absence), 0 (independent), +1 (presence guarantees presence)", fontsize=10, y=1.05)


def plot_cluster_heatmap(
    data: pd.DataFrame,
    col_colors: Optional[pd.Series] = None,
    title: str = "Heatmap of Missing Values",
    figsize: Tuple[int, int] = (12, 10),
    row_cluster: bool = False,
    col_cluster: bool = True,
    xticklabels: bool = True,
    yticklabels: bool = False,
    cbar_ticks: Optional[List[float]] = None,
    cbar_aspect: int = 10,
    adjust_params: Optional[Dict[str, float]] = None,
    legend_title: str = "Conditions",
    legend_loc: str = "upper right",
    legend_fontsize: int = 8,
    legend_bbox: Tuple[float, float] = (1.15, 0.95),
    binary_labels: Optional[Tuple[str, str]] = ["Present", "Missing"],
    cmap: Optional[LinearSegmentedColormap] = None,
    legend_mapping: Optional[Dict[str, any]] = None
):
    """
    Generates a clustermap heatmap for missing values with hierarchical clustering.

    Args:
        data (pd.DataFrame): DataFrame of binary values (e.g., 1 for missing, 0 for present).
        col_colors (pd.Series, optional): Series mapping column names to colors.
        title (str): Title for the clustermap.
        figsize (tuple): Figure size.
        row_cluster (bool): Whether to cluster rows.
        col_cluster (bool): Whether to cluster columns.
        xticklabels (bool): Show x-axis tick labels.
        yticklabels (bool): Show y-axis tick labels.
        cbar_ticks (list, optional): Ticks for the colorbar.
        cbar_aspect (int): Aspect ratio for the colorbar.
        adjust_params (dict, optional): Parameters to pass to plt.subplots_adjust.
        legend_title (str): Title for the legend.
        legend_loc (str): Legend location.
        legend_fontsize (int): Font size for legend text.
        legend_bbox (tuple): bbox_to_anchor for the legend.
        binary_labels (list): labels to the binary cmap.
        cmap (LinearSegmentedColormap, optional): Colormap to use. If None, a binary colormap is used.
        legend_mapping (dict, optional): Mapping from condition names to colors for the legend.

    Returns:
        sns.matrix.ClusterGrid: The clustermap object.
    """
    if cbar_ticks is None:
        cbar_ticks = [0.25, 0.75]
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list('CustomRed', ['#f0f0f0', '#e34a33'], 2)
    if adjust_params is None:
        adjust_params = {"right": 0.85, "bottom": 0.25, "top": 0.90}

    g = sns.clustermap(
        data,
        cmap=cmap,
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        col_colors=col_colors,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"ticks": cbar_ticks, "aspect": cbar_aspect},
    )

    plt.subplots_adjust(**adjust_params)
    colorbar = g.cax
    colorbar.set_position([.05, .4, .02, .3])
    if binary_labels is not None:
        colorbar.set_yticklabels(binary_labels)
    #colorbar.set_yticklabels(binary_labels)

    # If a legend mapping is provided, create a legend accordingly.
    if legend_title and legend_mapping is not None:
        legend_patches = [mpatches.Patch(color=color, label=cond)
                          for cond, color in legend_mapping.items()]
        g.ax_heatmap.legend(
            handles=legend_patches,
            title=legend_title,
            loc=legend_loc,
            fontsize=legend_fontsize,
            bbox_to_anchor=legend_bbox
        )

    g.fig.suptitle(title, fontsize=20, y=0.955)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    return g


def plot_regression_scatter(
    true_vals: np.ndarray,
    imputed_vals: np.ndarray,
    results_df: pd.DataFrame,
    orig_imputed_mask: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    scatter_kwargs: Optional[dict] = None,
    line_color: str = "red",
    line_kwargs: Optional[dict] = None,
    xlabel: str = "True Values",
    ylabel: str = "Imputed Values",
    title: str = "Imputation Regression Plot"
) -> plt.Figure:
    """
    Create a scatter plot of imputed vs. true values with density coloring,
    annotate with cross-validation metrics, and return the Figure.

    Args:
        true_vals (np.ndarray): Array of true values.
        imputed_vals (np.ndarray): Array of imputed values.
        results_df (pd.DataFrame): DataFrame containing metrics ('r2' and 'rmae').
        figsize (tuple, optional): Figure size. Defaults to (8, 8).
        cmap (str, optional): Colormap for density. Defaults to "viridis".
        scatter_kwargs (dict, optional): Additional kwargs for plt.scatter.
        line_color (str, optional): Color for the y=x reference line. Defaults to "red".
        line_kwargs (dict, optional): Additional kwargs for the reference line.
        xlabel (str, optional): Label for x-axis. Defaults to "True Values".
        ylabel (str, optional): Label for y-axis. Defaults to "Imputed Values".
        title (str, optional): Plot title. Defaults to "Imputation Regression Plot".

    Returns:
        plt.Figure: The generated Figure.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {"s": 10, "alpha": 0.5}
    if line_kwargs is None:
        line_kwargs = {"lw": 1.5, "linestyle": "--"}

    # Compute density
    xy = np.vstack([true_vals, imputed_vals])
    density = gaussian_kde(xy)(xy)

    # Compute metrics
    r2_mean, r2_std = results_df["r2"].mean(), results_df["r2"].std()
    rmae_mean, rmae_std = results_df["rmae"].mean(), results_df["rmae"].std()

    if scatter_kwargs is None:
        scatter_kwargs = {"s": 10, "alpha": 0.5}
    if line_kwargs is None:
        line_kwargs = {"lw": 1.5, "linestyle": "--"}

    fig, ax = plt.subplots(figsize=figsize)

    scatter = None

    if orig_imputed_mask is not None:
        # plot regular points
        mask_test = ~orig_imputed_mask
        scatter = ax.scatter(
            true_vals[mask_test],
            imputed_vals[mask_test],
            c=density[mask_test],
            cmap=cmap,
            marker='o',
            **scatter_kwargs
        )
        # plot originally imputed points
        ax.scatter(
            true_vals[orig_imputed_mask],
            imputed_vals[orig_imputed_mask],
            marker='^',
            color="red",
            edgecolor="black",
            linewidth=0.5,
            **scatter_kwargs
        )

        # add manual legend for datapoints
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Test MV', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Original Imputed', markerfacecolor='red', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    else:
        # if no mask given, plot normally
        scatter = ax.scatter(
            true_vals,
            imputed_vals,
            c=density,
            cmap=cmap,
            marker='o',
            **scatter_kwargs
        )

#
#    # Compute density using gaussian_kde
#    xy = np.vstack([true_vals, imputed_vals])
#    density = gaussian_kde(xy)(xy)
#
#    # Compute metrics
#    r2_mean, r2_std = results_df["r2"].mean(), results_df["r2"].std()
#    rmae_mean, rmae_std = results_df["rmae"].mean(), results_df["rmae"].std()
#
#    # Create figure and axis
#    fig, ax = plt.subplots(figsize=figsize)
#
#    # Scatter plot with density coloring
#    scatter = ax.scatter(true_vals, imputed_vals, c=density, cmap=cmap, **scatter_kwargs)
#    ax.plot(
#        [true_vals.min(), true_vals.max()],
#        [true_vals.min(), true_vals.max()],
#        color=line_color,
#        **line_kwargs
#    )

    # Add colorbar for density
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Density")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate with metrics
    metrics_text = (
        f"$R^2$: {r2_mean:.2f} ± {1.96 * r2_std:.2f}\n"
        f"RMAE: {rmae_mean:.2%} ± {1.96 * rmae_std:.2%}"
    )
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5")
    )

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    return fig

def get_color_palette(
    palette_name: str = "tab20",
    n: int = 0,
):
    """
    Generates a sns color palette for clustermap heatmap for missing values with hierarchical clustering.

    Args:
        palette_name (str): Name of the color palette
        n (int): number of colors in the palette
    """

    return sns.color_palette(palette_name, n)
