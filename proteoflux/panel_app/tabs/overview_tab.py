import panel as pn
from proteoflux.panel_app.session_state import SessionState
from proteoflux.panel_app.components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_h_clustering_heatmap
)
from proteoflux.panel_app.components.plot_utils import plot_pca_2d, plot_umap_2d
from proteoflux.panel_app.components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from proteoflux.utils.utils import logger, log_time

pn.extension("plotly")

# this doesn't do anything !
#pn.extension(
#    template="fast",
#    loading_spinner="dots",
#    loading_color="#00aa41",
#)

@log_time("Preparing Overview Tab")
def overview_tab(state: SessionState):
    """
    Overview Tab
     - Info on samples and filtering, config
     - Hists of Ids, CVs
     - PCAs, UMAP
     - Hierch. clustering
     - Volcanoes
    """
    # Plots
    im = state.intermediate_results

    ## IDs barplot
    hist_ID_fig = plot_barplot_proteins_per_sample(im)

    ## Metric Violins
    cv_fig, rmad_fig = plot_violin_cv_rmad_per_condition(im)
    rmad_pane = pn.pane.Plotly(rmad_fig, height=500, sizing_mode="stretch_width")
    cv_pane   = pn.pane.Plotly(cv_fig,  height=500, sizing_mode="stretch_width")

    ## UMAP and PCA
    pca_pane = pn.pane.Plotly(plot_pca_2d(state.adata),
                              height=500,
                              sizing_mode="stretch_width")
    umap_pane = pn.pane.Plotly(plot_umap_2d(state.adata),
                               height=600,
                               sizing_mode="stretch_width")

    ## hierch. clustering

    # this doesn't do anything !
    #def _make_cluster_plot():
    #    # this only runs when the pane actually needs to render
    #    return plot_h_clustering_heatmap(im)

    #h_clustering_pane = pn.panel(
    #    _make_cluster_plot,
    #    loading=True,            # show spinner while _make_cluster_plot runs
    #    height=800,
    #    sizing_mode="stretch_width"
    #)

    h_clustering_pane = pn.pane.Plotly(plot_h_clustering_heatmap(im),
                                       height=800,
                                       sizing_mode="stretch_width")

    ## Volcanoes

    # Texts
    intro_text = pn.pane.Markdown("some config text",
        width=1200,
        margin=(10, 0, 15, 0),
    )

    # Tab layout

    layout = pn.Column(
        pn.pane.Markdown("#   Summary of analysis"),
        intro_text,
        pn.pane.Markdown("##   Proteins Identified"),
        hist_ID_fig,
        pn.pane.Markdown("##   Metrics per Condition"),
        pn.Row(rmad_pane, cv_pane, sizing_mode="stretch_width"),
        pn.pane.Markdown("##   Clustering"),
        pn.Row(pca_pane, umap_pane, sizing_mode="stretch_width"),
        h_clustering_pane,
        pn.pane.Markdown("##   Volcano plots"),

        sizing_mode="stretch_both",
    )

    return layout
