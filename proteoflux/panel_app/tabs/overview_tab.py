import panel as pn
from proteoflux.panel_app.session_state import SessionState
from proteoflux.panel_app.components.overview_plots import (
    plot_barplot_proteins_per_sample,
    plot_violin_cv_rmad_per_condition,
    plot_h_clustering_heatmap,
    plot_volcanoes_wrapper
)
from proteoflux.panel_app.components.plot_utils import plot_pca_2d, plot_umap_2d
from proteoflux.panel_app.components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from proteoflux.utils.utils import logger, log_time

pn.extension("plotly")


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
    #pca_pane = pn.pane.Plotly(plot_pca_2d(state.adata),
    #                          height=500,
    #                          sizing_mode="stretch_width")
    #umap_pane = pn.pane.Plotly(plot_umap_2d(state.adata),
    #                           height=600,
    #                           sizing_mode="stretch_width")

    #h_clustering_pane = pn.pane.Plotly(plot_h_clustering_heatmap(im),
    #                                   height=800,
    #                                   sizing_mode="stretch_width")

    ## Volcanoes
    contrasts = state.adata.uns["contrast_names"]

    contrast_sel = pn.widgets.Select(
        name="Contrast",
        options=contrasts,
        value=contrasts[0],
    )
    show_measured = pn.widgets.Toggle(name="Observed in Both", button_type="default", button_style="outline", value=True)
    show_imp_cond1 = pn.widgets.Toggle(name=f"▲ Fully Imputed in {contrasts[0].split('_vs_')[0]}", button_type="default", button_style="outline", value=True)
    show_imp_cond2 = pn.widgets.Toggle(name=f"▲ Fully Imputed in {contrasts[0].split('_vs_')[1]}", button_type="default", button_style="outline", value=True)

    volcano_dmap = pn.bind(
        plot_volcanoes_wrapper,
        state=state,
        contrast=contrast_sel,
        show_measured=show_measured,
        show_imp_cond1=show_imp_cond1,
        show_imp_cond2=show_imp_cond2,
        sign_threshold=0.05,
        width=900,
        height=600,
    )

    # 3) assemble into a layout, no legend‐based toggles
    volcano_pane = pn.Column(
        pn.Row(
            contrast_sel,
            pn.Spacer(width=160),
            pn.Row(
                show_measured,
                show_imp_cond1,
                show_imp_cond2,
                margin=(20,0,0,0),
            ),
            sizing_mode="fixed",
            width=300,
            height=160,
        ),
        pn.panel(volcano_dmap,
                 width=900,
                 height=800,
                 margin=(-50, 0, 0, 0)),
        width=1200,
        sizing_mode="stretch_width"
    )
    #volcano_pane = pn.Column(
    #    pn.Row(contrast_sel, show_measured, show_imp_cond1, show_imp_cond2, sizing_mode="stretch_width"),
    #    pn.panel(volcano_dmap, sizing_mode="stretch_width", height=600),
    #    sizing_mode="fixed"
    #)

#    volcano_fig  = plot_volcanoes_wrapper(state, sign_threshold=0.05)
#    volcano_pane = pn.pane.Plotly(
#        volcano_fig,
#        height=600,
#        sizing_mode="stretch_width"
#    )


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
        #pn.Row(pca_pane, umap_pane, sizing_mode="stretch_width"),
        #h_clustering_pane,
        pn.pane.Markdown("##   Volcano plots"),
        volcano_pane,

        sizing_mode="stretch_both",
    )

    return layout
