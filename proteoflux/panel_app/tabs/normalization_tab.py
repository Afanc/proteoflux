import panel as pn
from proteoflux.panel_app.session_state import SessionState
from proteoflux.panel_app.components.normalization_plots import (
    plot_intensities_histogram,
    plot_violin_intensity_by_condition,
    plot_violin_intensity_by_sample,
    plot_cv_by_condition,
    plot_rmad_by_condition
    #create_ma_plot_for_sample,
)
from proteoflux.panel_app.components.texts import (
    intro_preprocessing_text,
    log_transform_text
)
from proteoflux.utils.utils import logger, log_time

pn.extension("plotly")

@log_time("Preparing Normalization Tab")
def normalization_tab(state: SessionState):
    """
    Panel layout for normalization overview:
     - Histogram of intensities
     - Violin by condition
     - Violin by sample
     - MA-plot selector
    """
    im = state.intermediate_results

    hist_fig    = plot_intensities_histogram(im)
    viol_cond   = plot_violin_intensity_by_condition(im)
    viol_sample = plot_violin_intensity_by_sample(im)

    # Create the toggle button
    toggle = pn.widgets.Toggle(name="▼ Why a log2 transformation?", button_type="light")

    # Texts
    intro_text = pn.pane.Markdown(intro_preprocessing_text,
        width=1800,
        margin=(10, 0, 15, 0),
    )
    tooltip_text = pn.pane.Markdown(log_transform_text,
        visible=False,
        margin=(10, 0, 0, 25),
    )

    # Update function to toggle visibility and update button label
    def show_hide_text(event):
        tooltip_text.visible = toggle.value
        toggle.name = "▲ Why a log2 transformation?" if toggle.value else "▼ Why a log2 transformation?"

    toggle.param.watch(show_hide_text, "value")


    # Your layout

    return pn.Column(
        pn.pane.Markdown("#   Preprocessing Overview"),
        intro_text,
        pn.pane.Markdown("##   Normalization"),
        pn.Column(toggle, tooltip_text, width=1600),
        pn.Row(hist_fig),
        pn.Row(viol_cond),#, viol_sample, sizing_mode="stretch_width"),
        pn.Row(viol_sample, sizing_mode="stretch_width"),
        pn.Row(plot_cv_by_condition(im), plot_rmad_by_condition(im), sizing_mode="stretch_width"),
        #pn.Row(sample_sel, ma_pane, sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )
