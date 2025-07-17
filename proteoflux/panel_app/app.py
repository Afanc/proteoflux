import logging, sys

# initialize Python logging for the Bokeh server’s main process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger().setLevel(logging.INFO)

import panel as pn
from proteoflux.panel_app.session_state import SessionState
from proteoflux.panel_app.tabs.overview_tab import overview_tab
from proteoflux.panel_app.tabs.normalization_tab import normalization_tab
#from .tabs.imputation_tab import imputation_tab

state = SessionState.initialize("config.yaml")

pn.extension('plotly')

app = pn.Tabs(
    ("Overview", overview_tab(state)),
    ("Normalization", normalization_tab(state)),
    #("Imputation", imputation_tab(state)),
)

if __name__.startswith("bokeh"):    # when run via `panel serve`
    app.servable()
else:                                # when run as `python app.py`
    # export instead of serving
    app.save("proteoflux_preview_embed.html", embed=True)
