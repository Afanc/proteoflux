import panel as pn
import yaml
from functools import lru_cache
from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.evaluation.normalization_evaluator import NormalizerPlotter
from proteoflux.export.differential_expression_plotter import DifferentialExpressionPlotter
from proteoflux.export.de_exporter import DEExporter

class SessionState:
    def __init__(self, config_path: str = "config.yaml"):
        # 1) Load full config (same as CLI init)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.session_loaded = False

        # placeholders
        self.dataset = None
        self.adata = None
        self.plotter = None
        self.exporter = None
        self.preprocess_results = None
        self.intermediate_results = None
        self.normalization_eval = None
        self.imputation_eval = None

    @staticmethod
    @pn.cache
    def initialize(cls, config_path: str = "config.yaml") -> "SessionState":
        inst = SessionState(config_path)
        inst._initialize_heavy()
        return inst

    def _initialize_heavy(self):
        if self.session_loaded:
            return self

        self.dataset = Dataset(**self.config)
        self.adata   = self.dataset.get_anndata()
        self.adata   = run_limma_pipeline(self.adata, self.config)
        self.preprocess_results    = self.dataset.preprocessed_data
        self.intermediate_results  = self.dataset.preprocessor.intermediate_results

        self.session_loaded = True

        return self

    def setup_analysis(self): #move to initialize later on
        """Phase 2: instantiate plotter & exporter."""
        analysis_cfg = self.config.get("analysis", {})
        self.plotter  = DifferentialExpressionPlotter(self.adata, analysis_cfg)
        self.exporter = DEExporter(
            self.adata,
            output_path=analysis_cfg["exports"]["path_table"],
            use_xlsx=analysis_cfg["exports"]["table_use_xlsx"],
            sig_threshold=analysis_cfg["sign_threshold"],
        )
        return self

    # not needed ?

    ## These two methods let you comment out later steps:
    #def export_plots(self):
    #    if self.config["analysis"]["exports"]["export_plot"]:
    #        self.plotter.plot_all()
    #    return self

    #def export_tables_and_h5ad(self):
    #    exp_cfg = self.config["analysis"]["exports"]
    #    if self.config["analysis"]["export_table"]:
    #        self.exporter.export()
    #    if exp_cfg["export_h5ad"]:
    #        self.exporter.export_adata(exp_cfg["path_h5ad"])
    #    return self
