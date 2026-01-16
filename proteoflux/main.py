from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.limma_pipeline import run_limma_pipeline, clustering_pipeline
from proteoflux.export.pdf_report_exporter import ReportPlotter
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.utils import debug_protein_view, logger, log_time


@log_time("Proteoflux Pipeline")
def run_pipeline(config: dict):
    dataset = Dataset(**config)
    adata = dataset.get_anndata()
    adata = run_limma_pipeline(adata, config)

    analysis_config = config.get("analysis", {})
    max_features = analysis_config.get("clustering_max", 8000)

    adata = clustering_pipeline(adata, max_features=max_features)

    if analysis_config.get("export_plot", True):
        plotter = ReportPlotter(adata, config)
        plotter.plot_all()

    export_config = analysis_config.get("exports")
    if not export_config:
        raise ValueError("Missing config.analysis.exports block (required for export paths).")
    exporter = DEExporter(adata,
                          output_path=export_config.get("path_table"),
                          sig_threshold=analysis_config.get("sign_threshold"),
                          annotate_matrix=export_config.get("annotate_matrix"),
                          )

    exporter.export_adata(export_config.get("path_h5ad"))

    if analysis_config.get("export_table", True):
        exporter.export()
