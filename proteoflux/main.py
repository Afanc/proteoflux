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
    adata = clustering_pipeline(adata, config)

    analysis_config = config.get("analysis", {})
    export_config = config.get("exports") or analysis_config.get("exports") #backward comp
    if not export_config:
        raise ValueError(
            "Missing exports block. Expected config.exports (new) or config.analysis.exports (legacy)."
        )

    path_pdf = export_config.get("path_pdf_report")
    path_table = export_config.get("path_table")
    path_h5ad = export_config.get("path_h5ad")

    # --- PDF export
    if path_pdf:
        plotter = ReportPlotter(adata, config)
        plotter.plot_all()

    # --- H5AD export
    if not path_h5ad:
        raise ValueError("exports.path_h5ad must not be null.")

    exporter = DEExporter(
        adata,
        output_path=path_table,
        sig_threshold=analysis_config.get("sign_threshold"),
        config=config,
    )
    exporter.export_adata(path_h5ad)

    # --- Table export
    if path_table:
        exporter.export()
