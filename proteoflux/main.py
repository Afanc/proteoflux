from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.limma_pipeline import run_limma_pipeline, clustering_pipeline
from proteoflux.analysis.pelsa_pipeline import run_pelsa_pipeline
from proteoflux.analysis.ksea_pipeline import run_ksea_pipeline, validate_ksea_database
from proteoflux.analysis.kinase_sequences import (
    configured_kinase_method,
    prepare_kinase_sequences,
    validate_kinase_sequence_config,
)
from proteoflux.analysis.phosx_pipeline import run_phosx_pipeline, validate_phosx_config
from proteoflux.analysis.validation import validate_analysis_input
from proteoflux.export.pdf_report_exporter import ReportPlotter
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.utils import debug_protein_view, logger, log_time
from proteoflux.utils.analysis_type import normalize_analysis_type


@log_time("Proteoflux Pipeline")
def run_pipeline(config: dict):

    dataset_config = (config or {}).get("dataset") or {}

    analysis_type = normalize_analysis_type(
        dataset_config.get("analysis_type")
    )
    if analysis_type == "phospho":
        validate_kinase_sequence_config(config)
        kinase_method = configured_kinase_method(config)
        if kinase_method == "ksea":
            validate_ksea_database(config)
        elif kinase_method == "phosx":
            validate_phosx_config(config)
    else:
        kinase_method = None

    dataset = Dataset(**config)

    adata = dataset.get_anndata()
    validate_analysis_input(adata)
    if analysis_type == "phospho":
        # Always export the peptide-derived ten-mer matrix for phospho data;
        # a configured FASTA enriches it independently of the selected method.
        adata = prepare_kinase_sequences(adata, config)

    if analysis_type == "pelsa":
        adata = run_pelsa_pipeline(adata, config)
    else:
        adata = run_limma_pipeline(adata, config)
        if analysis_type == "phospho":
            if kinase_method == "ksea":
                adata = run_ksea_pipeline(adata, config)
            elif kinase_method == "phosx":
                adata = run_phosx_pipeline(adata, config)

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
