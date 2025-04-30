from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.export.differential_expression_plotter import DifferentialExpressionPlotter
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.utils import debug_protein_view, logger, log_time

# main.py
@log_time("Proteoflux Pipeline")
def run_pipeline(config: dict):
    dataset = Dataset(**config)
    adata = dataset.get_anndata()
    adata = run_limma_pipeline(adata, config)

    analysis_config = config.get("analysis", {})
    plotter = DifferentialExpressionPlotter(adata,
                                            analysis_config)

    if analysis_config.get("export_plot", True):
        plotter.plot_all()

    export_config = analysis_config.get("exports")
    if analysis_config.get("export_table", True):
        exporter = DEExporter(adata,
                      output_path=export_config.get("path_table"),
                              use_xlsx=export_config.get("table_use_xlsx"),
                              sig_threshold=analysis_config.get("sign_threshold"),
                             )
        exporter.export()

    if export_config.get("export_h5ad", True):
        exporter.export_adata(export_config.get("path_h5ad"))

@log_time("Proteoflux Pipeline - Dev")
def main():
    config = {
            "dataset":
            {
                "input_file": "full_test2.tsv",
                #"input_file": "searle_test.tsv",
                "annotation_file": "annotation_test.tsv",
                "index_column": "PG.ProteinGroups",
                "signal_column": "FG.MS2RawQuantity",
                #"signal_column": "PG.Quantity",
                "qvalue_column": "PG.Qvalue",
                "pep_column": "PG.PEP",
                "condition_column": "R.Condition",
                "replicate_column": "R.Replicate",
                "filename_column": "R.FileName",
                "run_evidence_column": "PG.RunEvidenceCount",
                #"fasta_column": "PG.FastaHeaders",
                "fasta_column": "PG.FastaFiles",
                "protein_weight": "PG.MolecularWeight",
                "protein_descriptions": "PG.ProteinDescriptions",
                "gene_names": "PG.Genes",
            },
            "preprocessing":
            {
                "filtering":
                {
                    "qvalue": 0.01,
                    "pep": 0.003,
                    "run_evidence_count": 3,
                    "contaminants_files": ["Contaminants_2022.fasta"],
                },
                "normalization":
                {
                    "method": ["log2", "local_loess"],
                    "loess_span": 0.9
                },
                "imputation":
                {
                    "method": "tnknn",
                    "knn_k": 6,
                    "knn_tn_perc": 0.75,
                    "rf_max_iter": 20,
                    "rf_random_state": 42,
                    "evaluation":
                    {
                       "number_CV": 10,
                       "missing_total": 0.15,
                       "missing_mnar": 0.9,
                       "missing_quantile": 0.1,
                       "missing_q_lod": 0.05,
                    }
                },
                "exports":
                {
                    "normalization_plots": True,
                    "imputation_plots": False,
                    "normalization_plot_path": "0_normalization_plots.pdf",
                    "imputation_plot_path": "1_imputation_plots.pdf"},
                },
            "analysis":
            {
                "use_r_limma": False,
                "ebayes_method": "limma",
                "sign_threshold": 0.15,
                "exports":
                {
                    "export_plot": True,
                    "export_table": True,
                    "export_h5ad": True,
                    "table_use_xlsx": True,
                    "path_plot": "2_extensive_de.pdf",
                    "path_table": "3_log2fc_report.csv",
                    "path_h5ad": "4_processed_data.h5ad",
                    "umap_n_neighbors": 15,
                }
            },
            "design":
            {
                "mode": "default",
                "group_column": "CONDITION"
            },
        }

    dataset = Dataset(**config)
    adata = dataset.get_anndata()

    adata = run_limma_pipeline(adata, config)
    #debug_protein_view(adata, "Q15067")

    analysis_config = config.get("analysis")

    plotter = DifferentialExpressionPlotter(adata,
                                            analysis_config)
    if analysis_config.get("export_plot", True):
        plotter.plot_all()

    export_config = analysis_config.get("exports")
    if analysis_config.get("export_table", True):
        exporter = DEExporter(adata,
                      output_path=export_config.get("path_table"),
                              use_xlsx=export_config.get("table_use_xlsx"),
                              sig_threshold=analysis_config.get("sign_threshold"),
                             )
        exporter.export()

    if export_config.get("export_h5ad", True):
        exporter.export_adata(export_config.get("path_h5ad"))

if __name__ == "__main__":
    main()

