from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.export.differential_expression_plotter import DifferentialExpressionPlotter
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.utils import debug_protein_view

# main.py
def run_pipeline(config: dict):
    dataset = Dataset(**config)
    adata = dataset.get_anndata()
    adata = run_limma_pipeline(adata, config)

    analysis_config = config.get("analysis", {})
    plotter = DifferentialExpressionPlotter(adata)

    if analysis_config.get("plot_extensive", True):
        plotter.plot_all(analysis_config.get("extensive_plot_path"))

    if analysis_config.get("export_h5ad", True):
        adata.write(analysis_config.get("h5ad_export_path"), compression="gzip")

    if analysis_config.get("export_log2fc_csv", True):  # TODO: rename key
        exporter = DEExporter(
            adata,
            output_path="DE_export",
            use_xlsx=True,
            sig_threshold=0.05,
            include_uniprot_map=False
        )
        exporter.export()

def main():
    #TODO this is the part that gets imported
    config = {
            "dataset": {"input_file": "full_test2.tsv",
            #"dataset": {"input_file": "searle_test.tsv",
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
              "preprocessing": {"filter_qvalue": 0.01,
                                "filter_pep": 0.2,
                                "filter_run_evidence_count": 3,
                                "contaminants_files": ["Contaminants_2022.fasta"],
                                "normalization":
                                    {
                                    "method": ["log2", "median_equalization"],
                                     "loess_span": 0.9},
                                "imputation":
                                    {
                                    "method": "tnknn",
                                     "knn_k": 6,
                                     "knn_tn_perc": 0.75,
                                     "rf_max_iter": 20,
                                     "rf_random_state": 42},
                                "exports": {
                                    "normalization_plots": False,
                                    "imputation_plots": False,
                                    "normalization_plot_path": "0_normalization_plots.pdf",
                                    "imputation_plot_path": "1_imputation_plots.pdf"},
                                },
              "analysis":{"plot_extensive": True,
                          "de_plot_path": "2_extensive_de.pdf",
                          "export_log2fc_table": True,
                          "log2fc_path": "3_log2fc_report.csv",
                          "export_h5ad": True,
                          "h5ad_export_path": "processed_data.h5ad",
                          "ebayes_method": "limma",
                          "use_r_limma": False,
                          },
              "design": {"mode": "default", "group_column": "CONDITION"},
          }

    dataset = Dataset(**config)
    adata = dataset.get_anndata()

    adata = run_limma_pipeline(adata, config)
    #debug_protein_view(adata, "Q15067")

    analysis_config = config.get("analysis")

    plotter = DifferentialExpressionPlotter(adata)
    if analysis_config.get("plot_extensive", True):
        plotter.plot_all(analysis_config.get("de_plot_path"))

    if analysis_config.get("export_h5ad", True):
        adata.write(analysis_config.get("h5ad_export_path"),
                    compression="gzip")

    print(adata)
    if analysis_config.get("export_log2fc_table", True): #TODO rename
        exporter = DEExporter(adata,
                      output_path=analysis_config.get("log2fc_path"),
                              use_xlsx=True,
                              sig_threshold=0.05,
                             )
        exporter.export()

if __name__ == "__main__":
    main()

