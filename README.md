# ProteoFlux

ProteoFlux is a transparent and reproducible proteomics analysis pipeline.
It automates data loading, normalization, imputation, differential expression, clustering, and report generation for label-free quantitative proteomics and phosphoproteomics.

## Overview

ProteoFlux takes a single YAML configuration file and produces:

* Harmonized and preprocessed quantification data
* Differential expression results using the LIMMA-eBayes framework
* Clustering and missingness analyses
* A multi-page PDF report
* Portable `.h5ad` files compatible with **ProteoViewer**
* Summary tables in Excel or CSV format

The pipeline is fully modular and can be configured for DIA, DDA, or FragPipe-based workflows, with optional covariate handling for flow-through or parallel assays.

## Installation

ProteoFlux requires Python 3.8 or higher.

```bash
pip install proteoflux
```

or from source:

```bash
git clone https://gitlab.your-org/proteoflux.git
cd proteoflux
pip install -e .
```

## Command-line usage

```bash
# create a template configuration file
proteoflux init my_config.yaml

# run the full pipeline
proteoflux run --config my_config.yaml
```

This runs the following sequence:

1. Load and harmonize raw quantification tables
2. Preprocess (filtering, pivoting, normalization, imputation)
3. Run LIMMA differential expression
4. Perform PCA, UMAP, and hierarchical clustering
5. Generate a PDF report
6. Export Excel/CSV tables and `.h5ad` datasets

## Output files

| File                                     | Description                                                    |
| ---------------------------------------- | -------------------------------------------------------------- |
| `proteoflux_results.h5ad`                | Main AnnData object containing all processed data              |
| `proteoflux_table.xlsx`                  | Summary table of log2FC, q/p values, metadata, and missingness |
| `proteoflux_report.pdf`                  | Multi-page QC and results report                               |
| [`.csv` equivalents to .xlsx]            | If `use_xlsx: false` in the configuration                      |

## Viewing results

The `.h5ad` files can be inspected interactively with **ProteoViewer**.

* On Windows, run the `ProteoViewer.exe` application.
* On Linux or macOS, launch the panel app:

```bash
panel serve panel_app/app.py --autoreload
```

Then open the interface in a browser and load the `.h5ad` file using the client button.

## Configuration

All parameters are stored in a single YAML file.
The default template can be generated with `proteoflux init` and includes sections for:

* `dataset`: input files, annotation, and layout
* `preprocessing`: filtering, normalization, and imputation
* `analysis`: contrasts, thresholds, and export options

Every run is fully reproducible and self-documented through the metadata embedded in the exported `.h5ad`.

## Dependencies

Core dependencies (see `pyproject.toml`):

```
pandas, numpy, scanpy, anndata, matplotlib, seaborn, scikit-learn,
scikit-misc, polars, pyarrow, xlsxwriter, openpyxl, typer, pyyaml,
directlfq, inmoose, torch, tqdm, fastcluster
```

## License and citation

ProteoFlux is distributed under standard academic use terms.
A formal publication is in preparation; please refer to the upcoming release for citation details.
