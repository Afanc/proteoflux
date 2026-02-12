# ProteoFlux Configuration Reference

This document describes all available configuration parameters.

---

# 1. Dataset Section

## Core Parameters

| Key | Description |
|------|------------|
| `input_file` | Path to quantitative input table |
| `input_layout` | `long` or `wide` |
| `analysis_type` | `proteomics`, `peptidomics`, or `phospho` |
| `annotation_file` | Optional TSV with sample metadata |
| `exclude_runs` | String or list of run identifiers |
| `inject_runs` | Additional datasets to inject |

---

## Injected Runs

```yaml
inject_runs:
  RUN_NAME:
    input_file: ...
    indexing_type: ...
    annotation_file: ...
    is_covariate: true
    column_overrides:
      qvalue_column: ...
```

---

# 2. Column Mapping Keys

| Key | Description |
|------|------------|
| `index_column` | Feature identifier |
| `signal_column` | Intensity column |
| `qvalue_column` | q-value column |
| `pep_column` | PEP column |
| `condition_column` | Biological condition |
| `replicate_column` | Replicate number |
| `filename_column` | Run identifier |
| `ibaq_column` | iBAQ values |
| `gene_names_column` | Gene names |
| `uniprot_column` | UniProt accession |

---

# 3. PTM / Indexing Controls

| Key | Description |
|------|------------|
| `convert_numeric_ptms` | Default: true |
| `collapse_met_oxidation` | Default: true |
| `collapse_all_ptms` | Default: false |
| `phospho_multisite_collapse_policy` | `explode` or `retain` |

---

# 4. Preprocessing

## Filtering

| Key | Description |
|------|------------|
| `filtering.qvalue` | Maximum q-value |
| `filtering.pep` | Maximum PEP |
| `filtering.localization_threshold` | Phospho only |

---

## Quantification

| Key | Description |
|------|------------|
| `protein_rollup_method` | `sum` or `directlfq` |
| `peptide_rollup_method` | `sum`, `mean`, `median` |

---

## Normalization

| Key | Description |
|------|------------|
| `normalization.method` | List of normalization steps |
| `normalization.loess_span` | LOESS smoothing span |

---

## Imputation

Supported methods:

- `lc_conmed`
- `mean`
- `median`
- `knn`
- `tnknn`
- `mindet`
- `minprob`
- `randomforest`

---

# 5. Analysis

| Key | Description |
|------|------------|
| `only_contrasts` | Restrict comparisons |
| `only_against` | Restrict reference |
| `clustering_max` | Max features for clustering |

---

# 6. Export Section

| Key | Description |
|------|------------|
| `exports.path_h5ad` | Path to `.h5ad` output |
| `exports.path_table` | Path to table export |
| `exports.path_pdf_report` | Path to PDF report |

## PDF Options

| Key | Description |
|------|------------|
| `exports.pdf_report.volcano_top_annotated` | Annotated hits |
| `exports.pdf_report.volcano_sign_threshold` | FDR threshold |
| `exports.pdf_report.volcano_annotate_infinite` | Annotate semi-infinite |
| `exports.pdf_report.title` | Report title |
| `exports.pdf_report.intro_text` | Intro text |
| `exports.pdf_report.footer_text` | Footer text |
