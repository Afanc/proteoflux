# ProteoFlux Configuration Reference

This document describes all available configuration parameters.

---

# 1. Dataset Section

## Core Parameters

| Key | Description |
|------|------------|
| `input_file` | Path to quantitative (fragment group/precursor level) input table. Support .tsv, .csv and .parquet. |
| `input_layout` | `long` or `wide` |
| `analysis_type` | `proteomics`, `peptidomics`, or `phospho` |
| `annotation_file` | Path to .tsv with sample metadata. Optional (but recommended). File should have the following columns : Filename, Condition, Replicate. |
| `exclude_runs` | String or list of run identifiers (filenames). Optional. |
| `inject_runs` | Additional datasets to inject. Optional. |

---

## Injected Runs

Used for phospho flowthrough adjustement, RUN_NAME should be `flowthrough`.
Can be used to inject an additional .tsv/.csv/.parquet dataset.

| Key | Description |
|------|------------|
| `inject_runs.RUN_NAME.input_file` | Path to injected dataset. |
| `inject_runs.RUN_NAME.indexing_type` | Indexing strategy for this run: proteomics, peptidomics or phospho. Should be peptidomics if analysis is phospho and injected run is a flowthrough. |
| `inject_runs.RUN_NAME.annotation_file` | Path to annotation file for the injected run. |
| `inject_runs.RUN_NAME.is_covariate` | Boolean. If true, run is treated as covariate (phospho ANCOVA-style residualization using the flowthrough). |
| `inject_runs.RUN_NAME.column_overrides` | Dictionary overriding dataset column mappings for this run only. Optional. |
| `inject_runs.RUN_NAME.column_overrides.qvalue_column` | q-value column, for filtering. Optinoal. |
| `inject_runs.RUN_NAME.column_overrides.pep_column` | pep colum, for filtering. Optional. |
| `inject_runs.RUN_NAME.column_overrides.precursors_exp_column` | number of precursors column, for filtering. Optional. |

---

## Column Mapping Keys

| Key | Description |
|------|------------|
| `index_column` | Feature identifier. Optional (if wide, peptidomics or phospho). |
| `signal_column` | Intensity column. If wide, can be a suffix to match against. |
| `peptide_seq_column` | (PTM-Labeled) precursor sequence column. Used for peptide rollup. |
| `charge_column` | Precursor charge column. Used for peptide rollup. |
| `qvalue_column` | q-value column, for filtering. Optional. |
| `pep_column` | PEP column, for filtering. Optional. |
| `precursors_exp_column` | Number of precursors (experiment-wide)/Run Evidence count, for filtering. Optional. |
| `uniprot_column` | UniProt accession. Used for protein rollup. Optional (if peptidomics or phospho). |
| `condition_column` | Biological condition. Optional (if annotation provided). |
| `replicate_column` | Replicate number. Optional (if annotation provided). |
| `filename_column` | Run identifier. Optional (if wide). |
| `fasta_column` | FASTA Description. Used by median_equalization_by_tag. Optional. |
| `ibaq_column` | iBAQ values. Optional. |
| `spectral_counts_column` | Spectral counts. Optional. |
| `gene_names_column` | Gene names. Optional. |
| `protein_weight_column` | Protein weights. Optional. |

---

## Phospho Column Mapping Keys

| Key | Description |
|------|------------|
| `peptide_start_column` | Peptide position/start column. Optional (if not phospho). |
| `ptm_sites_column` | PTM site annotation string (sequence with inline [PTM]s). Optional (if not phospho). |
| `ptm_sites_column` | PTM sites column (; separated single amino acid strings). Optional (if not phospho). |
| `ptm_proteinlocations_column` | Absolute PTM positions on protein (; separated integers). Optional (if not phospho). |
| `ptm_probabilities_column` | PTM localization probabilities (; separated floats). Optional (if not phospho). |

---

# 2. Preprocessing

## PTM / Indexing Controls

| Key | Description |
|------|------------|
| `convert_numeric_ptms` | Whether to convert PTM masses to literals (no impact on analysis). Default: true |
| `collapse_all_ptms` | Whether to collapse all PTMs in peptide rollup. Default: false |
| `collapse_met_oxidation` | Whether to collapse oxydation on Methionine in peptide rollup. Default: true |

---

## Quantification

| Key | Description |
|------|------------|
| `protein_rollup_method` | Protein rollup method. Accepts `sum`, `directlfq`, `median`, `top3`, `min`, `max`, `count`, `mean` or null |
| `peptide_rollup_method` | Peptide rollup method. `sum`, `mean`, `median` |
| `directlfq_cores` | Number of cores used in DirectLFQ. |
| `directlfq_min_nonan` | Min number of shared to compute DirectLFQ, outputs NA (-> imputed) if below. Default: 1. |

---

## Filtering

| Key | Description |
|------|------------|
| `filtering.qvalue` | Maximum q-value |
| `filtering.pep` | Maximum PEP |
| `filtering.min_precursors` | Min number of precursors |
| `filtering.min_linear_intensity` | Min intensity in linear scale |
| `filtering.contaminants_files` | List of contaminants (FASTA files)) to exclude |
| `filtering.localization_threshold` | Minimum localization probability (for phospho) |

---

## Phospho

| Key | Description |
|------|------------|
| `phospho.multisite_collapse_policy` | `explode` or `retain` |
| `phospho.localization_filter_mode` | `soft` (max across samples) or `strict` (min across samples) |
| `phospho.localization_filter_threshold` | Default: 0.75 |
| `phospho.covariate_protein_rollup_method` | `directlfq` or `sum` |

---

## Normalization

| Key | Description |
|------|------------|
| `normalization.method` | List of normalization steps. Options include log2, median_equalization, median_equalization_by_tag, quantile, global_linear, global_loess, local_linear, local_loess. Data should be always log2 normalized first. |
| `normalization.reference_tag` | tag for median_normalization_by_tag. Default: null. |
| `normalization.loess_span` | LOESS smoothing span for loess normalizations. Default: 0.9. |

---

## Imputation

| Key | Description |
|------|------------|
| `imputation.method` | Imputation method. Options include lc_conmed, mean, median, knn, tnknn, mindet, minprob, randomforest |
| `imputation.lc_conmed_lod_k` | Number of lowest global datapoints to estimate LOD. Default: 10. |
| `imputation.lc_conmed_in_min_obs` | Min. number of observations per condition to use MAR (left-censoring) instead of MNAR (conditional median). Default : 1. |
| `imputation.lc_conmed_lc_shift` | Shift below LOD for MNAR estimation. Default: 0.20. |
| `imputation.lc_conmed_lc_sd_width` | Jitter width for MNAR estimation. Default: 0.05. |
| `imputation.lc_conmed_jitter_frac` | Multiplier of pooled SD for MAR estimation. Default: 0.20. |
| `imputation.lc_conmed_q_lower` | Lower quartile bound for MAR clipping. Default: 0.25. |
| `imputation.lc_conmed_q_upper` | Upper quartile bound for MAR clipping. Default: 0.75. |
| `imputation.knn_k` | Number of neighbors for knn-based methods. Default: 6. |
| `imputation.knn_tn_perc` | Truncation percentage threshold for tnknn. Default: 0.75. |
| `imputation.lc_quantile` | Lower quantile used for minimum for mindet. Default: 0.01. |
| `imputation.lc_shift` | Mean shift for mindet. Default: 0.2. |
| `imputation.lc_mu_shift` | Mean shift for minprob. Default: 1.8. |
| `imputation.lc_sd_width` | SD width for minprob. Default: 0.3. |
| `imputation.lc_clip` | Boolean. Clip to quantile in minprob. Default: true. |
| `imputation.rf_max_depth` | Max tree depth for RF. Default: 6. |
| `imputation.rf_max_iter` | Max number of iterations used for RF. Default: 20. |
| `imputation.rf_learning_rate` | Learning rate for RF. Default: 0.1. |
| `imputation.rf_n_estimators` | Number of estimators for RF. Default: 100. |
| `imputation.rf_tol` | Convergence tolerance for RF. Default: 5e-2. |
| `imputation.rf_nearest_features` | Number of nearest features used for RF. Default: 50. |
| `imputation.random_state` | Random seed for reproducibility. Default: 42. |

---

# 3. Analysis

| Key | Description |
|------|------------|
| `only_contrasts` | Restrict comparisons |
| `only_against` | Restrict reference |
| `clustering_max` | Max number features for clustering |

---

# 4. Export Section

| Key | Description |
|------|------------|
| `path_h5ad` | Path to `.h5ad` output |
| `path_table` | Path to table export. Optional. |
| `path_pdf_report` | Path to PDF report. Optional. |

## PDF Options

| Key | Description |
|------|------------|
| `pdf_report.volcano_top_annotated` | Number of annotated hits for the volcanoes |
| `pdf_report.volcano_sign_threshold` | FDR threshold for the volcanoes |
| `pdf_report.volcano_annotate_infinite` | Boolean. Whether to show semi-infinite (fully imputed in one condition) |
| `pdf_report.title` | Report title |
| `pdf_report.intro_text` | Intro text |
| `pdf_report.footer_text` | Footer text |
