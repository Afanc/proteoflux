# ProteoFlux Configuration Reference

This guide describes the public configuration used by ProteoFlux 1.10.4.
Top-level blocks are `dataset`, `preprocessing`, `analysis`, optional `ancova`,
and `exports`. The tables give runtime defaults when a key is omitted unless
explicitly labelled as template settings. Templates are editable starting points,
not a complete specification of defaults. Validation occurs at several pipeline
stages; unknown keys are not universally rejected.

Use real YAML booleans (`true`/`false`) and `null`, not quoted strings. Relative
paths resolve from the working directory. Preserve the original input and YAML
alongside results. See [RESULTS.md](RESULTS.md) for exported data semantics.

---

# 1. Dataset Section

## Core Parameters

| Key | Description |
|------|------------|
| `analysis_type` | `proteomics`, `peptidomics`, `phospho` or `pelsa`. |
| `input_file` | Path to quantitative (fragment group/precursor level) input table. Supports `.tsv`, `.csv`, `.parquet` and `.pq`. |
| `input_layout` | `long` (one precursor per run per row) or `wide` (rows: precursors. columns: runs. ) |
| `annotation_file` | Path to .tsv with sample metadata. Optional (but recommended). File should have the following columns : Filename, Condition, Replicate. |
| `exclude_runs` | String or list of run identifiers (filenames) to exclude from the analysis. Optional. |
| `inject_runs` | Additional datasets to inject. Optional. |
| `load_method` | `polars` (default and sole supported delimited-file loader). PyArrow remains required for Parquet and table conversions, not a selectable CSV/TSV backend. |
| `convert_numeric_ptms` | Convert recognized numeric PTM annotations to names; default `true`. This is a dataset key and can change feature identity/aggregation. |

---

## Injected Runs

For phospho background adjustment, the injection name is arbitrary (`flowthrough`
is a convention). Exactly one active covariate source is supported. Main and
background runs are paired by condition and replicate, not filename order;
unmatched or ambiguous pairs are rejected. Background quantities align by parent
protein and are broadcast to its phosphosites.

| Key | Description |
|------|------------|
| `inject_runs.RUN_NAME.input_file` | Path to injected dataset. |
| `inject_runs.RUN_NAME.indexing_type` | Indexing strategy for this run: proteomics, peptidomics or phospho. Use `peptidomics` or `proteomics` for the corresponding background input. |
| `inject_runs.RUN_NAME.annotation_file` | Path to annotation file for the injected run. |
| `inject_runs.RUN_NAME.exclude_runs` | String or list of run identifiers (filenames) to exclude from the injected analysis. Should be used to drop phospho/flowthrough pairs. Optional.  |
| `inject_runs.RUN_NAME.is_covariate` | Boolean. If true, run is treated as covariate (phospho ANCOVA-style residualization using the flowthrough). |
| `inject_runs.RUN_NAME.column_overrides` | Dictionary overriding dataset column mappings for this run only. Optional. |
| `inject_runs.RUN_NAME.column_overrides.qvalue_column` | q-value column, for filtering. Optional. |
| `inject_runs.RUN_NAME.column_overrides.pep_column` | PEP column, for filtering. Optional. |
| `inject_runs.RUN_NAME.column_overrides.precursors_exp_column` | number of precursors column, for filtering. Optional. |
| `inject_runs.RUN_NAME.reuse_main_input_file` | `true` reuses the main file instead of reading a separate input. Do not also supply `input_file`. |

Reusing the main input is restricted to phospho long input with
`indexing_type: peptidomics`, `is_covariate: true` and a mapped `ptm_sites_column`.
Rows with empty/null-like PTM-site annotations are selected automatically for
background. This means non-phosphorylated rows, not necessarily peptides without
other modifications. Do not combine reused and separate covariate sources.

Sample annotations normally use `Filename`, `Condition`, `Replicate`; extra
columns can supply batch factors or PELSA concentrations. Run extensions are
normalized for matching. Conditions are sanitized for model/contrast names:
non-word characters become underscores and initial digits receive `C_`.
Distinct original names that collapse to the same sanitized name are rejected.
Use the sanitized names in contrasts; original labels remain in `CONDITION_ORIG`.

---

## Column Mapping Keys

| Key | Description |
|------|------------|
| `index_column` | Protein feature identifier for proteomics; peptide and phospho feature IDs are derived from sequence/site annotations. |
| `signal_column` | Intensity column. If wide, can be a suffix to match against. |
| `peptide_seq_column` | Required precursor sequence, including PTM labels where applicable; used for rollup, evidence and feature identity. |
| `charge_column` | Required precursor charge; used to distinguish precursor evidence. |
| `qvalue_column` | q-value column, for filtering. Optional. |
| `pep_column` | PEP column, for filtering. Optional. |
| `precursors_exp_column` | Number of precursors (experiment-wide)/Run Evidence count, for filtering. Optional. |
| `uniprot_column` | UniProt accession. Optional (if proteomics). Used to derive parent protein in phospho with flowthrough adjustment. |
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
| `ptm_sites_column` | PTM residue annotations, e.g. semicolon-separated S/T/Y. Inline modifications instead belong in `peptide_seq_column`. |
| `ptm_positions_column` | Peptide-relative PTM positions where supplied. |
| `ptm_proteinlocations_column` | Absolute PTM positions on protein (; separated integers). Optional (if not phospho). |
| `ptm_probabilities_column` | PTM localization probabilities (; separated floats). Optional (if not phospho). |

---

# 2. Preprocessing

## PTM / Indexing Controls

| Key | Description |
|------|------------|
| `collapse_all_ptms` | Whether to collapse all PTMs in peptide rollup. Default: false |
| `collapse_met_oxidation` | Whether to collapse oxidation on methionine in peptide rollup. Default: true |

---

## Quantification

| Key | Description |
|------|------------|
| `protein_rollup_method` | Protein rollup method. Accepts `sum`, `directlfq`, `robust_regression`, `median`, `top3`, `min`, `max`, `count`, `mean` or null |
| `peptide_rollup_method` | Peptide rollup method. `sum`, `mean`, `median` |
| `directlfq_cores` | Runtime default `4`; templates commonly specify `8`. |
| `directlfq_min_nonan` | Minimum observed ions per protein/sample to report a quantity. Default `1`; below the threshold the output is missing before imputation. |
| `robust_regression_cores` | Number of cores used for robust regression summarization. Default: same as `directlfq_cores` if not set explicitly (runtime `4`). |
| `robust_regression_huber_t` | Huber tuning constant used in the IRLS robust regression. Default: `1.345`. |
| `robust_regression_max_iter` | Maximum number of IRLS iterations for robust regression summarization. Default: `20`. |
| `robust_regression_tol` | Convergence tolerance for robust regression summarization. Default: `1e-8`. |
| `robust_regression_min_nonan` | Minimum number of observed peptide-level values required in a sample to report a protein quantity for that sample. Below this threshold, the output is set to `NA`. Default: `1`. |
| `robust_regression_disconnected_policy` | Handling of disconnected sample/peptide observations in proteomics robust rollup: `raise` (default) or `component`. See below. |

Protein and peptide rollup default to `sum`. `null` protein rollup falls back to
the legacy `pivot_signal_method`, then `sum`; setting both non-null keys is an
error. These methods are not interchangeable: robust regression estimates
sample effects in log2 space with zero-sum peptide effects; directLFQ has its own
ratio-based estimation and intensity scaling. Top3 sums precursors per peptide,
then sums the top three *dense ranks*, so ties can include more than three peptides.
All-missing sum groups remain missing. Linear-scale input is expected for these
quantification methods.

### Disconnected observations in robust regression

These settings belong under `preprocessing` in YAML or the dictionary passed to
`run_pipeline` / `Dataset`:

```yaml
preprocessing:
  protein_rollup_method: robust_regression
  robust_regression_disconnected_policy: raise
```

A protein is disconnected when its observed peptides link separate sample groups
with no shared-peptide path between the groups. Positive finite measurements
define these links after filtering and peptide aggregation. The relative protein
abundance scale between such groups cannot be identified from those observations.

- `raise` stops preprocessing with `DisconnectedObservationError`, identifying
  the protein and sample groups. This is the default when the key is omitted.
- `component` explicitly opts into fitting each component with its own zero-sum
  peptide-effect constraint. It retains every sample coefficient subject to
  `robust_regression_min_nonan`, and emits `DisconnectedObservationWarning` with
  the sample groups. Estimates are comparable **only within the same component**.

Component mode does **not** automatically restrict downstream contrasts, variance
estimation, normalization or plots. Between-component fold changes remain
unsupported. Component membership is protein-specific; selecting one contrast
does not prove it is supported for every protein. Use this mode only when the
downstream analysis respects those limits. H5AD/XLSX do not store component
membership or mark unsupported comparisons. H5AD records the chosen policy in
`uns["preprocessing"]["quantification"]["robust_regression_disconnected_policy"]`
for proteomics robust rollup.

Direct Python callers of
`proteoflux.workflow.rollup.robust_regression.pivot_df_robust_regression` use the
keyword argument `disconnected_policy="raise"` or `"component"` (also defaulting
to `"raise"`). Other values, including null, are rejected. Connected fits retain
their existing behavior under either policy. This setting does not apply to
sum/directLFQ rollup or the phospho background's separate rollup method.

---

## Filtering

| Key | Description |
|------|------------|
| `filtering.qvalue` | Maximum q-value, default `0.01`. |
| `filtering.pep` | Maximum PEP, runtime `0.2`; templates commonly specify `0.05`. |
| `filtering.pep_direction` | `lower` (default) accepts values below the threshold; `higher` reverses this for confidence scores. |
| `filtering.min_precursors` | Minimum precursor evidence, runtime `1`; `null` disables. |
| `filtering.min_linear_intensity` | Censor signals below this threshold to missing, not zero; runtime `null`, templates commonly `2`. |
| `filtering.contaminants_files` | List of contaminant FASTA files to exclude |

---

## Phospho

| Key | Description |
|------|------------|
| `phospho.multisite_collapse_policy` | `explode` (site-centric) or `retain` (peptide-centric) |
| `phospho.localization_filter_mode` | `soft` (default: maximum finite score across samples) or `strict` (minimum finite score); missing scores are not zeros. |
| `phospho.localization_filter_threshold` | Localization score filter threshold. Default: 0.75 (class I). |
| `phospho.covariate_protein_rollup_method` | `directlfq` (default) or `sum` for background protein quantities. |

---

## Normalization

| Key | Description |
|------|------------|
| `normalization.method` | Ordered list or single string: `log2`, `log10`, `median_equalization`, `median_equalization_by_tag`, `median_equalization_intersection`, `quantile`, `global_linear`, `global_loess`, `local_linear`, `local_loess`, `none`. Specify explicitly. For linear inputs to differential/PELSA analysis, begin with `log2`. |
| `normalization.reference_tag` | String or list of tags for `median_equalization_by_tag` (case-insensitive matching). Looks for matches in the fasta_column, fails if column not defined. Default: null. |
| `normalization.loess_span` | LOESS smoothing span for loess normalizations. Default: 0.9. |

---

`median_equalization` shifts each sample median to the pooled median.
`median_equalization_intersection` estimates offsets only from features finite
in every sample, then applies those offsets to all features; an empty intersection
skips the step. The tag method similarly requires tagged features observed in
all samples. Regression methods use a per-feature mean across samples as the
global reference or, for local methods, the first sample within each condition; local results therefore depend
on the choice/order of reference sample. `loess_span` controls smoothing.

`quantile` uses sklearn's per-sample transformation to a uniform distribution,
not classical normalization to a shared intensity distribution. It does not
preserve log2 units. Although `log10` is accepted, downstream fields called
`log2fc` and PELSA expect log2 units; these alternatives need a deliberate scale
interpretation, not the usual log2FC interpretation.

For already-log2 input, `none` or an empty method list leaves analysis values on
that scale and reconstructs raw layers with `2**value`. Rollup happens before
normalization: supply appropriately aggregated values, since summing logged
precursors is not equivalent to summing linear intensities and then logging.

## Imputation

| Key | Description |
|------|------------|
| `imputation.method` | `lc_conmed`, `mean`, `median`, `knn`, `nsknn`, `tnknn`, `mindet`, `minprob`, `randomforest`, or `null` (no imputation; also the omission default). The string `"none"` is invalid. |
| `imputation.lc_conmed_lod_k` | Number of lowest global datapoints to estimate LOD. Default: 10. |
| `imputation.lc_conmed_in_min_obs` | Minimum observed values per feature/condition to use conditional-median (MAR) imputation; fewer observations use left-censoring (MNAR). Default : 1. |
| `imputation.lc_conmed_frac_min_obs` | Minimum observed fraction for conditional-median imputation, rounded up as `ceil(fraction * condition_size)`. Set `lc_conmed_in_min_obs: null`; otherwise the integer threshold takes precedence. Must be in `(0, 1]`. Default : null. |
| `imputation.lc_conmed_lc_shift` | Shift below LOD for MNAR estimation. Default: 0.20. |
| `imputation.lc_conmed_lc_sd_width` | Jitter width for MNAR estimation. Default: 0.05. |
| `imputation.lc_conmed_jitter_frac` | Multiplier of pooled SD for MAR estimation. Default: 0.20. |
| `imputation.lc_conmed_q_lower` | Lower quartile bound for MAR clipping. Default: 0.25. |
| `imputation.lc_conmed_q_upper` | Upper quartile bound for MAR clipping. Default: 0.75. |
| `imputation.knn_k` | Number of neighbors for knn-based methods. Default: `6` for knn/nsknn, `10` for tnknn. |
| `imputation.knn_tn_perc` | Truncation percentage threshold for tnknn. Default: 0.75. |
| `imputation.lc_quantile` | Lower quantile used for minimum for mindet. Default: 0.01. |
| `imputation.lc_shift` | Mean shift for mindet. Default: 0.2. |
| `imputation.lc_mu_shift` | Mean shift for minprob. Default: 1.8. |
| `imputation.lc_sd_width` | SD width for minprob. Default: 0.3. |
| `imputation.lc_clip` | Boolean. Clip to quantile in minprob. Default: true. |
| `imputation.rf_max_depth` | Maximum depth of the boosting trees. Default: 6. |
| `imputation.rf_max_iter` | Maximum imputation iterations. Default: 20. |
| `imputation.rf_learning_rate` | Boosting learning rate. Default: 0.1. |
| `imputation.rf_n_estimators` | Maximum boosting iterations per regressor. Default: 100. |
| `imputation.rf_tol` | Imputation convergence tolerance. Default: 5e-2. |
| `imputation.rf_nearest_features` | Maximum neighboring predictors used by the iterative imputer. Default: 50. |
| `imputation.random_state` | Random seed for reproducibility. Default: 42. |

---

LC-ConMed preserves observed values and classifies support from the original
missingness mask. With sufficient support it uses the condition median plus
pooled within-condition jitter, clipped to that condition's observed interquartile
range. Otherwise it draws below the median of the lowest `lod_k` observed values
in the block. A wholly missing block uses a zero log-scale detection limit;
these fills are not evidence. Main and protein-background blocks are processed
separately; background is imputed once per parent protein before broadcast.

The same seed reproduces the same ordered input, but reordering features or
samples can change which stochastic draws are assigned to missing cells.
`separate_contrasts` does not isolate normalization or imputation from other
conditions. QC embeddings use their own missing-value treatment.

The method selected by `randomforest` is iterative imputation with a histogram
gradient-boosting regressor. Interpret its `rf_*` settings as parameters of that
estimator.
`knn` and `randomforest` currently cannot preserve an entirely empty sample column;
check input completeness before selecting them. No-imputation mode is useful for
PELSA (which normally fits observed normalized values); downstream methods differ
in their ability to handle missing values.

# 3. Analysis

| Key | Description |
|------|------------|
| `only_contrasts` | Restrict analysis to list of contrasts. Syntax "A_v_B" or "A_vs_B". Optional. |
| `only_against` | Restrict to comparisons against this condition. Optional. |
| `separate_contrasts` | Boolean, default `false`. `true` restricts the differential fit and variance estimation to each contrast’s two conditions. Global preprocessing and stage-1 phospho slope estimation remain global. |
| `batch_effect_columns` | Optional categorical factors, e.g. `["batch", "sex"]`; even numeric entries are treated as categories, not continuous regressors. Supply complete sample annotations. Confounded/rank-deficient designs and insufficient residual degrees of freedom are rejected. |
| `clustering_max` | Max number features for clustering. Default: 8000. |
| `pelsa.concentration_column` | Concentration column in the annotation file. Required for Pelsa |
| `pelsa.control_concentration` | Control concentration, default `0.0`. At least one matching sample is required. |
| `pelsa.min_control_points` | Minimum observed controls per peptide, runtime `1`, PELSA template `2`. |
| `pelsa.layer` | Data layer used for PELSA fitting. Default: `"normalized"`. Input is expected to be log2-transformed. |
| `pelsa.fit_loss` | Loss used during 4PL curve optimization. Options: `"mse"` (runtime default), `"huber"` (template), `"l1"`. |
| `pelsa.fit_huber_delta` | Huber loss transition threshold in log2-ratio units. Used only when `fit_loss: "huber"`. Runtime default `1.0`; template `0.5`. |
| `pelsa.fit_steps` | Number of optimizer steps for 4PL fitting. Default: `300`. |
| `pelsa.fit_lr` | Optimizer learning rate. Default: `0.03`. |
| `pelsa.fit_batch_size` | Number of features fitted per batch. Larger values are faster but use more RAM. Default: `4096`. |
| `pelsa.fit_n_starts` | Number of initialization starts per feature. Higher values reduce local-minimum risk but increase runtime. Runtime default `1`; template `5`. |
| `pelsa.fit_betas` | Adam optimizer beta parameters. Advanced option. Default: `[0.9, 0.999]`. |
| `pelsa.fit_eps` | Adam optimizer epsilon. Advanced option. Default: `1e-8`. |
| `pelsa.fit_weight_decay` | Adam optimizer weight decay. Advanced option. Default: `0.0`. |
| `pelsa.fit_dtype` | Torch numeric precision. Options: `"float32"` or `"float64"`. Default: `"float32"`. |
| `pelsa.fit_num_threads` | Number of CPU threads used by torch. Optional. |
| `pelsa.bounds.slope` | Lower and upper bounds for 4PL curve steepness. Default: `[0.01, 10.0]`. |
| `pelsa.bounds.front` | Lower and upper bounds for the left/front plateau in log2-ratio units. Must span `0`. Default: `[-15.0, 15.0]`. |
| `pelsa.bounds.back` | Lower and upper bounds for the right/back plateau in log2-ratio units. Must span `0`. Default: `[-15.0, 15.0]`. |
| `pelsa.bounds.pec50_margin_log10` | Allowed pEC50 margin outside the observed log10 concentration range. Default `0.0` keeps EC50 within the fitted dose range, which includes the pseudo-control dose. |
| `pelsa.bounds.response_margin_fraction` | Extra margin allowed around the observed response range when bounding plateaus. Runtime default `0.5`; template `0.0` keeps plateaus inside the observed response range. |
| `kinase_activity.method` | `ksea` or `phosx`; omit/null to disable. Phospho analyses only. |
| `kinase_activity.min_substrates` | Minimum number of matched phosphosites with usable values required to test a kinase in a contrast. Default: `5` for KSEA, `4` for PhosX. |
| `kinase_activity.substrate_database` | Path or ordered list of paths to kinase–substrate databases. Relationships are combined and deduplicated (first relationship kept); database order does not add statistical weight. |
| `kinase_activity.database_columns` | Maps the standardized KSEA fields below to column names shared by the configured database files. |
| `kinase_activity.database_columns.kinase` | Kinase name column; used as a fallback identifier when no gene or UniProt accession is available. |
| `kinase_activity.database_columns.kinase_uniprot` | Kinase UniProt accession column. Kinase isoform suffixes are collapsed for activity scoring (e.g. `P27448-3` -> `P27448`). |
| `kinase_activity.database_columns.kinase_gene` | Kinase gene-symbol column and preferred identity for kinase-level activity results. |
| `kinase_activity.database_columns.substrate_uniprot` | Substrate UniProt accession column used for primary phosphosite matching. Substrate isoforms are retained. |
| `kinase_activity.database_columns.substrate_site` | Substrate phosphorylation-site column in residue notation, such as `S473`, `T308` or `Y15`. |
| `kinase_activity.database_columns.substrate_gene` | Substrate gene-symbol column, used with the site only when that database row has no substrate UniProt accession. |
| `kinase_activity.database_columns.source` | Relationship source or provenance column displayed and exported with matched kinase–substrate assignments. |

KSEA reports activity at the kinase-gene level. Accession-only kinase rows are assigned to a gene when the mapping is unambiguous; otherwise the normalized accession is retained. Substrates are matched using UniProt accession + site, with gene + site used only as a fallback when the accession is missing.

Duplicate kinase–substrate-site relationships within or across databases contribute only once, while all database sources are retained.

---

## Differential fitting and pilot mode

`A_vs_B` means A minus B on the analysis scale; `A_v_B` is accepted and stored
as `A_vs_B`. `only_contrasts` takes precedence over `only_against`. By default,
selecting contrasts does not remove other conditions from variance estimation.
BH correction is per contrast across retained features. If both contrast sides
are entirely missing in raw intensity, that feature/contrast is neutralized
before BH (logFC/t = 0, p/q = 1); imputed values do not count as observations.

A single condition, or any singleton condition anywhere in the experiment,
activates pilot mode for differential analysis, even if the singleton is not in
the requested contrast. Fold changes remain available when a contrast exists;
limma p/q values and phospho covariate adjustment are not produced. Optional kinase
analysis can still run on available fold changes; its probabilities are a
separate procedure, not restored limma significance.

## Phospho background adjustment

The default two-stage procedure subtracts a matched, centered protein-background
signal with slope one, then fits differential models to the residuals. It is not
generally equivalent to fitting a joint condition-plus-covariate ANCOVA model.
The optional **top-level** `ancova` block supports `beta_mode: fixed1` (default),
`free` or `nonneg`, and `ridge_lambda: 1e-6`.
Fitted slopes use all samples even with `separate_contrasts: true`.

For a site/contrast without observed background in either compared condition,
adjustment is suppressed: `log2fc`, `p_ebayes` and `q_ebayes` fall back to unadjusted
phospho results, `cov_part` is zero and FT support fields identify the lack of
support. Do not label every value in `log2fc` as background-adjusted merely because
a background was supplied. Fitted slope modes suppress globally flat backgrounds;
`fixed1` continues to apply slope one. The near-flat diagnostic threshold includes a
fixed SD floor of 0.1 log2 units and is not a configuration option.

The no-background fallback does not replace residual-model standard errors,
t-statistics or unmoderated p/q arrays. Those fields must not be interpreted as
statistics for the fallback fold change. Likewise the final mixed `q_ebayes`
array contains q-values from the respective model families, not a new BH pass
over the mixed p-vector.

`covariate_p` tests a stage-1 **phosphosite–background association slope of zero**,
not differential protein abundance and not a slope of one. `covariate_q` applies
BH over sites. Both contain one diagnostic per phosphosite, independent of
contrast count, and are model-based diagnostics that may include imputed values.
`fixed1` still computes this separate fitted-slope diagnostic while applying slope
one. `free`/`nonneg` neutralize wholly unobserved background diagnostics before BH;
flat diagnostics are neutral. FT differential p/q values are separate,
contrast-specific fields. See [RESULTS.md](RESULTS.md).

## Kinase activity

KSEA uses eligible finite phosphosite fold changes as its background, without
filtering by differential q-value. It compares each kinase's mean to the overall
mean using a normal z-test with the background sample SD; BH includes tested
kinases within each contrast. Missing/insufficient support remains untested,
not a zero activity estimate.

PhosX uses site-centered peptide windows and bundled PSSM resources. An optional
local protein FASTA (`analysis.kinase_activity.fasta`) extends peptide-derived
windows when accession/position/residue mappings agree; it is not required.
Conflicting or unmapped contexts can leave windows incomplete or ineligible.
PhosX ranks nonzero finite
fold changes, excludes fully imputed contrasts, and treats Ser/Thr and Tyr as
separate testing families within each contrast. Model identifiers remain distinct
(e.g. Tyr-specific models); canonical gene/accession fields support annotation.
PhosX scores and KSEA z-scores have different meanings and are not interchangeable.

PhosX settings under `analysis.kinase_activity`:

| Key | Runtime default / meaning |
|-----|---------------------------|
| `n_permutations` | `20000`; affects empirical probability resolution and runtime. |
| `random_seed` | `1`; also combined with contrast position, so changing contrast order can change draws. |
| `n_processes` | `1`; number of permutation workers. |
| `ser_thr_only`, `tyr_only` | Both `false`; cannot both be true. |
| `s_t_n_top_kinases`, `y_n_top_kinases` | `5` each. |
| `s_t_min_quantile`, `y_min_quantile` | `0.95`, `0.90`. |
| `no_upstream_activation_evidence` | `false`; activation evidence is enabled. |
| `a_loop_s_t_quantile_threshold`, `a_loop_y_quantile_threshold` | `0.95` each. |
| `upreg_redundancy_threshold`, `downreg_redundancy_threshold` | `0.5` each. |
| `decay_factor` | `256.0`, positive finite. |
| `phosx_resources` | Optional resource-file overrides; custom kinase models also require a known identifier mapping. |

## PELSA interpretation

The packaged `proteoflux/templates/spectronaut-pelsa.yaml` can be copied manually;
it is not currently registered by `proteoflux init`. The working example is
[examples/pelsa_small](../examples/pelsa_small/README.md). Its null limma settings
are not used by the separate PELSA analysis path.

PELSA fits four-parameter logistic responses on log2 ratios to each peptide's
**mean linear control abundance**, using the observed `normalized` layer by
default. Non-control concentrations must be positive and all concentrations
finite. A configured finite control sentinel is allowed. Controls are placed one
median distinct log-dose spacing below the lowest positive dose for fitting.
At least four distinct fitted dose positions (including control) are needed;
replicate count cannot substitute for dose diversity.

`pec50` is minus log10(EC50); `10**(-pec50)` is in the input concentration units.
Positive slope describes steepness, while `back - front` describes response
direction. A finite successful fit is not a guarantee of convergence,
identifiability or a well-constrained EC50. Inspect bounds, residuals and confidence
intervals. Curve p-values use a recalibrated F distribution, not the ordinary
nested-model F test; BH is over valid curve tests. Perfect zero-SSE fits currently
have no curve p-value. KSEA, PhosX, limma and PELSA probabilities belong to distinct
testing procedures and must not be pooled as one family.

# 4. Export Section

| Key | Description |
|------|------------|
| `path_h5ad` | Required path to `.h5ad` output. |
| `path_table` | Optional workbook path; exported with an `.xlsx` suffix. |
| `path_pdf_report` | Path to PDF report. Optional. |

## PDF Options

| Key | Description |
|------|------------|
| `pdf_report.volcano_top_annotated` | Number of annotated hits for the volcanoes |
| `pdf_report.volcano_sign_threshold` | FDR threshold for the volcanoes |
| `pdf_report.volcano_show_fully_imputed` | Show the separate fully-imputed point class. Default `true`. |
| `pdf_report.title` | Report title |
| `pdf_report.intro_text` | Intro text |
| `pdf_report.footer_text` | Footer text |
