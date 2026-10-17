# AnnData, serialized results, and downstream consumers

The result object is assembled in `workflow/dataset.py`, extended by analysis/QC,
and consumed by `export/de_exporter.py` and `export/pdf_report_exporter.py`.
`analysis/adata_schema.py` names many analysis keys but is not an exhaustive schema
or validator: dataset, kinase, PELSA and export code define additional contracts.

ProteoViewer is external and its source is not in this checkout. A read-only audit
of its additional workspace established the focused consumer contract recorded
in `tests/fixtures/viewer_contract.json`; see [testing.md](testing.md) for the
revision and limits. The suite does not execute the viewer. Renaming fields,
changing shapes/units or contrast/feature IDs may require coordinated changes.

## Current 1.10.4 policy additions

`uns.preprocessing.quantitative_scale` is log2. `uns.analysis.pilot_contrasts`
contains all selected names or is empty: pilot_study_mode applies uniformly to
the whole selected analysis. Pilot results omit inferential arrays; no mixed
pilot/non-pilot output is supported.
`varm.covariate_adjusted` is a boolean features × contrasts provenance array.
Fallback selects all model statistics before per-contrast BH. Slope diagnostics
are features × 1 and missing when unestimable. See [RESULTS.md](../RESULTS.md)
and the statistics/phospho context for the exact families and support criteria.
Existing matrix axes and evidence layouts are preserved.

## Orientation and lifecycle

Preprocessing matrices are feature x sample Polars DataFrames with explicit INDEX
and sample columns. `PreprocessResults` is the return bundle;
`IntermediateResults` additionally retains diagnostic frames/arrays/models that
are not all serialized. `Dataset._convert_to_anndata`:

1. Takes processed matrix row/sample order as canonical.
2. Reorders feature metadata by INDEX and sample metadata by Sample.
3. Converts primary matrices to float32 and transposes into AnnData.
4. Attaches covariate/evidence/preprocessing metadata and analysis-type markers.

`X` and every AnnData layer are **samples x features**. Differential arrays in
`varm` are normally **features x contrasts**, ordered by `var_names` and
`uns['contrast_names']`; covariate coefficient fields may have one column.
Embeddings in `obsm` are sample rows. Evidence matrices stored in `uns` are
**evidence rows x samples** and have explicit labels; they are not automatically
aligned or sliced by AnnData like layers/varm. Preserve those associations when
subsetting or exporting.

## Core fields

| Location | Meaning |
| --- | --- |
| `X` | Processed signal after configured normalization/imputation, log2; may still contain NaNs with no imputation or LC-ConMed `mnar_only` |
| `layers.raw` | Filtered, rolled-up, pre-normalization/pre-imputation signal, normally linear; not original vendor rows and not `adata.raw` |
| `layers.lognorm` | Snapshot after the last explicit log transform; name does not imply median/regression normalization |
| `layers.normalized` | After all configured normalization, before imputation; input for default QC and PELSA |
| `layers.qvalue`, `pep` | Aggregated identification confidence; not differential p/q-values |
| `layers.locprob` | Per-site/run maximum localization probability, phospho |
| `layers.spectral_counts` | Native distinct positive-signal precursor counts; not necessarily vendor spectral counts |
| `layers.ibaq` | Optional aggregated iBAQ |
| `layers.centered` | QC-filled selected normalized features minus per-feature sample mean; unselected features NaN-padded |

Read the normalization caveats in
[preprocessing-and-quantification.md](preprocessing-and-quantification.md) before
relying on raw/processed scale when an explicit log transformation is absent.
There is no dedicated exported main imputation-mask layer.

Raw is the post-filter/censoring, feature-level quantity, not an untouched input
row. With explicit log2 normalization, scalar and list settings preserve
the same raw values and NaNs; subsequent normalization/imputation changes only
analysis values. The 1.10.4 scalar-setting correction prevents accidental raw
exponentiation (and background substitution when covariates are enabled).
`Dataset._convert_to_anndata` stores raw as float32; `DEExporter` writes these
values directly to H5AD `layers.raw` and workbook `Raw_Intensities<sample>`.
Background raw similarly feeds `raw_covariate` and `FT_Raw_<sample>`. Evidence
tables retain independent pre-normalization values/axes. Names and shapes are
unchanged by the repair; affected historical raw values/masks are corrected.
The separate no-log convention remains; its cross-block raw overwrite was fixed
in the adversarial audit. See the preprocessing context for scale assumptions.

`obs` uses normalized filenames as its index and includes CONDITION, REPLICATE,
CONDITION_ORIG, ALIGN_KEY; optional ASSAY/IS_COVARIATE, Concentration and configured
batch annotations. Covariate runs are not extra observations in the final main
matrix: their values are aligned/broadcast into layers.

`var` is feature metadata, not always protein metadata. Depending on analysis it
contains FASTA_HEADERS, GENE_NAMES, PROTEIN_DESCRIPTIONS, PROTEIN_WEIGHT,
PRECURSORS_EXP, PRECURSORS_USED, IBAQ, UNIPROT, parent IDs, and per-condition
`CONSISTENT_PEPTIDE_<condition>` or `CONSISTENT_PRECURSOR_<condition>`. Metadata
aggregation often selects the first long-table value; do not infer complete
ambiguity provenance from a single retained annotation. Phospho and PELSA add
their own sequence/localization/support fields described in their documents.

Optional covariate layers are `raw_covariate`, `lognorm_covariate`,
`normalized_covariate`, `processed_covariate`, `centered_covariate`,
`qval_covariate`, `pep_covariate`, `sc_covariate`, `ibaq_covariate`.
The abbreviated `qval`/`sc` spellings are intentional stored contracts.
`residuals_covariate` is added by adjusted limma. Background values are repeated
for sites sharing a parent; they are not independent protein measurements.

## Analysis and diagnostic payloads

- `uns.analysis` stores type/method and limma design metadata when applicable.
  Dataset construction sets `analysis_method='limma_ebayes'` and `de_method` for
  non-PELSA; PELSA uses `pelsa_curve_fit`. Pilot state is a separate top-level flag.
- Limma stores `log2fc`, `se_raw`, `t_raw`, `p_raw`, `q_raw`, `se_ebayes`,
  `t_ebayes`, `p_ebayes`, `q_ebayes` in varm, with `contrast_names` in uns.
  Pilot runs legitimately lack inferential arrays; one-condition runs also lack
  log2fc. Do not synthesize zero statistics to make a consumer happy.
- `uns.residual_variance` is a feature vector for global fits and a feature x
  contrast matrix for separate fits. Design metadata likewise changes shape/scope.
- `uns.missingness` is a full-feature DataFrame of NaN counts by condition, with
  source/rule metadata. It prefers identification QVALUE, which need not agree
  with raw signal observed counts.
- `has_covariate`, `pilot_study_mode`, `n_fully_imputed_cells`, optional
  `model_type`/`decomposition_rule`, extended raw/FT/covariate statistics, and
  `nrsc`/`nrsc_misalignment` are consumed separately. See the statistical documents
  for masking, dimensions and current covariate-field caveats.
- `covariate_se/t/p/q` are stage-1 slope diagnostics, features x 1, independent of
  contrast selection; `covariate_beta` stores the applied unit slope.
  `ft_p_ebayes`/`ft_q_ebayes` are distinct background differential
  tests, features x contrasts. The 1.10.4 repair restores slope p/q values and
  their two-dimensional shape; old files may instead contain FT probabilities
  with an extra axis. Do not silently reinterpret those old diagnostic fields.
  See the phospho context for hypotheses, missingness and testing scopes.
- `uns.preprocessing` stores input layout/type, exclusions, filtering summaries,
  quantification settings, normalization/imputation dictionaries, distributions,
  and optional phospho/background-source provenance. It is not the complete
  original YAML and does not capture every resolved default or dependency version.
- `obsm.X_pca`, optional X_umap/X_mds/X_pca_design_adjusted and matching uns
  metadata support sample plots. Neighbor graphs can live in obsp. Linkages and
  ordered IDs use `intensity_*`, `centered_*`, `missing_*` keys. Feature-capped
  clustering stores subset indices/names; orders need not span all var rows.

Specialized payloads remain inside the same AnnData:

- `uns.pelsa`: curve points/results, control/model/scales, fit summary and
  localization. It does not provide limma contrast matrices.
- `uns.kinase_sequences` (version 2), plus var ACCESSION_SITES, KINASE_10MER,
  SITE_15MER: prepared phosphosite identity/windows, independent of kinase method.
- `uns.kinase_activity` (version 1): method, provenance and DataFrames named
  `results`, `substrates`, `condition_summary`, `condition_kinases`,
  `contrast_summary`; KSEA also supplies database metadata/summary. Result IDs
  join contrast and kinase with `|`; substrate IDs add phosphosite. QC can append
  `clustering` with `significant`/`all_tested` profiles and legacy top-level fields.
  KSEA and PhosX share column names but not the meaning of activity_score/effect
  or kinase_id. Respect `method` and identifier provenance.

## Drilldowns and consistency compatibility

`uns.peptides` is attached for proteomics, with parallel keys `rows`,
`protein_index`, `peptide_seq`, `cols`, `raw`, `centered`. Rows are PEPTIDE_ID;
protein_index maps them to the main feature index. Raw and centered are float32
evidence x sample arrays; centered is raw divided by its row mean.

`uns.precursors` uses `rows`, `protein_index`, `peptide_seq`, `charge`, `cols`,
`raw`, `centered`, plus `phosphoform_positions` for phospho. For phospho,
protein_index actually stores **SITE_ID** and peptide_seq stores **PEPTIDE_INDEX**.
Its site rows are filtered to retained features before attachment. For non-phospho
the builder derives identifiers from INDEX before `|`; do not assume this table
always has literal peptide-sequence granularity merely because it says precursor.

`uns.consistent_peptides_per_condition` always exposes the legacy key, even when
counts actually came from precursors; that case also adds
`consistent_precursors_per_condition`. Payloads use `index`, `columns`, `values`
in full feature order. Preserve this alias along with the explicit var columns.

The three centering operations must not be interchanged:

| Context | Operation |
| --- | --- |
| Evidence `uns.peptides/precursors.centered` | Divide linear values by row mean |
| `layers.centered_covariate` | Subtract per-feature median after background imputation |
| QC `layers.centered` | Subtract per-feature mean from QC-filled normalized values |

## H5AD writing

Proteomics robust rollup records its disconnected-observation policy at
`uns.preprocessing.quantification.robust_regression_disconnected_policy`.
This additive metadata key does not store component membership or certify
between-component comparisons. See the preprocessing context and public
configuration reference for the explicit `component` opt-in limits.

`DEExporter.export_adata` writes a **copy**, because AnnData serialization can
mutate metadata types. Explicit selected metadata becomes categorical. It adds
`uns.proteoflux.pf_version` and `created_at` using setdefault, summarizes/removes
filtering `values` arrays, and uses gzip with
`allow_write_nullable_strings=True`, `convert_strings_to_categoricals=False`.
This is not a universal sanitizer: `raw_values` arrays and many nested payloads
remain. The in-memory analysis should remain usable for later exports.

`created_at` currently appends `Z` to local `datetime.now()`; do not infer verified
UTC provenance from that suffix. Installed package metadata supplies pf_version;
source modifications can therefore exist under an unchanged installed version.
There is no global schema-version field beyond specialized payload versions.

The H5AD writer and PDF path do not create missing parent directories; table
export does. Since PDF precedes H5AD and table, failures can leave partial outputs
or prevent the required H5AD write. Main returns no result object/status bundle.

## Tabular output contracts

`DEExporter.export` builds Summary plus optional Identification Qvalue,
Identification PEP, Spectral Counts, IBAQ Values, Phosphosite Evidence and
Peptides (raw) sheets. Excel also has README. These identification matrices are
aggregated feature/run values, despite older workbook text referring to PSMs.

The pipeline uses `use_xlsx=True`; the exporter replaces the supplied suffix with
`.xlsx`. A `.csv` path alone does not choose CSV. Direct library callers can use
`use_xlsx=False`, which writes multiple prefix-based CSV files rather than the
single path returned by `export`. `annotate_matrix` is forced off. Summary is
not filtered to significant hits by `sig_threshold`.

Important column contracts:

- Non-phospho keeps `log2FC_<contrast>`, `QVALUE_`, `PVALUE_`, `NRSC_`,
  `NRSC_MISALIGNMENT_`. Processed and raw sample columns use exact prefixes
  `Processed_log2_Intensities` and `Raw_Intensities` without an added separator.
- Phospho supplies `RAW_log2FC_`, `ADJUSTED_log2FC_`, RAW/ADJUSTED_QVALUE/PVALUE,
  optional `COVARIATE_PART_` and FT statistics/intensities, LOCSCORE per sample,
  MAX_LOC_PROB and site evidence metadata. Without a covariate model raw and
  adjusted labels can contain the same statistics. RAW_PVALUE falls back to
  standard p-values only when raw_p_ebayes is absent. Covariate analyses now store
  raw_p_ebayes explicitly, fixing mislabeled RAW_PVALUE values in earlier outputs.
- `Observed_<condition>` counts non-NaN raw intensities. `Observed_<A_vs_B>` is
  the **maximum** of the two condition counts, not their sum/minimum. FT observed
  columns use the same maximum rule. MAX_OBSERVED is omitted when contrast-level
  equivalents exist; consistency columns retain MAX_CONSISTENT.
- Summary index name is PEPTIDE for PELSA, PHOSPHOSITE for phospho, otherwise
  UNIPROT_AC (including peptidomics). These names do not redefine feature identity.
- Phosphosite Evidence counts distinct contributing stripped sequences and
  peptide/site/charge precursor IDs with any finite raw measurement, exports
  numbered PEPTIDE_n/PRECURSOR_n columns, and rejects duplicate Summary headers.
- PELSA adds selected `PELSA_` curve fields and omits usual observed/consistency
  summary blocks. EC50 is derived from pec50; full fit diagnostics remain in H5AD.
- Proteomics NUM_UNIQUE_PEPTIDES is derived by ID-aligned drilldown sequence
  counts, not by var position. The exporter requires uns.peptides for this.

Kinase payloads currently remain in H5AD; `DEExporter` does not create kinase
workbook sheets. Do not claim all H5AD analysis products appear in table exports.

## PDF behavior and validation

`ReportPlotter` reads config plus AnnData; it directly requires var.GENE_NAMES.
It makes a title/preprocessing page, IDs/metrics, intensity heatmap, then limma
volcanoes if moderated q-values exist or the PELSA volcano. Its intensity heatmap
uses X with precomputed clustering orders, while clustering itself used a separate
QC-filled normalized matrix. Matching orders does not imply identical input values.

`_default_volcano_masks` implements stated viewer defaults using raw counts plus
stored missingness, separates observed-both/one-side-missing points, and excludes
both-side-missing points. Default minimum observed count is the smaller group
size, satisfied by **either** condition for observed-both points. It also applies
maximum per-condition consistency (default threshold 0), nrSC misalignment <=1.5
when available, and optional FT observed evidence (default larger group size,
satisfied by either side). These are display filters, not refits or FDR changes.
Current controls include `volcano_show_fully_imputed`, `volcano_min_per_condition`
and FT/consistency/misalignment settings; the unused `volcano_annotate_infinite`
setting was removed from templates.

`tests/contracts/test_results.py` checks representative workflow H5AD schemas,
independent evidence axes, viewer-consumed fields, scientific round-trip and
nonmutation, workbook headers/counts, QC capping and PDF evidence masks. Ten
workflow references protect numerical serialization semantics, including the
corrected slope p/q diagnostics and distinct FT workbook probabilities. See [testing.md](testing.md)
for the complete scope and limits. Optional metadata, PDF rendering and executed
viewer parity need further protection; schema changes
still require inspecting affected producers and actual consumers.
