# Reading ProteoFlux results

The required H5AD output contains an AnnData object. XLSX and PDF are optional.
Keep the input files, YAML and dependency environment separately: the H5AD
metadata are useful provenance but are not a complete input/configuration archive.

```python
import anndata as ad
import pandas as pd

result = ad.read_h5ad("proteoflux_results.h5ad")
processed = pd.DataFrame(result.X, index=result.obs_names, columns=result.var_names)
```

## Matrix axes and scales

`X` and all `layers` are **samples × features**, labelled by `obs_names` and
`var_names`. Features can be proteins, peptides or phosphosites. Covariate runs
are aligned to main samples and stored in layers, not appended as extra samples.

| Field | Meaning |
|-------|---------|
| `X` | Normalized log2 analysis values, imputed if configured. |
| `layers['raw']` | Filtered/censored, rolled-up feature quantities before analysis normalization and imputation. Linear; not the original vendor rows and not AnnData's separate `raw` attribute. |
| `layers['lognorm']` | Values after the explicit log step, before later normalization. |
| `layers['normalized']` | Values after normalization, before imputation. Default source for PELSA and QC. |
| `layers['qvalue']`, `layers['pep']` | Identification confidence; not differential significance. |
| `layers['locprob']` | Phosphosite localization scores where available. |
| `layers['spectral_counts']` | Native distinct positive-signal precursor counts; not necessarily a copy of a vendor spectral-count column. |
| `layers['centered']` | QC-filled, feature-centered values; feature capping can leave other columns missing. Not the differential analysis matrix. |

Primary matrices are stored as float32. Analysis matrices use log2 units, recorded
as `uns['preprocessing']['quantitative_scale']`. For supported prelogged input,
raw quantities are reconstructed with `2**value`; see the
[configuration guide](CONFIGURATION.md#normalization).

A missing raw intensity, an identification-confidence NaN and an imputed analysis
value represent different evidence states. Censoring makes intensity missing
without necessarily removing identification evidence. There is no exported
universal imputation-mask layer; compare `normalized` with `X` on their labelled
axes for the analysis fills. `uns['missingness']` normally uses identification
q-values when available and therefore need not match raw-intensity missingness.

Spectral-count evidence is optional for analysis: absent layers omit nrSC outputs.
An existing all-zero count pair instead gives an undefined nrSC ratio, not an
absent layer or a fabricated zero statistic.

## Differential and background results

Differential `varm` arrays are **features × contrasts**, ordered by
`uns['contrast_names']`. `A_vs_B` means A minus B. Standard arrays include
`log2fc`, `se_raw`, `t_raw`, `p_raw`, `q_raw`, `se_ebayes`, `t_ebayes`, `p_ebayes`
and `q_ebayes`. Here `raw` in a statistic name means **unmoderated**, not linear
raw intensity. `pilot_study_mode` applies to the whole selected analysis. If any
selected condition has one sample, the flag is true and all inferential arrays
are omitted. Otherwise it is false; there are no mixed pilot/non-pilot outputs.
`uns['analysis']['pilot_contrasts']` contains all selected contrasts or an empty
list, respectively. A single condition has no contrast matrix. Missing statistics
must not be replaced with zero probabilities.

With phospho background adjustment:

- `raw_covariate`, `lognorm_covariate`, `normalized_covariate`,
  `processed_covariate` and `centered_covariate` are aligned layers. A parent
  protein's values are repeated across its sites; these are not independent
  protein measurements. `residuals_covariate` contains the stage-2 input.
- `raw_log2fc`, `raw_p_ebayes`, `raw_q_ebayes`, `ft_log2fc`, `ft_p_ebayes`, `ft_q_ebayes` and
  `cov_part` are separate contrast matrices. Unsupported site/contrast adjustment
  uses one coherent fallback model for FC/SE/t/p. Standard q-values are BH of
  the final selected p-vector per contrast, not copies of fallback-model q-values.
  `covariate_adjusted` is a boolean feature × contrast matrix identifying adjusted
  cells. The residual layer and `uns['residual_variance']` remain residual-model
  diagnostics. `cov_part` describes the difference in mean log2 background/protein
  abundance between conditions, using processed background values. By design it
  is not a fitted batch-adjusted coefficient and need not exactly decompose
  batch-adjusted phosphosite fold changes.
- `covariate_beta`, `covariate_se`, `covariate_t`, `covariate_p` and `covariate_q`
  have shape **features × 1**. Beta is the applied unit slope; the other fields
  describe the global stage-1 slope-zero diagnostic. Unestimable diagnostics stay
  missing and are excluded from diagnostic BH; see the
  [support rules](CONFIGURATION.md#phospho-background-adjustment).

Undefined raw p-values retain NaN in both p and q. BH uses the finite hypotheses
in that contrast, preserving feature positions. Moderated probabilities use
an independent family. Fully imputed contrasts have their separate explicit
neutralization rule (p/q=1), which must not be confused with an undefined test.

## Evidence, kinase and PELSA tables

`uns['peptides']` and `uns['precursors']` contain independently labelled evidence
arrays. Their `rows`, `cols` and associated identity fields must be used explicitly;
AnnData slicing does not automatically subset these payloads. Evidence raw values
are captured before analysis normalization. Their `centered` values are linear
ratios to row means, unlike the subtractive centering used in QC layers.

`uns['kinase_activity']` contains method metadata, results, substrate assignments
and condition summaries. Read `method`, `tested`, `reason`, `pvalue` and `qvalue`;
KSEA z-scores and PhosX activity scores are not interchangeable. PhosX keeps
`kinase_id`/`kinase_model_id` as unique PSSM model identifiers and supplies
`kinase_gene`/`kinase_uniprot` for biological annotation. Its FDR families are
separate for Ser/Thr and Tyr within each contrast.

`uns['pelsa']` contains `curve_points`, `curve_results`, fit metadata and
localization arrays. PELSA does not produce limma contrast matrices. Inspect
`fit_success`, controls, dose support, bounds, response direction, EC50 and
confidence intervals together. `localization['peptide_length']` counts residues,
excluding inline PTM annotation text.

## Workbook interpretation

`Summary` joins feature annotations, statistics and intensities. Stable prefixes
include `Processed_log2_Intensities<sample>`, `Raw_Intensities<sample>`,
`FT_processed_log2_<sample>`, `FT_Raw_<sample>` and `PELSA_` curve fields.
Additional sheets contain available identification/evidence, curve and kinase
information. The workbook README describes its sheets.

`Observed_<condition>` counts raw nonmissing intensities.
`Observed_<A_vs_B>` is the **maximum** of the two condition counts, not their sum
or minimum. Interpret it alongside the condition-specific columns.

For covariate-enabled phospho workbooks, `RAW_PVALUE_` and `RAW_QVALUE_` come from
the unadjusted phospho model (`raw_p_ebayes` and `raw_q_ebayes`). Earlier outputs
without `raw_p_ebayes` could export adjusted p-values under the raw prefix;
regenerate those outputs to obtain a matched unadjusted p/q pair.
