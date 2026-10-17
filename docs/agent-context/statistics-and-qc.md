# Differential modelling and QC

The active differential implementation is `analysis/limma_pipeline.py`, using
`inmoose.limma`; mechanical t-test/BH operations live in `analysis/stats_ops.py`.
`analysis/statisticaltester.py` is not used by `main.run_pipeline`. PELSA bypasses
limma entirely; phospho protein-background residualization is detailed in
[phosphoproteomics-and-kinases.md](phosphoproteomics-and-kinases.md).

LC-ConMed `mnar_only` can leave NaNs in analysis X. The standard adapter passes
these to InMoose without filling them; missing-design support depends on the
InMoose version. The covariate branch still uses ordinary NumPy sums/means in
stage-1 residualization and `cov_part`, so partial NaNs can propagate through
entire feature rows. Default QC fills a separate normalized-layer copy, while
direct `run_clustering(layer=None)` assumes complete X. These paths are unchanged
by the optional imputation mode.

## Inputs, pilot mode, and contrasts

`run_limma_pipeline(adata, config)` copies the input when assembling results.
`adata.X` is samples x features, normally normalized/imputed log2 signal. The
model converts it to a labelled features x samples pandas frame. `obs.CONDITION`
contains sanitized names, not necessarily the original annotation labels.

`_build_limma_fit_plan` resolves selected contrasts, then computes one pilot flag
from the union of participating conditions. It counts samples per CONDITION, not
REPLICATE labels. If any tested condition is a singleton, the entire selected
analysis is pilot, independently of separate_contrasts. A3/B3/C1 with A_vs_B alone
supports inference; adding A_vs_C makes both pilot. only_against=A includes C_vs_A
and therefore remains pilot. A single condition has no contrasts.

Pilot analyses bypass limma, batch and protein adjustment; they report raw
condition-mean differences. `uns.analysis.pilot_contrasts` is all selected names
or empty. `pilot_study_mode` is the authoritative uniform flag. Pilot outputs omit
inferential arrays and set has_covariate=False; mixed statuses are unsupported.
Single-condition output remains empty. Do not fill missing probabilities or
neutralize pilot cells. The fit plan shares this decision across raw/residual/
background paths; individual entries inherit it rather than deciding separately.

Pilot scope differs from variance-estimation scope: shared mode still includes
all experiment conditions, even an untested singleton; separate mode fits each
pair. Sample/evidence axes are retained. Normalization/imputation remain
experiment-wide. This separation is explicit user policy, not accidental scope
inference from the design.

`_contrast_defs_from_cfg` uses sorted condition levels. Precedence:

1. Nonempty `analysis.only_contrasts`, string or list, accepts `A_vs_B` or `A_v_B`.
2. `analysis.only_against=B` produces each other condition minus B.
3. Otherwise each ordered combination `a - b` from sorted levels.

Stored names always use `A_vs_B`, and positive log2FC means A minus B. Explicit
selection validates the supplied names against sanitized levels; it does not
sanitize the config values for the caller. Downstream code parses this separator
in masks, exports, kinase analysis and reports. Labels containing `_vs_` and
sanitization collisions deserve special care rather than a local rename. Preprocessing
now explicitly rejects distinct labels that sanitize to the same condition.

## Design matrix and fitting scope

`_design_with_optional_covariates` manually creates bare condition columns and
uses Patsy formula `0 + <conditions> + <batch dummies>`. This keeps condition names
available to `inmoose.makeContrasts`.

`analysis.batch_effect_columns` accepts a string/list, resolves columns
case-insensitively, rejects duplicate configured names and missing values,
and dummy-codes **all** specified factors categorically with `drop_first=True`.
Numeric-looking age/batch values are not continuous covariates in this code.
Partial null/blank batch values are rejected with affected sample names. Incomplete condition coverage warns; deficient
rank or residual degrees of freedom <=0 raises.

`_build_limma_fit_plan` defines two different statistical scopes:

- Default `separate_contrasts=False`: one experiment-wide design and fit,
  with selected contrasts applied afterward. Selecting only contrasts does not
  remove other samples from variance estimation.
- `True`: build and fit each pair independently using only its two conditions.
  Constant batch columns in that pair are omitted; the global mode instead
  rejects constant batch columns. Empirical Bayes runs separately per pair.
  Preprocessing still used the full experiment; this does not rerun normalization
  or imputation for each pair.

`uns['analysis']` stores design formula/columns/rank/residual df. Separate mode
uses `design_formula='per-contrast'`, no single rank/df, and a `contrast_models`
mapping. `uns['residual_variance']` is a feature vector in global mode and a
features x contrasts matrix in separate mode. Preserve both layouts.

`_fit_limma_with_contrasts` executes `lmFit`, `contrasts_fit`, then `eBayes` with
inmoose defaults. Raw SE is `stdev_unscaled * sigma`; moderated SE uses posterior
variance. Raw p-values are two-sided Student t with residual df; moderated
p-values come from inmoose. BH is column-wise across features, not across all
contrasts jointly. Changing selected samples, design coding, feature universe,
or the inmoose dependency can change numerical outputs.

## Observed evidence and fully-imputed cells

`_fully_imputed_mask_from_layer` examines **NaNs in pre-imputation `layers['raw']`**.
A feature/contrast is flagged only when every sample in **both** conditions is
missing. A fully missing single side is intentionally still analyzed. Values
elsewhere in the experiment do not prevent a pair from being flagged.

Flagged p-values are set to 1 **before BH** (raw and moderated). After fitting,
`_neutralize_fully_imputed_contrasts` sets standard outputs to:

| Output | Neutral value |
| --- | --- |
| `log2fc`, `t_raw`, `t_ebayes` | 0 |
| `p_raw`, `q_raw`, `p_ebayes`, `q_ebayes` | 1 |
| `se_raw`, `se_ebayes` | positive infinity |

`n_fully_imputed_cells` counts feature x contrast entries, not samples or features.
The fit itself and empirical Bayes still see the input matrix; masking is not
equivalent to removing features before modelling. Pilot mode returns before this
neutralization. Extended raw/FT/decomposition outputs have their own handling;
do not assume the standard output overwrite updates them all.

`analysis/missingness.compute_missingness` is a different operation: it chooses
QVALUE if present, else raw, else X and counts NaNs for every feature and condition.
`uns['missingness']` is a full-feature pandas DataFrame; `missingness_source` and
`missingness_rule='nan-is-missing'` identify provenance. Its docstring mentions
subsetting, but current code always uses the full feature set. This count can
disagree with raw-intensity missingness after low-signal censoring. Do not replace
it with a ratio, an imputation mask or a clustering subset.

`stats_ops.bh_qvalues` adjusts finite p-values separately in each column and
scatters them back to their original rows. Undefined p/q stay NaN; they do not
count in that family's BH denominator. This is distinct from intentional p=1
neutralization of fully imputed nonpilot contrasts. Raw/moderated families remain
independent. Empty finite families return all NaNs.

Canonical R comparisons live in `tests/reference/limma/`: balanced/shared/batch
fits agree with R 4.3.3/limma 3.65.1 in the reference Python environment. Equal
variance moderation and feature-specific rank deficiency are upstream limitations;
strict xfails remain and no local workaround is permitted.

## Spectral evidence metrics

`analysis/nrsc.py` reads native `layers['spectral_counts']`, sums counts across
each condition, and calculates `(a-b)/(a+b)`, with NaN if the denominator is zero.
It does not divide by condition sample count. Unbalanced replication therefore
matters. `nrsc_misalignment` actually computes
`1 - sign(log2fc) * nrsc` (0 aligned, 1 neutral, 2 opposed); its docstring's
displayed plus formula is stale. Arrays align to `contrast_names` and var order.
If the layer is absent, `compute_nrsc` logs a warning and returns `None`.
Standard, pilot and covariate limma callers omit `nrsc` and `nrsc_misalignment`
when analyzing fresh inputs without that layer. They do not fabricate counts or
change differential statistics. A present all-zero count vector instead yields
NaN nrSC (undefined 0/0); equal positive condition totals yield nrSC=0. H5AD
preserves zero counts, whereas absent evidence remains an absent layer/metrics
and produces no Spectral Counts sheet or NRSC columns in XLSX.

The 1.10.4 Issue 6 repair imports the existing `log_warning` helper; previously
the missing-layer branch raised `NameError` instead of returning `None`. No count
formula, numerical default or export schema was changed.

This optional AnnData-layer contract does not establish that source files can
omit peptide/charge metadata. Main counts are derived from positive sequence/
charge observations, not merely copied from an optional vendor count column.
An Issue 6 input audit found that `build_peptide_tables` invokes missed-cleavage
QC before its missing-column guard, explicitly rejecting absent `PEPTIDE_LSEQ`;
with sequence present but `CHARGE` absent, its proteomics precursor-statistics
selection raises Polars `ColumnNotFoundError`. Source-input optionality and that
charge-validation gap require separate review; no preprocessing bypass or
synthetic counts were introduced to reach the nrSC branch in tests.

## QC is separate from inferential imputation

`clustering_pipeline` always calls `analysis/clustering.run_clustering` on
`layers['normalized']`, capped by `analysis.clustering_max=8000`.
`_pick_feature_indices` selects highest observed variance by default, then sorts
the selected indices for stable slicing. Statistical results remain full-feature.

Clustering uses a scratch AnnData and `_lc_mindet_impute`, not the condition-aware
imputed X. Its global LoD uses adaptive lowest 1% bounded to 10..50 observations,
shift 0.20, jitter SD 0.05, seed 0; all-NaN input becomes zero. This is QC-only and
must not replace analysis X or normalized-layer missingness.

- PCA requests sample count minus one components from the pipeline. Current code
  caps by samples, not by the number of remaining features: tiny feature sets
  need validation.
- For <=3 samples, neighbors/UMAP/MDS are skipped with metadata reasons; PCA and
  hierarchical clustering remain. UMAP exceptions are swallowed, so absence of
  `X_umap` does not necessarily mean a fatal run failure.
- Neighbors use PCA, with a square-root sample-count heuristic bounded by
  10..30 and then `n_obs-1`.
- MDS uses correlation distance between samples in PCA space and random seed 0.
- Hierarchical clustering defaults to Ward/Euclidean. Intensity and centered
  linkages store sample and feature orders by name. The full-size
  `layers['centered']` subtracts each selected feature's sample mean from QC-filled
  values; unselected features are NaN-padded. This is not the peptide drilldown's
  division-by-mean transformation.
- Capped feature orders/linkages only cover their selected subset; sample
  embeddings are written back for every sample. Subset indices/names and
  selection metadata accompany these outputs.

`run_clustering_missingness` clusters the binary NaN mask of normalized data,
selecting highest binary variance `p*(1-p)` when capped. This subset is independent
of the intensity subset and of full-feature missingness counts.

When the fitted limma metadata includes batch factors, `run_design_adjusted_pca`
fits the design by pseudoinverse on the QC-filled normalized matrix and subtracts
only nuisance columns. Condition effects are retained. It stores
`X_pca_design_adjusted`/`pca_design_adjusted`; analysis X is unchanged. The caller
rebuilds a global design even after separate-contrast fits, which is relevant to
rank/degree-of-freedom failures in this diagnostic path.

Kinase QC is added for phospho results with `uns['kinase_activity']`.
`run_kinase_activity_clustering` keeps `significant` and `all_tested` profiles:
significance means tested, finite activity/q, q strictly below the threshold in
at least one contrast. Linkage-only gaps use contrast medians (or zero).
Kinases use `1-Spearman`, contrasts `1-abs(Spearman)`, average linkage. The old
top-level significant-profile fields are mirrored for compatibility. A PhosX
activity score is not a KSEA z-score despite sharing this display schema.

## Validation targets

`tests/unit/test_statistics.py` covers independent raw OLS, BH ranks, contrast
selection/variance scope, an unselected singleton, categorical batches/confounding,
missingness and nrSC, including absent versus zero spectral evidence and unchanged
standard/pilot statistics when counts are absent. `test_covariate_statistics.py`
checks the corresponding covariate branch; `tests/contracts/test_results.py`
checks absent/zero/observed counts through analysis and H5AD/XLSX export.
Integration references protect moderated outputs and pilot/
batch workflows; they are not external validation of empirical Bayes. Inmoose's
equal-residual-variance failure is a temporary expected-failure repair target.
`tests/contracts/test_results.py` checks QC nonmutation, capping and three-sample
omissions. Embedding coordinates/topology and every small-sample edge are not
frozen. See [testing.md](testing.md) for policy and gaps. PDF selection rules live
with [results-and-exports.md](results-and-exports.md), not in the statistical model.

The undefined-raw-p counterexample now passes with finite-only BH, unchanged
moderated probabilities and preserved missing positions. Canonical R references
independently validate balanced/shared/batch fits; equal-variance and partially
non-estimable cases retain explicit upstream failures. See
[testing.md](testing.md) for reference provenance and
[adversarial-audit-1.10.4.md](adversarial-audit-1.10.4.md) for remaining decisions.
