# Input preparation, quantification, normalization, and imputation

The active path is `workflow.dataset.Dataset` -> `utils.harmonizer.DataHarmonizer`
-> `workflow.preprocessing.Preprocessor`. Read all affected stages before changing
identifiers, null handling or quantitative transforms: their effects propagate to
statistical masks, evidence counts, QC and exports. Phosphosite construction and
protein-background handling have additional rules in
[phosphoproteomics-and-kinases.md](phosphoproteomics-and-kinases.md).

## Loading and the canonical long table

`Dataset._load_rawdata` accepts `.csv`, `.tsv`, `.parquet` and `.pq`. Polars is the
sole CSV/TSV backend; Parquet always uses PyArrow then converts to Polars.
Omit `dataset.load_method` or set it to `polars`. Other values, including null,
are rejected by the constructor before file I/O, also for Parquet inputs. The
obsolete PyArrow and pandas delimited paths have been removed. Injected files
use the same reader; PyArrow remains a required dependency for Parquet.
`_build_polars_schema_overrides` protects inference-sensitive statistical columns
and explicitly mapped semicolon-valued metadata. `_read_csv_polars` uses 10,000
rows for inference and up to five retries, forcing the named failing column to
text. Preserve values during parser repairs rather than silently dropping rows.

`DataHarmonizer.DEFAULT_COLUMN_MAP` maps configured vendor columns to `INDEX`,
`SIGNAL`, `QVALUE`, `PEP`, `FILENAME`, `CONDITION`, `REPLICATE`, `PEPTIDE_LSEQ`,
`CHARGE`, protein metadata and phospho fields. Missing mappings usually warn;
required-column failures can occur much later. Renaming over an existing canonical
column raises. The pipeline is not a vendor-name dispatcher: templates supply the
vendor-specific mappings.

Sample/annotation rules:

- `_normalize_run_name` trims whitespace and strips one recognized trailing
  `.raw`, `.d`, `.mzML`, `.mzXML`, or `.wiff`, case-insensitively. It does not
  extract a basename or lowercase the whole identifier. The same normalization
  is used for annotations and exclusions.
- Annotation is tab-delimited, with exact `Condition` and a join column chosen
  from `FILENAME`, `File Name`, `File name`, `filename`, `Filename`, `Channel`.
  Optional `Replicate` is cast to integer; missing values are filled by
  deterministic numbering after sorting by condition/filename.
- Annotation overrides existing condition/replicate columns. Unannotated kept
  data runs fail; extra annotation rows warn and are ignored. Exclusions apply
  before parity checks, so excluded runs need not be annotated.
- Without annotation, canonical condition and integer replicate must already
  exist for long data. Arbitrary extra annotation columns are joined but only
  selected metadata (including configured batch columns) reaches `obs`.
- Wide input identifies `<LABEL> <signal_column>` columns. Without annotation,
  all labels must match `^([A-Za-z]+)_(\d+)$`; replicates are renumbered within
  condition. If any label fails, every run gets `CONDITION='NA', REPLICATE=1`
  and the later analysis is a pilot. This is not a biological grouping inference.
  Optional spectral-count channel melting exists in this no-annotation branch;
  do not assume the annotation branch performs the same melt.

`_get_condition_map` preserves `CONDITION_ORIG` and sanitizes condition strings:
non-alphanumeric/non-underscore runs become `_`, leading digits get `C_`, and
empty labels become `C_UNLABELED`. Contrasts use the sanitized names. Distinct
original names that sanitize identically are rejected. Missing/blank conditions,
reserved model/annotation names, keywords and contrast separators are rejected. Replicate
alignment uses `ALIGN_KEY = CONDITION + '#R' + REPLICATE`.

Feature identity:

- Proteomics keeps configured `INDEX`, falling back to `UNIPROT` only if INDEX
  is absent. Protein groups remain groups; they are not generally exploded.
- Peptidomics/PELSA replace INDEX with normalized modified peptide sequence.
  `PARENT_PEPTIDE_ID` is stripped sequence; `PARENT_PROTEIN` normally comes from
  UNIPROT. PELSA additionally preserves original INDEX as a parent fallback.
  Same-sequence rows from different parents can therefore share a feature.
- `utils/sequence_ops.py` and `ptm_map.py` define numeric PTM conversion and
  sequence normalization. Methionine oxidation collapses by default, all PTMs
  do not. Collapsing changes identities and rollups; numeric conversion is not
  necessarily cosmetic. Do not apply PTM stripping globally to phosphosite IDs.
- Semicolon-valued IBAQ, protein weight and peptide start are coerced to their
  first entry in the harmonizer. Later metadata aggregation cannot recover the
  original group values. Phospho separately preserves aligned start strings.

## Stage order and missing values

`Preprocessor.fit_transform` calls `_filter`, metadata/condition extraction,
`_pivot_data`, `_normalize`, `_impute`, then builds `PreprocessResults`.
`IntermediateResults` carries Polars frames, arrays, models and metadata.
After `_pivot_data` freezes INDEX/sample order, its `add_df` checks exact row-index
equality; `add_matrix` only checks row count. It does not provide universal sample
order validation. AnnData conversion transposes these feature-by-sample matrices.

Filtering is separate for main/covariate blocks, in this order:

| Step | Constructor default and actual behavior |
| --- | --- |
| Contaminants | No files. With files, parse pipe-delimited FASTA accessions; split INDEX on `;`, strip `REV_` for comparison, drop a row if any member matches. This does not independently remove all reverse hits. Matching uses INDEX, not a universal parent-protein field. |
| QVALUE | Keep `<= 0.01`; absent column skips. |
| PEP | Keep `<= 0.2`; `pep_direction='higher'` reverses to `>=`. Absent column skips. |
| PRECURSORS_EXP | Keep `>= 1`; skip if absent, all null, threshold null or <=0. This is vendor evidence, not native spectral counts. |
| Low signal | `min_linear_intensity=None` disables. Otherwise replace SIGNAL strictly below threshold with null; retain the row and its identification metadata. |

Null QVALUE/PEP cells do not pass comparisons. NaN and Polars null are not
interchangeable throughout this pipeline; do not normalize them incidentally.
In wide peptidomics, zeros are excluded only from censoring *counts*, while still
being censored. Main/covariate filtering shares metadata keys, so a later covariate
pass can overwrite earlier filtering summaries; these are not per-assay ledgers.

`_pivot_df` first aggregates each feature/run and counts non-null input values.
All-null groups are restored to null before pivoting, then nulls become NaN.
This explicitly counters Polars sum/pivot behavior that would otherwise introduce
zeros. Its valid count tests nullness, not finiteness. Intensity is the primary
row order; auxiliary pivots are left-joined to it. Native spectral counts count
unique `PEPTIDE_LSEQ/CHARGE` with non-null, non-NaN, positive SIGNAL. QVALUE/PEP
pivots are means, LOC_PROB is max, IBAQ is mean. Changing intensity censoring can
leave QVALUE observed while intensity is missing.

## Protein and peptide rollup are different operations

`_pivot_main_block` selects protein rollup only for proteomics; the other three
analysis types use the peptide rollup (`sum`, `mean`, `median`). Generic protein
sum/mean/etc. aggregate long rows directly. `peptide_rollup_method` does not
automatically preaggregate every protein method.

- **Top3:** `_pivot_df_top_n` normalizes peptide identity, sums peptide signal
  within each run, then sums peptides with dense rank <=3. Ties can include more
  than three peptides. Its peptide summation does not use `peptide_rollup_method`.
- **directLFQ:** `_pivot_df_LFQ` aggregates positive signal by protein, literal
  `PEPTIDE_LSEQ`, and sample (charge is not part of this ion key). It sorts a
  pandas MultiIndex by protein and passes log2 values to the local
  `workflow/rollup/directlfq.py` wrapper, which uses upstream
  `directlfq.normalization.NormalizationManagerProtein`. The local code caps
  large protein groups at 100 ion rows, prioritized by missingness and then
  summed log intensities; summarizes shifted ions with a per-sample median;
  and rescales against their total linear signal. The wrapper returns linear
  protein values and restores missingness from its original-data mask instead
  of accepting upstream zero-filled outputs. `directlfq_min_nonan=1` is a
  per-sample minimum observed-ion count, not a generic pairwise shared-peptide
  threshold. Constructor cores default to 4; the caller fixes
  `num_samples_quadratic=2`. Local directLFQ configuration contains mutable
  protein/ion key globals; concurrent runs in one process deserve care.
- **Robust regression:** `workflow/rollup/robust_regression.py` starts from
  `build_proteomics_peptide_wide`, so this method does honor peptide rollup and
  PTM collapse. Positive finite observations become log2 responses in a sample
  effect plus sum-coded peptide effect model. Huber IRLS uses MAD scale (RMS/1
  fallbacks), tuning 1.345, 20 iterations, relative tolerance 1e-8. A connected
  sample/peptide graph uses structured normal equations; disconnected designs
  raise by default or fit each component separately on explicit opt-in.
  Sample coefficients are exponentiated to
  linear values, with `min_nonan=1` masking insufficient peptide evidence.
  This is its own numerical model, not an interchangeable sum/directLFQ method.

### Robust regression with disconnected observations

The observation graph links a sample to each peptide with a positive finite
measurement. With multiple components, the global sum-to-zero peptide constraint
does not uniquely separate sample abundance from peptide response in each
component. Within-component sample differences can be identified; a common
absolute scale between components cannot be inferred without extra assumptions.

`_observation_component_codes` assigns each positive finite observation to its
sample/peptide component. Unobserved samples/peptides do not create components.
`preprocessing.robust_regression_disconnected_policy` is validated by Preprocessor
and forwarded to `pivot_df_robust_regression(disconnected_policy=...)`:

- `raise` (default) raises `DisconnectedObservationError` with protein/sample
  groups before analysis/export. Invalid values, including null, are rejected.
- `component` warns with `DisconnectedObservationWarning` and fits each component
  with `_fit_rlm_numpy_irls_structured`, remapping its sample/peptide codes and
  imposing its own zero-sum peptide constraint. All observed sample coefficients
  are retained; evidence-count masking still applies. Each component also uses
  its own residual scale/Huber fit. The connected fit is unchanged.

Component estimates do not establish a common biological scale between components.
The pipeline does not automatically mask cross-component contrasts or restrict
variance estimation/normalization. Component membership varies by protein and is
not stored in H5AD/XLSX. The chosen policy is recorded under
`uns.preprocessing.quantification.robust_regression_disconnected_policy` for
proteomics robust rollup; it does not replace membership information. This option
does not select robust rollup for phospho backgrounds, whose separate method
currently supports sum/directLFQ.

Historically, disconnected designs used pivoted QR, potentially dropping sample
columns and changing the meaning of retained coefficients. The minimal observations
`(a, 8)` and `(b, 32)` yielded `[256, missing]`, then `[1024, missing]` after doubling
input: a confirmed scale-equivariance defect, separate from Issue 5's repaired
uninitialized solver. Explicit component mode now yields `[8, 32]` then `[16, 64]`.
Those are component-local quantities; their between-component ratio is unsupported.
Legacy explicit-solver/QR helpers remain but are no longer the rollup path.

## Evidence tables and QC counts

`workflow/peptide_tables.py` builds post-filter, pre-normalization drilldowns.
Protein peptide IDs are `INDEX|PEPTIDE_SEQ`; raw peptide values use configured
peptide aggregation. Its `centered` values **divide linear intensities by the row
mean**, unlike AnnData clustering centering or covariate centering.

Precursor tables also exist. Non-phospho precursor keys use the INDEX portion
before `|`, followed by `/+CHARGE`; phospho uses peptide/site position and charge.
Do not infer the stored `protein_index` meaning from its name; see the result
contract. Per-condition consistency requires non-null/non-NaN signal in every
represented run column of that condition; this is not the same as native
positive-signal spectral counts. Proteomics counts peptides; other analyses count
`INDEX/+CHARGE` precursor keys.

Missed-cleavage QC is computed from the filtered peptide universe, excluding
covariate rows, before localization row filtering. `_compute_missed_cleavages_per_sample`
implements a tryptic K/R-not-followed-by-P rule excluding the terminal residue,
deduplicates modified peptide IDs and exports unweighted and intensity-weighted
fractions. Peptidomics/PELSA gate on positive signal; proteomics/phospho do not.
Counts, drilldowns, native spectral counts and model intensities therefore need
not describe identical observation sets.

## Normalization and scale

`_normalize_matrix` processes the configured methods **in order** on feature x
sample matrices, separately for main and broadcast covariate data. A scalar
method is converted to a one-item sequence, used for both transformations and
log-step detection. There is no usable default sequence when `method` is absent.

NumPy arrays stay feature x sample throughout normalization and imputation.
Their Polars reconstruction explicitly uses `orient="row"` with the existing
sample-column list; feature INDEX stays attached to its row. This is essential:
Polars 1.36.1 otherwise interprets square Fortran-contiguous arrays as columns.
Before the 1.10.4 correction, main/covariate normalized frames and centered
covariate frames could silently transpose values (and NaNs) under valid labels
when retained feature count equalled sample count. Subsequent imputation,
statistics, QC and exports could therefore use incorrectly labelled values.
Exposed outputs include `layers.normalized`, `X`, `normalized_covariate`,
`processed_covariate`, `centered_covariate` and downstream residuals/statistics.
The correction changes affected values, not H5AD/workbook field names or shapes.
Standard list-form log transformations copied their post-log snapshot in C order,
and linear raw/identification/evidence tables were not transposed by this defect.
Raw-scale handling below is independent of this orientation defect.

- `log2` clips values below `1e-10`, preserves NaNs and snapshots `postlog`.
  `utils.semantics.log2_normalization_methods` enforces one initial log2 step
  for linear input. log10/quantile are rejected; downstream inference is log2.
- `median_equalization` shifts each column's observed median to the entire
  matrix's observed median.
- `median_equalization_intersection` estimates all shifts from the same features
  finite in every sample; skips if the intersection is empty.
- `median_equalization_by_tag` finds case-insensitive substrings in FASTA metadata
  (OR across tags), then uses only matching features finite in every sample.
  Empty matches/intersection skip; missing tag configuration/metadata raises.
  Changing to independently observed tag sets per sample alters normalization.
- `global_linear`, `global_loess`, `local_linear`, `local_loess` call
  `workflow/normalizers/regression_normalization.py`. Global reference is per-feature
  mean; local reference is the condition's first sample with global-mean fallback.
  Temporary row-median fills support fitting, then original NaNs are restored.
  Linear uses seeded RANSAC; LOESS span defaults to 0.9. The fitted sample-minus-
  reference trend is subtracted from the original data.
- `none` leaves model values unchanged. If no log step is detected, code assumes
  input was already log2 and exponentiates the corresponding raw frame (main or background).

With an explicit log step, raw means filtered/censored, feature-rolled signal
before normalization or imputation. `_normalize_matrix` copies its input into
`original_mat`; later analysis arrays and imputation copies do not alias that
raw evidence. Independent peptide/precursor tables are built before normalization.
Before the 1.10.4 scalar repair, log-step detection iterated the characters in
`"log2"`/`"log10"`, erroneously exponentiating raw values (often to infinity).
List forms were unaffected. With covariates enabled, the same erroneous branch
also overwrote main raw values and missingness with exponentiated background
values. The repair reuses the canonical method sequence; it preserves raw values
and masks without changing analysis transformations or serialization fields.

## Imputation

`workflow/imputer_factory.get_imputer` receives feature x sample values, an
explicit sample order and `condition_pivot`. Null/omitted method returns no
imputer; string `none` is not an alias. Do not transpose without examining each
algorithm: sklearn KNN sees feature rows as neighbors, whereas NSkNN explicitly
uses sample neighbors. Mean/median are row-wise and fill all-NaN rows with zero.
MinDet/MinProb estimate per-sample low quantiles; TNKNN uses feature correlations
and attempted truncated-normal estimation. These methods are not equivalent.

LC-ConMed (`workflow/imputers/lc_conmed_imputer.py`) is condition-aware and expects log scale.
The default `preprocessing.imputation.lc_conmed_mode: standard` follows:

1. Establish observed values using the **original** NaN mask. Imputed values
   never become evidence for later cells.
2. If the condition has at least `lc_conmed_in_min_obs` observations (default 1),
   draw around its median with `0.20 * pooled SD`, clipped to its 0.25/0.75
   quantiles. A singleton clips exactly to itself. Pool only groups meeting the
   threshold, with sample-SD and epsilon fallbacks in `_pooled_sd`.
3. Otherwise draw below the global detection limit: median of the lowest K
   observations minus 0.20 plus normal jitter SD 0.05, capped at the limit.
   Factory K defaults to 10; explicitly null K selects adaptive 1% bounded to
   10..50 and capped by available observations. All-missing input gets LoD zero.
4. Integer minimum takes precedence over fraction. To use
   `lc_conmed_frac_min_obs`, set the integer to null; the minimum is
   `max(1, ceil(fraction * condition_size))`.

Optional `mnar_only` restricts the fill mask to entirely missing feature/condition
blocks with observations elsewhere in that feature. Partial missingness and
fully missing features stay NaN, independently of the support thresholds.
It reuses the same LoD fallback; only eligible cells consume RNG draws. Original
raw/normalized masks, intermediate `imputed_only` and parent broadcasting remain
unchanged. The configured mode is carried in `uns.preprocessing.imputation`.

The factory seed defaults to 42. LC-ConMed preserves row-major draw order;
reordering features/samples or vectorizing RNG calls can change results with the
same seed. Covariate imputation occurs once per unique parent protein, then
broadcasts to all its sites; see the phospho document. Main `imputed_only` remains
an intermediate artifact; it is not an exported observation-mask layer.

## Current caveats and checks before a change

These are implementation hazards, not rules to preserve as scientific intent:

- Prelogged input has no log step and requires one row per final feature/sample.
  Repeated-row aggregation, directLFQ/robust/count rollup and non-null linear
  censoring thresholds are rejected. Linear-input rollups happen before logging;
  directLFQ/robust internal log fits still return linear quantities. Evidence
  tables/counts convert a copy of prelogged signal to linear units, so a log2 zero
  is an observed linear one. Original frames/files are not modified.
- `iterative_hgb` uses sklearn IterativeImputer with HistGradientBoostingRegressor.
  Old randomforest/rf_* configuration is rejected without aliases. Defaults are
  unchanged, with truthful hgb_* names (hgb_estimator_max_iter for boosting cycles).
- Empty quantitative samples are rejected after filtering and again before
  main/background imputation. Reconstruction checks exact shape. An all-missing
  feature is distinct from an empty sample; numerical low-level fallback methods
  must never create sample evidence. Preprocessing validates sequence/charge
  values before filtering; null/malformed sequences and nonpositive/noninteger
  charges fail clearly.

`tests/unit/test_preprocessing.py` and `test_imputation.py` protect pivots,
censoring/identification distinctions, rollup examples, labelled stage values,
reference normalization and LC-ConMed semantics. Workflow references cover
protein/peptide, phospho and PELSA paths. The demonstrated preprocessing defects
are repaired and have ordinary regression tests. Remaining strict expected
failures concern upstream InMoose; see the testing context.
`test_robust_rollup.py` independently checks the restored solver's linear equations
and Huber location solution, within-component fold changes with reordered sample
labels, and missing/minimum-evidence behavior. A full robust-rollup workflow
checks connected geometric-mean quantities, disconnected component ratios and
H5AD/workbook agreement; it does not validate disconnected absolute abundances.
`test_matrix_orientation.py` protects square/rectangular main and covariate
normalization, parent broadcasting/centering, and labelled AnnData/H5AD/XLSX
values; the original square-matrix expected failure is now a normal test.
`test_raw_provenance.py` independently checks censored/rolled raw quantities,
scalar/list log2, rejection of log10, analysis-buffer independence, normalization/imputation
choices, and main/background H5AD/XLSX evidence preservation. Its scalar-setting
regression is converted from the original expected failure.
See [testing.md](testing.md) for commands, reference policy and gaps including
alternative normalizers/imputers. Assert alignment as well as values; do not
silently fix scientific behavior during unrelated work.
