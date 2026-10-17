# Phosphosite identity, protein adjustment, and kinase analysis

This workflow crosses `utils/harmonizer.py`, `workflow/dataset.py`,
`workflow/preprocessing.py`, `workflow/peptide_tables.py`,
`analysis/limma_pipeline.py`, and the kinase modules. A site-ID or missingness
change can affect all of them and the external viewer. Shared fitting behavior is
in [statistics-and-qc.md](statistics-and-qc.md); serialized layouts are in
[results-and-exports.md](results-and-exports.md).

## Declared sites versus localization candidates

`DataHarmonizer._build_phospho_index` currently expects Spectronaut-style PTM
information after mapping. Key fields are UNIPROT, PEP_SEQUENCE, PEPTIDE_LSEQ,
PEPTIDE_START, PTM_SITES_STR, PTM_POSITIONS_STR, PTM_PROTEINLOCATIONS and
PTM_PROBS_STR. The initial guard does not cover every field later dereferenced;
warnings about absent fields are not proof that an input can run without them.

The algorithm deliberately separates three concepts:

1. `PTM_SITES_STR` declares whether a row is phosphorylated; non-null/nonblank
   rows pass the initial gate.
2. `PTM_PROTEINLOCATIONS` supplies biological S/T/Y sites. For the primary site
   identity, use its first semicolon-delimited protein-group entry and the first
   peptide start. Absolute position minus peptide start plus one gives the
   **1-based peptide-local position**.
3. `PTM_POSITIONS_STR` contains localization candidates, paired positionally with
   probabilities. Mismatched list lengths produce null probabilities, not
   inferred matching. Values >1 are divided by 100 while the pairing is intact.
   Declared sites must match a candidate position; candidates are not all turned
   into biological sites.

Default `preprocessing.phospho.multisite_collapse_policy='explode'` (`duplicate`
alias) creates one row per matched declared site and duplicates the measured
signal without splitting its intensity. Rows with zero probability are excluded;
null probabilities survive this stage. IDs are `UNIPROT|S123`, peptide IDs are
`PEP_SEQUENCE|p<local position>`; parent protein is UNIPROT.

`retain` collects matched sites into a composite feature such as
`UNIPROT|S123,T126`, with a comma-separated local peptide index. Its LOC_PROB is
the **minimum** across retained matched site probabilities, despite a nearby
comment saying MAX. Do not substitute the explode and retain policies as an
equivalent reshape; they define different features and quantitative evidence.

`ACCESSION_SITES` preserves explicit semicolon-separated `ACCESSION:SITE` pairs.
The harmonizer retains `_PEPTIDE_STARTS_STR` before scalar coercion and aligns
accessions, starts and protein-location groups by position. It uses an
unambiguous single-accession fallback when those lists are unavailable. A
multi-accession group must not become a Cartesian product of accessions/sites.

`PHOSPHOFORM_POSITIONS` instead comes from phosphorylation tokens in the modified
peptide sequence (`_modified_phospho_positions`), not from localization candidates.
It is used to mark co-phosphorylated residues in kinase windows. The numeric PTM
map is limited; do not assume every vendor phosphorylation encoding is recognized.

## Localization filtering and precursor evidence

The main signal uses `peptide_rollup_method` even though features are sites.
`_pivot_main_block` takes max LOC_PROB within each site/run.
`_apply_phospho_localization_filter` then computes a site score across samples:

- Default `soft`: max finite per-run probability.
- `strict`: min finite per-run probability, ignoring missing samples.
- Default threshold 0.75, inclusive. Sites without any finite probability fail.

Filtering drops entire feature rows from every aligned main pivot. It does not
censor individual low-localization cells at this stage. Strict does not require
presence in every sample. Diagnostics store aggregation, threshold, unscored
counts and scores under preprocessing filtering `loc`.

Phospho precursor keys in `peptide_tables.py` are `PEPTIDE_INDEX/+CHARGE`.
Metadata includes SITE_ID and a `|`-joined set of phosphoform-position variants.
The implementation aggregates by precursor ID and retains the first SITE_ID;
shared peptide/site-local indices across proteins are a collision-sensitive case.
`Dataset._convert_to_anndata` removes precursor rows belonging to dropped sites
so orphan evidence cannot survive localization filtering. Export evidence counts
use precursors with at least one finite raw value; they are not just row counts
from the initial vendor file.

## Injected protein background and alignment

`Dataset._load_injected_runs` starts with main column mappings, applies a limited
`column_overrides` allowlist, and harmonizes each injection independently.
Separate-file injections default to no annotation or exclusions rather than
inheriting those of the main data. Indexing can be peptidomics even when the main
analysis is phospho. Each injection gets an assay/tag and `IS_COVARIATE` flag;
active flagged injections enable the covariate preprocessing/model path.

`reuse_main_input_file: true` is a restricted alternative:

- Only phospho, long main input, covariate true, peptidomics indexing.
- No simultaneous injected `input_file`, only one reused assay, and no additional
  active covariate source. A mapped `ptm_sites_column` is required.
- Reharmonize the original main input as peptides, select nonphosphorylated rows
  whose normalized PTM-site text is empty/NA/nan/none/null, require coverage of
  all main samples and some parent-protein overlap.
- Inherit main exclusions and, by default, main annotation. Append a sanitized
  `__<assay>` filename suffix. Store source provenance as
  `main_input_nonphosphorylated` in preprocessing metadata.

After filtering, `_get_condition_map` requires exactly one main and one covariate
run for each sanitized `(CONDITION, REPLICATE)` pair, with identical pair sets.
This is not alignment by filename, position, or by averaging all runs in a
condition. Duplicate or unmatched pairs fail.

`_pivot_data` rolls background peptides to `PARENT_PROTEIN` (exact group string,
not individually exploded accessions), default directLFQ, using
`phospho.covariate_protein_rollup_method`. It broadcasts parent values to retained
sites and renames/reorders covariate samples by validated run pairs to match main
columns. Unmatched parents give missing site rows. Covariate q/PEP are means;
spectral counts use explicit vendor counts if present, else precursor evidence.

Normalization occurs on this **broadcast** covariate matrix, so site multiplicity
can affect its normalization statistics. Imputation then groups to unique parent
proteins, imputes once per parent, and broadcasts back; separate draws for each
site would change the scientific model. Preprocessing subtracts the per-row median
across samples to form `centered_covariate`. Both centered and uncentered processed
background layers are retained. `qval_covariate` and `sc_covariate` are historical
stored names, not the main layer spellings.

## Protein adjustment in limma

`run_limma_pipeline_covariate` receives the same effective fit plan as ordinary
limma. Only `ancova.beta_mode=fixed1` is supported. `free`/`nonneg` and their dead
branches were removed by explicit user policy in the 1.10.4 cleanup. The user
reports comparing fixed, fitted and zero slopes; unit slope performed best while
fitted alternatives were unreliable/converged poorly. No benchmark dataset was
added to establish that claim independently.

The model mean-centers `centered_covariate` again, subtracts a feature intercept
and unit-slope background, then fits residuals. Original phospho Y and background
C use the same shared/separate fit plan. This is not a joint ANCOVA model.

For each feature/contrast, either side with zero raw background observations
selects the unadjusted phospho model. Pilot analyses bypass adjustment entirely.
All standard FC/SE/t/p fields follow this selection. Raw and moderated q-values
are each recomputed once over their selected finite p-vector per contrast.
Never splice q-values corrected in different families. Boolean
`varm.covariate_adjusted` (features × contrasts) records selection in nonpilot
analyses. Pilot outputs omit this field and p/q/t/SE arrays. Fully imputed nonpilot
phospho contrasts retain the separate p=1 rule.

`raw_log2fc/raw_p_ebayes/raw_q_ebayes` remain the independent unadjusted family;
`ft_log2fc/ft_p_ebayes/ft_q_ebayes` are the background family. The residual layer
and `uns.residual_variance` describe the residual model, not the selected fallback
model. `cov_part` is intentionally descriptive: the condition-mean difference in
processed log2 background/protein abundance, set to zero for fallback cells.
The processed values can include configured imputation; global centering cancels
in the mean difference. It is not defined as a fitted batch-adjusted coefficient
and is not required to exactly decompose batch-adjusted phosphosite logFC.
Without batches it also equals raw−adjusted coefficients. The existing independent
batch test protects this distinction as intended behavior, not an unresolved
defect. No model-based decomposition redesign is planned for 1.10.4.

### Slope diagnostics versus background contrasts

The separate global association diagnostic fits Y ~ 1+C across all samples,
even when differential fits are separate. `covariate_se/t/p/q` are features × 1;
`covariate_p` tests slope=0, not slope=1 or background differential abundance.
`covariate_beta` is the applied unit slope and is not the fitted diagnostic slope.
Diagnostics use processed values, with the existing scale-dependent ridge
(default 1e-6), N−2 residual df and near-flat guards (SD <0.1 log2 or low energy).
Flat diagnostics are missing. Fewer than three finite raw background observations
or no observed background variation also makes SE/t/p/q missing; numerical
imputation is not evidence. BH excludes missing diagnostic p-values. Contrast-local
no-FT support does not erase an otherwise estimable global diagnostic.

Independent scipy simple regression at ridge=0 and least-squares residual fits
protect the diagnostic/statistical meaning; they do not establish calibration of
ridge or imputation. All-pilot runs bypass this model. The old fitted-mode-only
crash/neutralization tests were retired with those unsupported modes; generic
fixed1 diagnostic, missingness and coherent fallback tests remain.

Dataset attachment still dereferences background q-values unconditionally;
optional input confidence-column support there remains a separate hardening item.
The previous covariate_p dimension collision and missing raw_p_ebayes export field
are historical repairs documented in the stabilization record.

## Shared kinase sequences

`main.run_pipeline` prepares sequences for every phospho result, even without a
kinase method. `analysis/kinase_sequences.py` writes:

| Field | Semantics |
| --- | --- |
| `var.ACCESSION_SITES` | Explicit accession/residue/absolute-position mappings |
| `var.KINASE_10MER` | PhosX window -5..+4, focal residue at index 5 |
| `var.SITE_15MER` | Export window -7..+7 |
| `uns.kinase_sequences` | Schema version 2, window/policy and FASTA/conflict metadata |

Peptide evidence comes from retained `uns['precursors']`. Unknown residues/flanks
are `_`; focal S/T/Y is uppercase; auxiliary phosphosites are lowercase s/t/y.
Auxiliary positions are the union of contributing phosphoforms. Consensus
conflicts become unknowns. Retained multisite composite features get all-underscore
windows, making them ineligible for PhosX rather than multiplying their signal.

Optional `analysis.kinase_activity.fasta` is a local file, validated early and
cached by resolved path/size/mtime. Accessions must resolve and focal amino acid
and absolute position must agree. One unique resolved window replaces the peptide
window; conflicting accession windows leave peptide evidence in place. FASTA
contents and full paths are not exported; no network lookup is performed.

## KSEA

`analysis.kinase_activity.method` defaults to disabled; supported methods are
`ksea` and `phosx`. `ksea_pipeline.py` reads one local TSV or an ordered list under
`substrate_database`, using explicit `database_columns` mappings. It requires a
kinase identifier, substrate identifier, and S/T/Y absolute site. Do not infer
columns by position or silently switch identifier systems.

- Kinase identity prefers gene, then UniProt, then name; kinase isoform suffixes
  collapse. Accession-only kinase entries are remapped to a gene only when
  unambiguous across databases. Substrate isoforms remain unchanged because site
  numbering can differ.
- A database row uses substrate UniProt if present; gene matching is a fallback
  only when that accession is absent, not when an accession match fails.
- `_build_feature_site_table` prefers ACCESSION_SITES with a legacy INDEX fallback.
  Gene/accession lists are paired only when alignment is supportable. Composite
  features with explicit mappings can enter KSEA but still count once per kinase.
- Database order picks duplicate display metadata, not statistical weight.
  Deduplicate relationships and finally kinase/feature assignments; retain source
  provenance. Cache reads by file signatures and mapping configuration.
- Input is standard `varm.log2fc` (adjusted with raw fallback if applicable),
  falling back to `raw_log2fc` only if the standard matrix is absent.
- Per contrast, background comprises all parseable localized features with finite
  FC, excluding features missing on both sides in raw. It includes sites without
  a database match and is not filtered by DE significance.
- For each matched kinase, `z=(mean_substrate_FC-mean_background_FC)*sqrt(m)/sd`,
  with background sample SD (`ddof=1`). Default minimum m is 5. Insufficient
  support/variance produces untested rows with reason and NaN statistics.
  Two-sided normal p-values and BH use tested kinases within each contrast.

Condition-level membership counts use sites observed (finite raw signal) at
least once in each condition; these are support summaries, not condition-level
activity tests. Results, assignments, condition/contrast/database summaries live
under `uns['kinase_activity']`, schema version 1. No complete substrate database
is embedded.

## PhosX adapter contract

`analysis/phosx_adapter.py` isolates private imports/resources for **PhosX 0.23.1**,
enforced at runtime and by the exact PyPI dependency pin. The adapter implements
in-memory scoring, assignments, seeded permutations and activation-evidence
adjustment. Do not replace it with the file CLI and assume numerical equivalence.

`phosx_pipeline.py` accepts valid 10-mers with uppercase central S/T/Y, finite
nonzero FC, and excludes both-side fully-imputed sites. Unknown `_` flanks are
allowed. It requires unique feature IDs, partitions S/T and Y, and sorts by FC
descending. Assignment row order must match that exact ranking when mapping
hits back to phosphosite IDs. A contrast with no eligible nonzero sites fails.

Defaults: 20,000 permutations, seed 1, one process, minimum 4 hits, top 5 kinases,
S/T quantile 0.95, Y quantile 0.90, both families, activation evidence enabled.
Legacy aliases include `n_perm`, `seed`, `n_proc`, `min_n_hits`; see `_parse_config`
for precedence. `phosx_resources` can override local PSSM, quantile and metadata
paths, but custom model IDs also need entries in `utils/kinases_mapping.py`.

Numerical details that must survive optimizations:

- PSSM products ignore `_`; quantile scaling uses `searchsorted(..., side='right')`.
  Binary assignments use strict `>` against the quantile/top-kinase threshold,
  so ties do not necessarily produce exactly five assignments.
- Weighted signed KS uses absolute FC weights. Maximum/minimum near-ties choose
  the negative extremum. Observed KS is rounded to five decimals.
- Empirical p-values use the tail selected by observed KS sign, strict
  inequalities, and `(count+1)/(permutations+1)`. BH is within each contrast **and
  family**, unlike KSEA. Precision/rounding is tied to permutation count.
- SeedSequence derives streams by contrast index, family and permutation;
  worker chunking preserves order. Changing contrast order can change null draws.
- Activity initially is signed capped `-log10(p)`. Optional upstream activation
  evidence changes activity using activation-loop adjacency and substrate
  redundancy; it retains `legacy_activity_score` and does not redo p/q.

PhosX shares the `kinase_activity` schema with KSEA but `kinase_id` remains the
unique PSSM model ID. Use `kinase_uniprot`/`kinase_gene` for cross-method joins.
`kinase_mean_log2fc` is the arithmetic mean across unique assigned quantified
sites; `effect` is KS and `z_score` is NaN. The local identifier map includes source
and checksum provenance; there is no runtime identifier lookup.

## Validation targets

`tests/unit/test_phospho.py` and `test_kinases.py` cover pairing/localization,
background alignment, parent-shared imputation, sequence windows, independent
KSEA and PhosX calculations. End-to-end references cover separate/reused/absent
backgrounds with KSEA and both PhosX families; the latter uses reduced permutation
counts for CI. Separate/reused observed backgrounds align, but input ordering can
change imputed draws; each has its own reference. `test_covariate_statistics.py` checks
slope p/q against independent simple regression, full-experiment scope, flat
backgrounds, label alignment and pilot omission. Fixed1 tests independently check
residuals and stage-2 coefficients/SE/p/q using NumPy least squares, and separate
diagnostic tests use SciPy regression. Mixed adjusted/fallback tests select all
inferential fields coherently and calculate BH on the final family. Fitted-mode
tests were removed with those unsupported modes. Contract tests protect slope
versus FT dimensions/probabilities through H5AD and FT workbook exports;
selection-scope workflow tests preserve uniform pilot status, absent inference
and raw descriptive FC whenever a selected condition is a singleton.
See [testing.md](testing.md) for fixture details, worker reproducibility and gaps.
These checks do not establish complete equivalence to external PhosX or execute
ProteoViewer; consumer compatibility was inspected read-only at a recorded revision.
