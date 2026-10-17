# 1.10.4 scientific stabilization review

## Final readiness review — 2026-10-13

**1.10.4 stabilization is ready for the user's review/commit.** This disposition
supersedes historical pending-item language below. A fresh complete
`python -m pytest -q --durations=15` run passed with **335 passed, 3 skipped,
3 strict xfailed in 29.42 s**, using the reference Python 3.12.7 environment.
This is one actual full-suite result, not an aggregate of partial runs. It includes
all five canonical R-limma cases and the reference-input provenance check; no R
runtime was invoked. No unexpected failures occurred.

Remaining strict expected failures, unchanged:

- `tests/unit/test_known_defects.py::test_equal_residual_variances_still_allow_moderation`:
  upstream InMoose infinite-prior-df KeyError(-2).
- `tests/contracts/test_limma_reference.py::test_limma_adapter_against_canonical_r[equal_variance]`:
  the same upstream failure against the canonical R numerical reference.
- `tests/contracts/test_limma_reference.py::test_limma_adapter_against_canonical_r[missing]`:
  upstream feature-specific singular-design LinAlgError; R retains estimable contrasts.

Optional skips, unchanged:

- `tests/full/test_local_regression.py::test_local_scientific_regression[disabled]`:
  requires an explicit `--full-regression MANIFEST`.
- `tests/integration/test_slow_phosx.py::test_phosx_permutation_stream_is_independent_of_worker_count`:
  opt-in worker validation, requires `--run-slow`.
- `tests/integration/test_slow_robust_rollup.py::test_disconnected_rollup_worker_results_and_errors_match_serial`:
  opt-in worker validation, requires `--run-slow`.

The review found no remaining documented, confirmed ProteoFlux correctness defect
in ordinary supported/default paths, apart from the explicitly retained upstream
InMoose limitations. This is the known-item disposition, not proof that no other
defect exists. PELSA validation, stochastic ordering, continuous batch support,
robust component provenance and other documented future hardening remain deferred.

`cov_part` is settled intentional design: a descriptive difference in mean log2
background/protein abundance between conditions, calculated from processed
background values. It is not a fitted batch-adjusted condition coefficient and
need not exactly decompose batch-adjusted phosphosite logFC. No redesign, new
field or numerical change was made. Historical suggestions to redesign it were
reclassified; repaired raw-provenance/validation findings and replaced API modes
are now explicitly historical or marked resolved.

README, CONFIGURATION, RESULTS and release notes agree on the supported imputers,
fixed1-only adjustment, uniform selected-analysis pilot status, log2 contract,
finite-only raw-p BH, upstream exceptions and descriptive cov_part semantics.
Only documentation changed in this final pass: CONFIGURATION, RESULTS, release
notes, and the agent-context adversarial audit, phosphoproteomics,
preprocessing, stabilization and testing documents. Production code, configuration,
tests, references, assertions, tolerances and xfail behavior are unchanged.
ProteoViewer was untouched. No commits, pushes, tags or publication were performed.

## Pilot-selection clarification — supersedes the pilot policy below

The user clarified that a result must never mix pilot and non-pilot contrasts.
`_build_limma_fit_plan` now decides **one pilot flag** from the union of conditions
in the requested contrasts. Any singleton in that union makes all selected
contrasts pilot. With A3/B3/C1, only A_vs_B supports inference; all contrasts,
A_vs_B plus A_vs_C, or only_against=A are entirely pilot. This applies in both
shared and separate fitting modes. only_contrasts takes precedence over
only_against, as before.

This deliberately separates pilot selection from variance-estimation scope:
shared fits retain all experiment conditions, separate fits use each pair, and
neither selection removes sample/evidence axes. All pilot results take the early
descriptive path without batch/background adjustment or inferential arrays;
pilot_contrasts is all selected names or empty. The plan's single flag also
controls its fit executor, removing mixed-result assembly.

The new desired-behavior tests first reproduced six failures. After the repair,
74 focused statistical/covariate/workflow tests passed in 10.02 s. The complete
normal suite then passed: **335 passed, 3 skipped, 3 strict InMoose xfails in
30.39 s**. No new skips or xfails were introduced. Tests independently calculate
OLS/SE/raw-p for the selected AB analysis and condition-mean pilot FC; cover
selection order, only_against and precedence; and check uniform status, labels,
absence/presence of inference through H5AD and protein/phospho workbooks.

The pilot reference fixture now selects A_vs_C alongside A_vs_B so its singleton
participates. Only contrast metadata and the added AC descriptive/count column
were amended. Original AB values and all preprocessing/evidence remain exact;
new FC was independently checked against stored condition means. Other references
are unchanged in this follow-up. Public and agent documentation now state the
uniform rule. ProteoViewer and InMoose were untouched; no commits or remote writes.

Exact files changed in this follow-up (relative to the preceding policy pass):

```text
README.md
docs/CONFIGURATION.md
docs/RELEASE_NOTES_1.10.4.md
docs/RESULTS.md
docs/agent-context/adversarial-audit-1.10.4.md
docs/agent-context/phosphoproteomics-and-kinases.md
docs/agent-context/results-and-exports.md
docs/agent-context/stabilization-1.10.4.md
docs/agent-context/statistics-and-qc.md
docs/agent-context/testing.md
proteoflux/analysis/limma_pipeline.py
tests/fixtures/workflows.py
tests/integration/test_workflows.py
tests/references/pilot.json
tests/unit/test_statistics.py
```

## Historical policy pass — 2026-10-13

This section records the policy pass before the pilot clarification above and
final readiness review. Its original mixed-pilot behavior was superseded; its
other repairs remain in place. It supersedes the supported-mode descriptions,
pending decisions and reference status in the older reports below. Work remains uncommitted on
`dev/1.10.4`. ProteoViewer was read-only; no dependency implementation was patched.
Public behavior is described in [the release notes](../RELEASE_NOTES_1.10.4.md),
[configuration guide](../CONFIGURATION.md) and [output guide](../RESULTS.md).

### Repairs and deliberate policy changes

| Boundary | Correction and independent evidence |
| --- | --- |
| Raw p/q missingness | Common BH helper excludes nonfinite p-values per contrast and restores original positions; undefined p/q stay NaN. Hand-ranked examples and a zero-residual feature test verify finite neighbors and unchanged moderated probabilities. |
| Quantitative scale | Reject log10, uniform quantile, repeated/late log2 and inconsistent declared AnnData scales. Linear aggregation precedes logging. Prelogged input must be preaggregated and cannot request linear censoring, directLFQ, robust or count rollup. A log2(4+8) counterexample protects against summing logs. |
| Prelogged evidence | Convert a copy to linear values for precursor/peptide evidence and native counts. Observed log2 zero is linear one, not a missing signal; independently compare equivalent prelogged and linear inputs, raw layers and evidence. |
| Phospho model selection | Copy FC/SE/t/p together from the selected model, then run BH on the final raw and moderated families independently. A mixed adjusted/fallback example checks every statistic and independent final-family BH. New boolean `varm.covariate_adjusted` records selection. |
| Slope diagnostics | Keep global features-by-one slope-zero diagnostics separate from applied fixed1 slopes and contrast-specific FT statistics. Insufficient observed support/variation yields missing diagnostics excluded from BH, not invented p=1. |
| Empty samples | Reject samples with no finite observed main or aligned-background quantities before imputation, identifying labels. Guard after filtering and before array reconstruction. Tests include KNN, iterative HGB, mean, median, LC-ConMed and no imputation. |
| Imputer API | Rename the unchanged IterativeImputer/HistGradientBoostingRegressor estimator to `iterative_hgb`; use truthful `hgb_*` settings. Reject `randomforest` and `rf_*` with migration guidance; no aliases. |
| Phospho API | Support only fixed1. Remove free/nonnegative branches and mode-only tests. Retain independent regression machinery for association diagnostics; no replacement fitted model. |
| Model scope | Construct pilot status inside the shared/per-pair fit plan and use it for raw, adjusted and FT fits. A3/B3/C1 tests compare separate A_vs_B inference to the actual AB subset, preserve pilot A_vs_C NaNs, and check labels/H5AD/XLSX. Shared fits still include C in their scope. |
| Input/configuration hardening | Reject partial missing batch factors, reserved/colliding conditions, self/duplicate contrasts, malformed/null peptide sequences and nonpositive/noninteger charges. Add the packaged PELSA CLI template; remove unused volcano_annotate_infinite from templates/examples. |

Intentional numerical changes are finite raw q-values beside undefined tests,
coherent fallback SE/t/raw-p and final-family q-values, missing unestimable slope
diagnostics, inference for replicated pairs previously suppressed by unrelated
singletons, and correct linear evidence/counts for prelogged input. Existing
ordinary linear-input/fixed1 fitting defaults are unchanged. Removed/incompatible
configurations now fail rather than silently producing mis-scaled output.

Sample/feature/contrast axes and existing workbook columns are unchanged. H5AD
adds `preprocessing.quantitative_scale`, `analysis.pilot_contrasts`, per-pair
`contrast_models[*].pilot`, and `varm.covariate_adjusted`. Mixed separate-fit
results retain inference arrays with NaN pilot columns; `pilot_study_mode` is
true only when all selected fits are pilot. Workbook values follow these results.
This is producer-side compatibility coverage, not a ProteoViewer UI validation.

`cov_part` is the descriptive difference in mean background/protein abundance
between conditions. A categorical batch example confirms that this differs from
a fitted batch-adjusted coefficient. The user subsequently confirmed this as
intentional design: exact batch-adjusted decomposition is not a requirement.
Its numbers are unchanged; no redesign is pending. Residual layers/variance
remain explicitly residual-model diagnostics, not fallback-model statistics.

Nine non-PELSA snapshots gained fit-scope metadata. Only the affected selection,
diagnostic and inference fields in three covariate snapshots were numerically
amended; unrelated original values were retained. X/layers/FC/kinase results and
the fixtures' moderated q<0.05 decisions did not change. These remain current-
behavior goldens, distinct from independently generated R references.

### Canonical reference and remaining expected failures

`tests/reference/limma/` contains five authored, deterministic cases: balanced
two-group, shared three-group, categorical batch, equal residual variance, and
partially non-estimable features. Each has expression, sample, design and contrast
inputs plus expected statistics. The 29 reference/provenance files total 75,565
bytes; no private input or H5AD/XLSX artifact was added.

The generator ran against the existing **R 4.3.3 / limma 3.65.1** installation.
Session information and input checksums accompany the results. Normal pytest/CI
read CSVs only; no R, rpy2 or Rscript dependency was added. Balanced/shared/batch
coefficients, residual df and raw/moderated statistics agree with InMoose 0.9.0
at rtol=1e-7, atol=1e-12, with exact labelled axes and NaN positions.

| Current strict xfail | Status |
| --- | --- |
| `tests/unit/test_known_defects.py::test_equal_residual_variances_still_allow_moderation` | Original input/assertion unchanged; upstream infinite-prior-df KeyError(-2). |
| `tests/contracts/test_limma_reference.py::test_limma_adapter_against_canonical_r[equal_variance]` | Same upstream exception; canonical R now supplies the independent numerical expectation. |
| `tests/contracts/test_limma_reference.py::test_limma_adapter_against_canonical_r[missing]` | Upstream feature-specific singular-design LinAlgError; R retains estimable contrasts and marks unsupported ones missing. |

The prior undefined-raw-p xfail now passes. The prior pairwise nonnegative xfail
was removed with the unsupported mode, not repaired or hidden behind another
configuration. The current three xfails represent two upstream limitations, not
three distinct ProteoFlux defects. No canonical reference was changed to match
InMoose and no local statistical workaround was introduced.

### Validation record

The previous audit baseline was 279 passed, 3 skipped, 3 xfailed in 48.69 s.
It was not rerun at the beginning of this pass. Focused validation used the
reference Python 3.12.7 environment, one numerical-library thread, temporary
outputs and plugin autoload disabled. No production-scale permutations or private
datasets were run.

| Check | Result | Elapsed |
| --- | --- | --- |
| Grouped policy/statistical/reference/workflow checks before final run | 137 passed, 2 expected upstream failures | 17.13 s |
| One complete normal suite, `python -m pytest -q --durations=15` | 319 passed, 4 failed, 3 skipped, 3 expected upstream failures | 22.68 s |
| Corrected affected subset: phospho unit module and serialized slope diagnostics | 12 passed | 9.06 s |
| Compact regression, PYTHONHASHSEED=1 | 27 passed, 2 expected upstream failures | 15.47 s |
| Compact regression, PYTHONHASHSEED=8675309 | 27 passed, 2 expected upstream failures | 16.45 s |
| Small PhosX one/two-worker check with --run-slow | 1 passed | 1.12 s |

The full run exposed four obsolete expectations, not a production defect: three
serialized slope checks fed NaNs to SciPy BH instead of selecting testable sites;
one localization fixture expected success after filtering an entire sample away.
The former now independently check finite-site BH and exact missing positions.
The latter now checks both valid localization and named empty-sample rejection,
with a separate aligned-background rejection test. No assertions were loosened,
tolerances widened or production code changed after the full run.

The final collection has 331 cases: **325 verified passing, 3 strict xfails and
3 optional skips**, combining the full run with the passing corrected subset.
This is not a claim that a second clean full run occurred. Following the requested
execution budget, only affected tests were rerun after the full run. Optional
skips are local full-data regression, slow robust multiprocessing and slow PhosX;
the last was separately executed as recorded above.

CI still runs Python 3.11/3.13 on pushes/PRs with cached constrained dependencies,
separate core/workflow steps and optional worker validation. New policy and
R-reference consumers were added to the core step. GitHub CI itself was not run
locally. Packaging/public minimum Python >=3.10 is consistent; 3.10 is not in
the scientific CI matrix. Changed production files parse with Python 3.10 grammar.

### Scope and remaining validation gaps

Reviewed public README, configuration/results guides, installation/version
declarations, all packaged templates and three example READMEs/configs. Updated
the relevant user guides, release notes, paper's phospho description and obsolete
template/example option. Example READMEs and dependencies needed no change in
this pass. Refreshed agent routing, subsystem context and the follow-up ledger.

Deferred: InMoose implementation, PELSA optimizer/calibration, universal stochastic
ordering invariance, continuous batch regression, robust component provenance/
between-component restrictions, and background confidence-column requirements.
cov_part is settled descriptive design, excluded from deferred work. Actual quantitative units cannot be inferred from
an unlabelled numerical matrix; direct callers must honor the log2 contract.

The exact task-specific file list follows. It is derived from the initial file
snapshot and the authoritative `git status --short`, including initially untracked
context documents. Pre-existing `pyproject.toml` edits were untouched. Pre-existing
`paper.md` edits were preserved; only the phospho behavior paragraph changed in
this pass. `.codex/`, `dist/`, `proteoflux.egg-info/` and
`proteoflux/quick_comparison.py` were untouched user artifacts. No commits, index
changes, destructive Git operations or remote writes were performed.

```text
.github/workflows/smoke-test.yml
AGENTS.md
README.md
docs/CONFIGURATION.md
docs/RELEASE_NOTES_1.10.4.md
docs/RESULTS.md
docs/agent-context/adversarial-audit-1.10.4.md
docs/agent-context/architecture-and-configuration.md
docs/agent-context/phosphoproteomics-and-kinases.md
docs/agent-context/preprocessing-and-quantification.md
docs/agent-context/results-and-exports.md
docs/agent-context/stabilization-1.10.4.md
docs/agent-context/statistics-and-qc.md
docs/agent-context/testing.md
examples/pelsa_small/config_pelsa.yaml
examples/searle_small/config.yaml
paper.md
proteoflux/analysis/limma_pipeline.py
proteoflux/analysis/pelsa_pipeline.py
proteoflux/analysis/stats_ops.py
proteoflux/analysis/validation.py
proteoflux/cli.py
proteoflux/export/pdf_report_exporter.py
proteoflux/templates/diann-peptidomics.yaml
proteoflux/templates/diann-proteomics.yaml
proteoflux/templates/fragpipe-peptidomics.yaml
proteoflux/templates/fragpipe-proteomics.yaml
proteoflux/templates/spectronaut-pelsa.yaml
proteoflux/templates/spectronaut-peptidomics.yaml
proteoflux/templates/spectronaut-phosphoproteomics.yaml
proteoflux/templates/spectronaut-proteomics.yaml
proteoflux/utils/semantics.py
proteoflux/workflow/dataset.py
proteoflux/workflow/imputer_factory.py
proteoflux/workflow/peptide_tables.py
proteoflux/workflow/preprocessing.py
tests/contracts/test_inputs_and_config.py
tests/contracts/test_limma_reference.py
tests/contracts/test_results.py
tests/contracts/test_scientific_policy.py
tests/integration/test_workflows.py
tests/reference/limma/README.md
tests/reference/limma/balanced/contrasts.csv
tests/reference/limma/balanced/design.csv
tests/reference/limma/balanced/expected.csv
tests/reference/limma/balanced/expression.csv
tests/reference/limma/balanced/samples.csv
tests/reference/limma/batch/contrasts.csv
tests/reference/limma/batch/design.csv
tests/reference/limma/batch/expected.csv
tests/reference/limma/batch/expression.csv
tests/reference/limma/batch/samples.csv
tests/reference/limma/equal_variance/contrasts.csv
tests/reference/limma/equal_variance/design.csv
tests/reference/limma/equal_variance/expected.csv
tests/reference/limma/equal_variance/expression.csv
tests/reference/limma/equal_variance/samples.csv
tests/reference/limma/generate_reference.R
tests/reference/limma/input_checksums.tsv
tests/reference/limma/missing/contrasts.csv
tests/reference/limma/missing/design.csv
tests/reference/limma/missing/expected.csv
tests/reference/limma/missing/expression.csv
tests/reference/limma/missing/samples.csv
tests/reference/limma/sessionInfo.txt
tests/reference/limma/shared/contrasts.csv
tests/reference/limma/shared/design.csv
tests/reference/limma/shared/expected.csv
tests/reference/limma/shared/expression.csv
tests/reference/limma/shared/samples.csv
tests/references/batch_pairwise.json
tests/references/peptidomics.json
tests/references/phospho.json
tests/references/phospho_reuse.json
tests/references/phospho_unadjusted.json
tests/references/phosx.json
tests/references/pilot.json
tests/references/proteomics.json
tests/references/searle_directlfq.json
tests/unit/test_covariate_statistics.py
tests/unit/test_imputation.py
tests/unit/test_phospho.py
tests/unit/test_preprocessing.py
tests/unit/test_raw_provenance.py
tests/unit/test_statistics.py
```

Authoritative working-tree status at review (includes pre-existing artifacts):

```text
 M .github/workflows/smoke-test.yml
 M README.md
 M docs/CONFIGURATION.md
 M examples/pelsa_small/config_pelsa.yaml
 M examples/searle_small/config.yaml
 M paper.md
 M proteoflux/analysis/limma_pipeline.py
 M proteoflux/analysis/pelsa_pipeline.py
 M proteoflux/analysis/stats_ops.py
 M proteoflux/analysis/validation.py
 M proteoflux/cli.py
 M proteoflux/export/pdf_report_exporter.py
 M proteoflux/templates/diann-peptidomics.yaml
 M proteoflux/templates/diann-proteomics.yaml
 M proteoflux/templates/fragpipe-peptidomics.yaml
 M proteoflux/templates/fragpipe-proteomics.yaml
 M proteoflux/templates/spectronaut-pelsa.yaml
 M proteoflux/templates/spectronaut-peptidomics.yaml
 M proteoflux/templates/spectronaut-phosphoproteomics.yaml
 M proteoflux/templates/spectronaut-proteomics.yaml
 M proteoflux/utils/semantics.py
 M proteoflux/workflow/dataset.py
 M proteoflux/workflow/imputer_factory.py
 M proteoflux/workflow/peptide_tables.py
 M proteoflux/workflow/preprocessing.py
 M pyproject.toml
 M tests/contracts/test_inputs_and_config.py
 M tests/contracts/test_results.py
 M tests/integration/test_workflows.py
 M tests/references/batch_pairwise.json
 M tests/references/peptidomics.json
 M tests/references/phospho.json
 M tests/references/phospho_reuse.json
 M tests/references/phospho_unadjusted.json
 M tests/references/phosx.json
 M tests/references/pilot.json
 M tests/references/proteomics.json
 M tests/references/searle_directlfq.json
 M tests/unit/test_covariate_statistics.py
 M tests/unit/test_imputation.py
 M tests/unit/test_phospho.py
 M tests/unit/test_preprocessing.py
 M tests/unit/test_raw_provenance.py
 M tests/unit/test_statistics.py
?? .codex/
?? AGENTS.md
?? dist/
?? docs/RELEASE_NOTES_1.10.4.md
?? docs/RESULTS.md
?? docs/agent-context/
?? proteoflux.egg-info/
?? proteoflux/quick_comparison.py
?? tests/contracts/test_limma_reference.py
?? tests/contracts/test_scientific_policy.py
?? tests/reference/
```

## Historical initial sequence

Initial sequence validated on 2026-10-02 on `dev/1.10.4`, source revision
`cb15ce3`, with pending CI/version/context changes. The user approved issues 1–7
sequentially. Follow-up repairs requested afterward are recorded below. 1.10.3 is
frozen; ProteoViewer was inspected read-only. No agent commits or remote changes
were made. Counts and unresolved-item wording below describe that historical
checkpoint; later repairs and current disposition are recorded above. They are
not the final release-readiness status.

## Approved repairs and historical exposure

| Issue | Finding and correction | Previous outputs/configurations at risk |
| --- | --- | --- |
| 1: orientation | Confirmed defect. Explicit feature-row orientation when rebuilding Polars frames from normalization/imputation arrays. | Square matrices with feature count equal to sample count could transpose asymmetric values/NaNs under valid labels. Main/background normalized values, imputation, centered background, statistics and exports could be affected. |
| 2: raw provenance | Confirmed defect. Scalar log settings now use the canonical method sequence for log detection. | Scalar `log2`/`log10` iterated as characters during detection, incorrectly exponentiating stored raw values, often to infinity. With covariates it could overwrite main raw with background values/masks. Equivalent list settings did not trigger this defect. |
| 3: slope diagnostics | Confirmed collision. Separate local names for stage-1 slope p/q and FT contrast p/q. | Non-pilot covariate analyses stored FT probabilities under `covariate_p/q`, with an extra contrast dimension. These fields now contain per-site stage-1 diagnostics. |
| 4: free/nonnegative ANCOVA | Confirmed control-flow and dimensionality defect. Suppress global slopes only for entirely imputed background; apply missing-FT fallback per contrast after fitting. | `ancova.beta_mode: free/nonneg` crashed before producing results. Fixed1 behavior is preserved. Newly executable modes exposed separate BH (follow-up 1 below) and deferred Inmoose failures. |
| 5: disconnected robust rollup | Confirmed crash. Restore the historical least-squares initialization of the explicit IRLS solver. | Rank-deficient/disconnected rollup called an uninitialized coefficient vector. Execution was restored; the separate scaling defect is addressed by follow-up 2 below. |
| 6: optional counts | Confirmed robustness defect. Import the existing warning helper. | Absent AnnData `spectral_counts` caused `NameError`. Absence now omits nrSC outputs; present zeros remain distinct from absence. Source metadata requirements are separate. |
| 7: old loaders | Reclassified as obsolete code. Remove PyArrow and pandas delimited branches; validate `load_method` before file I/O. | CSV/TSV now uses only Polars. Omit the key or set `polars`; other values, including null, are rejected even for Parquet. Retain PyArrow's Parquet reader and dependency. |

See the preprocessing, phosphoproteomics and export context for precise field
semantics. Raw means filtered/censored, feature-rolled signal before analysis
normalization/imputation when a log transform is configured; it is not a copy of
every original input row. Independent peptide/precursor evidence keeps its own
axes and provenance.

H5AD's intentional shape/meaning correction is `varm.covariate_p` and
`varm.covariate_q`: `features x 1 x contrasts` becomes `features x 1`.
The hypothesis is stage-1 slope=0 in `Y ~ 1 + C`, not a condition contrast;
`ft_p_ebayes`/`ft_q_ebayes` remain `features x contrasts`. Only these two diagnostic
fields were added/amended in the phospho, phospho_reuse and phosx references.
Their ridge/flat-covariate inferential calibration is not newly established.
The inspected viewer/workbook consumers use the separate FT probabilities.
No XLSX sheet/column contract was changed. Orientation and raw-provenance repairs
intentionally change affected cell values, not field names or labelled axes.
No production changes were made solely for testability.

## Validation after the initial numbered sequence

Linux, Python 3.12.7, reference numerical versions in `tests/constraints.txt`,
pytest 9.1.1. Tests used the intended Miniforge Python with pytest supplied from
a temporary directory, plugin autoload disabled, one numerical-library thread,
Agg plotting, and temporary test outputs. Timings exclude dependency installation.

| Command/check | Result | Time |
| --- | --- | --- |
| `PYTHONHASHSEED=0 python -m pytest -q --durations=15` | 216 passed, 5 strict xfailed, 2 optional skipped | 20.07 s |
| `PYTHONHASHSEED=1 python -m pytest -m regression -q --durations=5` | 19 passed | 14.40 s |
| Same regression command with `PYTHONHASHSEED=8675309` | 19 passed | 14.33 s |
| `python -m pytest tests/integration/test_slow_phosx.py --run-slow -q --durations=5` | 1 passed; 2,047 permutations exactly equal across 1/2 workers | 1.09 s |
| `python -m pytest tests/contracts/test_inputs_and_config.py tests/contracts/test_results.py -q --durations=5` | 57 passed | 14.45 s |
| Additional full `make_workflow("phosx", ...)` runs with 1 and 2 workers, read through H5AD | Canonical scientific payloads exactly equal; worker metadata correctly differs | 6.52 s |

The last check used the maintained authored fixture and scientific payload
extractor from a temporary script; it is additional release evidence, not a new
permanent test. It excludes the same QC coordinates/non-scientific metadata as
the documented reference strategy. The maintained slow test protects the
permutation stream itself. Existing deprecation/runtime warnings were retained,
including the multiprocessing fork warning; these checks do not establish
cross-platform process behavior.

Seven original xfail cases now pass (issues 1, 2, 3, both issue-4 modes, 5 and 6).
The obsolete loader xfail was replaced by supported-Polars configuration and
container tests. At that checkpoint the original Inmoose xfail and four additional
cases exposed by repairs remained strict expected failures. No tolerances,
assertions or goldens were weakened in final validation.

The ten small workflow references cover protein/peptide analyses, global pilot
detection, batch/pairwise modelling, separate/reused/absent phospho background,
KSEA, both PhosX families, PELSA, and reduced real Spectronaut/directLFQ data.
Additional full runs cover free/nonneg covariate modes and robust rollup.
See [testing.md](testing.md) for each fixture's purpose and independent checks.
The ten JSON references total 711,482 bytes; the largest is 229,293 bytes. No
private dataset or generated H5AD/XLSX artifact is required in Git or ordinary CI.

CI uses Ubuntu and the existing Python 3.11/3.13 workflow versions, pip caching,
CPU Torch, reference constraints, dependency checks and JUnit artifacts. Push/PR
jobs separate algorithms/configuration from workflow/serialization tests;
manual dispatch can enable worker-count validation. Large local datasets require
an explicit manifest and are excluded from normal CI. GitHub jobs have not been
executed during this local validation. Package Python >=3.8 and README >=3.9
declarations were inconsistent then; the later audit aligned both to >=3.10.

## Historical InMoose/limma reference investigation

**Originally blocked pending a canonical R limma calculation.** The later policy
pass generated R 4.3.3/limma 3.65.1 references for five cases; the upstream repair
remains deferred. This section preserves the original investigation. No InMoose
implementation or existing upstream expected-failure assertion was changed.

The open upstream issue list and relevant issue/PR bodies were read on
2026-10-02. Applicability is based on the inspected local integration:

- [Issue #181](https://github.com/epigenelabs/inmoose/issues/181) directly matches
  the failure mechanism: infinite scalar prior df, Python boolean inversion and
  a `KeyError: -2` during B-statistic construction. The report is for 0.9.1;
  this suite independently reproduces the same mechanism in installed 0.9.0.
  This explains the exception, but is not independent validation of the intended
  moderated numerical result. Do not adopt the issue's suggested workaround.
- [Issue #183](https://github.com/epigenelabs/inmoose/issues/183) describes an
  `lmFit` singular-matrix failure when an entire condition is missing for one
  feature. This is a separate risk if nonfinite values reach model input (for
  example, without imputation); it does not explain the finite equal-variance
  fixture. No new missing-data fallback was introduced here.
- [PR #177](https://github.com/epigenelabs/inmoose/pull/177) proposes robust
  moderation and states that the non-robust path stays unchanged. It is open;
  ProteoFlux calls the default non-robust path. Switching to robust moderation
  would be a scientific change, not a repair justified by this review.
- [Issue #170](https://github.com/epigenelabs/inmoose/issues/170) concerns
  `limma.model_matrix`; ProteoFlux instead builds its design with Patsy.
  [#164](https://github.com/epigenelabs/inmoose/issues/164) concerns an RNA-seq
  voom/logCPM workflow, whereas ProteoFlux passes transformed proteomics values.
  [#108](https://github.com/epigenelabs/inmoose/issues/108) requests workflow
  documentation. None supplies a numerical reference for this failure.
- [Issue #120](https://github.com/epigenelabs/inmoose/issues/120) reports pandas
  import/export problems around PyComBat, a different failure from the executed
  limma code here; a pandas downgrade is not justified by that report.

`analysis/limma_pipeline.py` converts AnnData samples x features to a labelled
features x samples DataFrame, builds a no-intercept condition design (plus any
categorical batch terms), calls `lmFit`, then `contrasts_fit`, then `eBayes(fit)`.
The installed eBayes defaults are `trend=False`, `robust=False`,
`proportion=0.01`, `stdev_coef_lim=(0.1, 4)` and `winsor_tail_p=(0.05, 0.1)`.
ProteoFlux extracts posterior variance, moderated t/p, and applies its own
per-contrast BH after fully-imputed probability masking. Global fitting uses
the full experiment; separate fitting moderates each pair independently.

The original equal-variance fixture has 24 identical features across nine
samples: A=`[-1,0,1]`, B=`[0,1,2]`, C=`[-2,-1,0]`. The design is `0+A+B+C`;
contrasts are A-B, A-C, B-C. Pre-moderation coefficients are -1, 1, 2, with
residual df=6 and residual variance=1 for every feature (up to roundoff).
Local traceback inspection verifies that `fitFDist` returns Python float
`inf` as prior df; `_ebayes` constructs Python bool `True`, inverts it to -2,
and attempts a column lookup on a 24 x 3 t-squared DataFrame. This happens
after the internal moderated t/p calculation but before eBayes returns.
ProteoFlux does not consume the B-statistic, yet the exception aborts the call.

The requested R reference was to retain these inputs, feature count, design,
contrasts and moderation settings, identify versions, and compare coefficients,
residual/posterior variance, prior/total df, t, p and per-contrast BH. Those
references now exist under tests/reference/limma. Bypassing B-statistics, changing
priors, jittering variances or modifying expectations to match InMoose remains
outside the authorized scope.

## Follow-up 1: fitted-mode diagnostic BH

**Confirmed correctness defect, repaired in user commit `04a380a`.** Both original
`test_fully_imputed_background_diagnostics_apply_bh_after_neutralization` cases
(`free`, `nonneg`) failed with `--runxfail`: 23 of 24 q-values disagreed with BH
on the final p-vector. The expanded tests also reproduced the error for multiple
fully imputed sites, while partial missingness and fixed1 controls already passed.

Stage 1 computed BH before setting fully imputed-background slope p-values to 1.
Only the corresponding q-values were replaced; other sites retained obsolete BH
ranks. The repair replaces that assignment with BH on the final full p-vector.
It preserves the site-level family, including neutralized sites, and changes no
model or imputation setting. This can affect non-pilot user analyses using
`ancova.beta_mode: free/nonneg` with entirely missing raw background whose imputed
analysis vector yields a non-neutral slope p-value before the guard. Default
fixed1 is unaffected. Normalized/imputed input and diagnostic flatness determine
whether a particular dataset exhibits a numerical difference.

Only `varm.covariate_q` changes, including its H5AD-stored values. Before/after
comparison on the reproducer found all other saved matrices, inference fields,
labels and contrasts exactly unchanged; all fixed1 values were unchanged.
For example, protein0's q-value changes from 0.8425678821 to 0.8826901622 in both
fitted modes. There was no q<0.05 crossing in this fixture; that does not guarantee
threshold stability in other data. No named workbook or inspected Viewer consumer
of this diagnostic was found; their separate FT/DE probabilities do not change.

Original assertions moved intact to ordinary tests in `test_covariate_statistics.py`
and were strengthened with H5AD round-trip checks. Nine further cases independently
use SciPy regression/BH for partial/multiple/all-missing backgrounds in all three
modes. Focused: **11 passed in 3.05 s**. Surrounding covariate/workflow/export tests:
**68 passed, 1 unchanged Inmoose xfail in 16.87 s**. Full suite, including the ten
workflow references: **227 passed, 3 xfailed, 2 skipped in 20.78 s**. No references,
tolerances or Inmoose code/cases changed. See [testing.md](testing.md) for the
individual original/new xfail ledger and the required follow-up review order.

## Follow-up 2: disconnected robust rollup and public policy

**Confirmed correctness defect, repaired by the user's component-fitting patch;
public configuration/tests/docs updated, awaiting review.** The previous pivoted
QR path could discard a sample coefficient, then interpret a retained reduced-
basis coefficient as a sample abundance. Minimal observations `[8, missing]` and
`[missing, 32]` produced `[256, missing]`, changing to `[1024, missing]` when the
inputs doubled. This was not a Huber weighting problem or a valid protein scale.

The patch gives each disconnected component its own zero-sum peptide constraint
and preserves its sample effects. Default `raise` rejects the protein with its
sample groups; explicit `component` warns and returns component-local estimates.
The independently calculated singleton case now gives `[8, 32]` then `[16, 64]`.
Connected fitting is unchanged. Separate component residual scales/fits and full
retention of supported sample coefficients intentionally replace QR pruning.

`preprocessing.robust_regression_disconnected_policy` exposes the function's
policy in YAML and Dataset/run_pipeline configuration. It defaults to `raise`,
accepts only `raise`/`component`, and is checked before file I/O. Three proteomics
templates explicitly retain that default; public configuration docs explain both
policies. The only additional production changes are configuration validation,
forwarding and policy provenance. The user's rollup implementation is unchanged.

H5AD adds `uns.preprocessing.quantification.robust_regression_disconnected_policy`
only for proteomics robust rollup. Matrix orientation, existing field layouts,
evidence tables and XLSX schema are unchanged. No component membership is exported
and no downstream guard restricts contrasts/variance fitting/normalization after
opt-in. The chosen policy is not evidence supporting between-component inference.
Historical disconnected robust-rollup quantities and downstream analyses may be
affected; the separate sum/directLFQ and phospho background rollups are unchanged.

The original scaling xfail moves to an ordinary parameterized test with its
twofold assertion intact, additional scale factors and independent baseline
quantities. Previous assertions requiring a QR-dropped sample are obsolete under
the explicit new policy and are replaced by stricter checks of every supported
sample. Unit checks cover sparse connections, three components, ordering,
within-component ratios, independent component rescaling and evidence thresholds.
The existing authored workflow now tests default/explicit rejection before output,
connected geometric means and explicit-component H5AD/XLSX values. Its component
case selects B versus A with pairwise variance fitting, avoiding unsupported C
comparisons; this does not imply universal component-safe inference.

Validation: focused **39 passed in 7.27 s**; surrounding preprocessing, config,
workflow and export tests **124 passed in 17.07 s**; full suite **253 passed,
2 unchanged Inmoose xfailed, 3 optional skipped in 22.02 s**. The new opt-in robust
worker test separately passed in **0.44 s** (one/two-worker values, order and error
propagation). Its addition accounts for the third default skip. No numerical
goldens, tolerances, Inmoose integration or its two cases were changed.

## Historical follow-up checkpoint: failures and validation gaps

Two strict xfails remained then: the original Inmoose fixture and the additional pairwise
nonnegative ANCOVA case reaching the same exception. The initial report had five;
follow-up 1 resolves the two BH cases and follow-up 2 resolves scaling, with their
original numerical assertions retained. Ordinary skips are optional PhosX worker
testing, robust-rollup worker testing and full local-data regression (no manifest
supplied). The robust worker case was explicitly run above; earlier PhosX worker
validation remains recorded in the initial report.

Findings still open at that checkpoint, since resolved:

- Genuine no-log covariate normalization overwrote main raw with exponentiated
  background values/masks. The second audit fixed it with independent raw/export
  regression tests; the later scale contract preserves prelogged provenance.
- Source inputs without charge reached a Polars missing-column exception. The
  second audit added early required-column validation, followed by sequence/charge
  value validation in the policy pass. Precursor-less input is unsupported;
  optional spectral-count layers do not imply that input format is supported.

Broader weak areas recorded then included external R moderation calibration
(now covered by five compact reference cases, with upstream exceptions), ridge diagnostics,
alternative normalizers/imputers, real redistributable DIA-NN/FragPipe inputs,
phospho pilot/extreme mapping cases, larger kinase permutation runs, PELSA
optimizer extremes, viewer execution and multi-platform numeric reproducibility.
These are validation limits, not reasons to accept or work around known bugs.
