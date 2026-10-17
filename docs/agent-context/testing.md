# Scientific testing

The suite protects scientific behavior, labelled data and compatibility, not a
coverage percentage. Read the relevant subsystem context and implementation
before changing a test. No production algorithm was changed to establish this
suite. The initial reference revision is `297ca9b`.

The second audit and subsequent user policy decisions are tracked in
[adversarial-audit-1.10.4.md](adversarial-audit-1.10.4.md). Current behavior is
specified in the public configuration/results guides and subsystem context.
Canonical R references now exist under `tests/reference/limma/`; no R dependency
is introduced into ProteoFlux or normal tests.

## What constitutes evidence

Three complementary kinds of assertion are deliberately separate:

- **Invariants:** labels stay attached to values; observed data survive imputation;
  all-null sums remain missing; parent-protein imputation is shared by sites;
  QC leaves analysis values unchanged; serialization preserves scientific fields.
- **Independent numerical checks:** NumPy least squares and Student t probabilities
  for unmoderated limma; hand-calculable rollups/normalization and BH ranks; KSEA
  normal-tail probabilities; dense weighted KS walks for sparse PhosX; linear-mean
  controls, analytic 4PL/Jacobian checks and the specified PELSA recalibrated F test.
  These do not prove equivalence to every upstream statistical implementation.
- **Current-behavior references:** ten JSON snapshots from real
  `main.run_pipeline` runs, read back through H5AD. They freeze retained features,
  numerical stages and results, with independent checks alongside them. They are
  explicitly labelled snapshots, **not independently validated scientific truth**.

Complete PELSA optimization and real PhosX resource integration rely substantially
on the third category. Limma balanced/shared/batch fits additionally agree with
independent R 4.3.3/limma 3.65.1 references; two upstream edge cases remain blocked. A passing snapshot alone does not
establish that a newly discovered behavior is correct.

## Organization and risk boundaries

Paths below are relative to the repository root.

| Tests | Protected behavior |
| --- | --- |
| `tests/unit/test_preprocessing.py` | Null/zero pivots, dense Top3 ties and precursor-first sums, connected robust regression, directLFQ scale/missing masks, independent stage values, shared/tag references, evidence ratio centering |
| `tests/unit/test_robust_rollup.py` | Independent solver checks; default rejection and opt-in component quantities, scaling, within-component ratios, labels, sparse connectivity and evidence masks; no claim of between-component comparability |
| `tests/unit/test_matrix_orientation.py` | Square/rectangular labelled normalization, parent imputation/broadcast/centering, original 9x9 regression and exact AnnData/H5AD/workbook cell identities |
| `tests/unit/test_raw_provenance.py` | Independently calculated filtered/rolled quantities, scalar/list log settings, raw/evidence preservation across normalization/imputation, buffer independence and main/background H5AD/XLSX exports |
| `tests/unit/test_imputation.py` | LC-ConMed condition alignment, original observed mask, integer/fraction precedence, seeded draws, all-missing LoD, pooled sample variance; simple mean/median |
| `tests/unit/test_statistics.py` | Global versus per-pair variance scope, categorical batches, confounding, uniform pilot status from selected conditions, contrast direction, pre-BH both-side masking, full-universe missingness, nrSC and absent-versus-zero evidence |
| `tests/unit/test_covariate_statistics.py` | Independent fixed1 residual fits, global slope diagnostics, coherent per-cell fallback and final-family BH, missing diagnostics, descriptive cov_part distinct from batch-adjusted coefficients, labels and pilot omission; distinct FT probabilities |
| `tests/unit/test_phospho.py` | Declared site/probability/accession pairing, retain policy, localization/orphan filtering, background pairing and once-per-parent imputation, focal/auxiliary sites and FASTA extension |
| `tests/unit/test_kinases.py` | KSEA support/deduplication/background/FDR, accession-before-gene matching; PhosX weighted KS, tie rule, PSSM quantiles and seeded nulls |
| `tests/unit/test_pelsa.py` | Linear controls, pseudo-dose placement, distinct-dose eligibility, 4PL sign/midpoint/Jacobian, recalibrated significance and excluded zero SSE |
| `tests/contracts/test_inputs_and_config.py` | CSV/TSV/Parquet/pq with default/explicit Polars settings, early rejection of unsupported loaders, Spectronaut/DIA-NN long and FragPipe wide mappings, required metadata/optional confidence columns, annotations/exclusions, aliases, distributed validation, template versus constructor defaults, CLI template bytes |
| `tests/contracts/test_results.py` | H5AD axes/evidence/schema, viewer-consumed columns, scientific round-trip/nonmutation, optional spectral evidence through analysis/H5AD/XLSX, workbook names/counts, QC cap and three-sample handling, PDF volcano evidence masks |
| `tests/contracts/test_reference_comparison.py` | Comparison machinery detects drift, NaN changes, small probabilities and significance-boundary changes; optional full comparison handles specialized tables |
| `tests/integration/test_workflows.py` | Ten reference workflows plus pilot-selection protein/phospho and robust-rollup runs; independent quantities/component ratios, background decomposition/default-ridge fitting, PELSA recovery, family-specific PhosX BH; CLI working-directory paths and legacy export block |
| `tests/integration/test_slow_phosx.py` | Explicit one/two-worker permutation reproducibility |
| `tests/integration/test_slow_robust_rollup.py` | Opt-in one/two-worker component estimates, protein ordering and disconnected-error propagation |
| `tests/unit/test_known_defects.py` | Temporary strict expected failures described below |
| `tests/full/test_local_regression.py` | Explicitly opted-in local experiments against supplied H5AD references |

`unit`, `contract`, `integration`, `regression`, `known_bug`, `slow` and `full`
markers are declared in `pyproject.toml`. A file can belong to multiple categories.
Ordinary pytest includes all but slow/full cases. The workflow fixture caches each
run within a pytest session; copy its AnnData before mutating it.

## Fixtures and end-to-end coverage

`tests/fixtures/workflows.py` creates inputs/configs in temporary directories.
Most are authored experiments with fixed seeds and deliberately heterogeneous
residual variance. Rows are scrambled to expose positional alignment errors.
Every workflow runs preprocessing, analysis, QC, H5AD and XLSX export without
mocking scientific stages. PDF rendering is not part of these workflows.

| Reference/fixture | Samples x retained features | Purpose |
| --- | --- | --- |
| `proteomics` | 9 x 12 | Sum rollup, log2/median, LC-ConMed, three-condition limma; partial/all-group missingness, zero censoring, q/PEP rejection and distinct identification evidence |
| `peptidomics` | 9 x 24 | Same authored experiment at peptide resolution, precursor evidence and peptide result/export schema |
| `pilot` | 7 x 12 | A_vs_B and A_vs_C selected together: singleton C makes both pilot; no fabricated p/q-values |
| `batch_pairwise` | 9 x 12 | REPLICATE as a categorical nuisance factor, explicit B/C versus A contrasts and separate variance fits |
| `phospho` | 9 x 12 | Separate renamed flowthrough runs; two sites per parent, localization filtering, partial/missing protein background, fixed1 adjustment/fallback and KSEA |
| `phospho_reuse` | 9 x 12 | Background selected from unmodified rows of the same input; independent reference for its RNG/feature order |
| `phospho_unadjusted` | 9 x 12 | Phospho/KSEA without background injection; no fabricated covariate layers, with FC checked against the adjusted workflow's unadjusted fit |
| `phosx` | 9 x 12 | S/T and Y families, pinned real PSSMs, activation-enabled analysis, assignments, within-family BH and shared export schema; 127 seeded permutations for CI, not research-level p-value precision |
| `pelsa` | 10 x 16 | Two replicates of five doses, 16 authored curves with signed effects/known generating EC50, partial/absent controls, fitted/failed/absent results, parent counts and curve export |
| `searle_directlfq` | 9 x 20 | Runtime reduction of the already tracked `examples/searle_small` input: 16 complete and four sparse proteins, actual Spectronaut mappings, directLFQ, tag normalization and LC-ConMed |

Four additional end-to-end cases reduce the protein/phospho builders to A3/B3/C1.
Selecting only A_vs_B retains inference; adding A_vs_C makes the whole analysis
pilot with no inferential arrays or workbook p/q columns. Labels and descriptive
FC are checked through H5AD/XLSX. Unit tests cover both shared and separate fitting,
only_against, selection precedence and unchanged variance-estimation scope.
Unsupported free/nonneg workflow tests were
removed with those modes; independent fixed1 least-squares tests retain the
meaningful mathematical checks.

Robust-rollup workflow cases reuse the authored protein experiment with connected
observations or one protein restricted to disconnected sample groups. Default and
explicit `raise` stop before exports; explicit `component` is forwarded through
preprocessing and recorded in H5AD quantification metadata. Successful runs check
geometric means (including the component-local singleton-peptide estimates),
within-component ratios and H5AD/XLSX raw values. The component fixture selects
B versus A and fits their variance separately, avoiding its disconnected C group
in inference. This is a fixture constraint, not automatic pipeline enforcement.
No fixture files or goldens are added, and no cross-component fold change is
endorsed. Independent unit checks retain the original scaling assertion and add
multiple scale factors, sparse components, ordering and minimum-evidence checks.

Optional spectral-count checks cover the AnnData analysis interface in standard,
pilot and covariate modes, with matching statistical results. Contract tests use
fresh Dataset output with absent, zero or observed count layers and run real
analysis/export. They do not claim source-file-to-output coverage for absent
counts: preprocessing requires peptide sequence and charge before QC/evidence
construction; the adversarial audit added an explicit required-column guard.
Trial source files omitting those
fields fail earlier, so the assumption that they would exercise nrSC was incorrect.
See the statistics context for this separate input-validation boundary.

Separate and reused phospho backgrounds agree on observed values. Their imputed
draws can differ because first-seen feature order changes RNG consumption. The
tests protect each result separately and do not assert unverified equivalence.
All sites of a parent must still receive exactly the same parent imputation.

The Searle reduction selects complete/sparse identifiers deterministically and
copies only required columns into a temporary file. No new copy of its source
data is versioned. Local private/larger data and historical result files were
inspected for workflow/schema context but were not redistributed or adopted as
validated references; their precise producing code/config was not established.
The new reference JSON files total less than 1 MB; no H5AD/XLSX outputs are added.

`tests/fixtures/viewer_contract.json` records consumed field names and source
symbols from a read-only audit of ProteoViewer revision `4a20d0a`. Tests require
neither that checkout nor Streamlit. The JSON and workflow builders can support
future consumer tests. This is a producer contract, not an executed viewer UI
compatibility suite. ProteoViewer must remain read-only during this work.

## References, tolerances and intentional changes

`tests/scientific_reference.py` canonicalizes sample/feature labels while retaining
contrast order and pairing every matrix with its correct axes. It handles uns
evidence using its own row/sample labels, not AnnData slicing. References include
X, non-QC layers, inference arrays, obs/var, analysis metadata, residual variance,
missingness, evidence and consistency tables, kinase tables and PELSA curves.
NaN/Inf locations, table schemas, identifiers and q<0.05 masks are exact checks.
Ordering is tested as a labelled association, not an incidental filesystem order.

Paths/timestamps, PCA/UMAP/MDS orientation, clustering linkages and the QC-centered
analysis layer are excluded from numerical goldens. QC input preservation and
selection/shape invariants are asserted separately. Evidence-table `centered`
ratios **are** compared. Corrected stage-1 `covariate_p` and `covariate_q` are now
included with probability tolerances; their reference amendments are explicitly
recorded. Independent ridge=0 simple-regression tests establish the diagnostic
meaning, while default ridge/flat behavior also has snapshot protection. Not
every optional preprocessing/diagnostic metadata key is frozen.

- Ordinary stored values use `rtol=3e-6`, `atol=2e-6`, allowing float32 storage and
  small BLAS differences without accepting material intensity/FC changes.
- Probabilities use `rtol=3e-5`, `atol=1e-12`, including raw/FT/adjusted arrays and
  table p/q columns. Tiny p-values must not inherit intensity absolute tolerances.
- PELSA curve parameters/diagnostics use `rtol=1e-4`, `atol=2e-4` for CPU optimizer
  variation; their p/q use `rtol=1e-4`, `atol=1e-12`. Independent recovery checks
  additionally assess sign and EC50 against the generating model (25% tolerance
  because finite pseudo-controls and bounded fits approximate that model).
  `optimizer_best_start` is an ordinal diagnostic, not a scientific quantity:
  both reference comparators check integer indices within the configured starts
  (and the existing zero sentinel for unfitted rows), rather than exact equality.
  The column remains in the stored reference and H5AD schema. The CPU-kernel CI
  failure was reproduced with `ATEN_CPU_CAPABILITY=default`: starts 0 and 3 for
  ACDEFGHIK differed in SSE by about 1e-10, switching the winner while all fitted
  parameters, errors and p/q remained within the unchanged tolerances. Unit tests
  independently check minimum-SSE selection and candidate reordering; comparator
  tests ensure this exception cannot hide numerical drift or invalid indices.
- Unit tolerances are specified for each independently calculated example, often
  tighter. Dependency/platform changes require examining deviations; these bounds
  are not a guarantee of equivalence across arbitrary library versions.

Never regenerate references merely because a test fails. Inspect the first
changed stage, its labels/missing mask, the mathematical expectation and the
producing code. For an intentional scientific change, first add a small independent
regression test, explain which results should change, and generate **candidates**:

```bash
python -m tests.generate_references --output-dir /tmp/proteoflux-candidates
# Or restrict to one affected fixture:
python -m tests.generate_references --output-dir /tmp/proteoflux-candidates --workflow phospho
diff -u tests/references/phospho.json /tmp/proteoflux-candidates/phospho.json
```

The generator records revision/dependency provenance and refuses the reference
directory as its output. Pytest never updates references. Copy reviewed candidates
explicitly only after explaining numerical, feature-set, missingness, FDR and
schema changes. Include fixture and dependency changes in that review.

## Existing defects are temporary exceptions to fix

`test_known_defects.py` and `test_limma_reference.py`
assert desired behavior and mark only demonstrated
failures with `xfail(raises=...)`. `xfail_strict=true` makes an unexpected pass a
failure requiring review. These markers are a **temporary repair list**, not
accepted behavior or a normal testing pattern. Fixing the defect and removing
its marker is the intended resolution. Production workarounds should be
exceptional and explicitly justified; do not add skips, broad exception handling,
looser tolerances or alternate scientific paths to make this list look green.

| Defect demonstrated by tests | Current failure / repair boundary |
| --- | --- |
| Infinite prior df in moderation | Original unit reproducer remains a strict KeyError xfail. The canonical R equal_variance case independently confirms finite expected moderation and is also a strict KeyError xfail. |
| Missing/non-estimable feature | Canonical R missing case is a strict LinAlgError xfail: InMoose cannot fit a feature whose B observations are all missing; R retains estimable contrasts. |

The undefined-raw-p xfail is now an ordinary regression: finite-only BH is explicit
user policy. The pairwise nonneg xfail was retired because that scientific mode
was removed, not because InMoose was repaired. No local dependency workaround.

For a bug fix, run its existing desired-behavior test with `--runxfail` before and
after the change. Then remove the marker, cover nearby affected cases and run the
relevant workflow references. If fixing a previously excluded field, add it back
to the scientific payload and contract assertions with reviewed expectations.
These defects were reproduced but deliberately not repaired in the initial
testing task. Subsequent repairs follow the review sequence below.

## 1.10.4 stabilization review sequence

The original user-reviewed sequence below is historical; its one-at-a-time
approval procedure was superseded by the later adversarial audit request.
Continue on `dev/1.10.4`; 1.10.3 is frozen. Reproduce/classify defects, retain
independent tests, report numerical/schema effects. No commits or publishing;
ProteoViewer remains read-only. Historical issue order:

1. Square matrix orientation: corrected with explicit feature-row reconstruction;
   original xfail converted, square/rectangular/serialization tests added.
   **Approved by user.** No numerical references updated.
2. Scalar `log2` and raw-data provenance: corrected by reusing the canonical
   normalization sequence for log detection. Original xfail converted; independent
   raw quantities, log2/log10/case variants, three imputation choices, buffer
   independence and main/background exports covered. **Approved by user.**
   No numerical references updated at that stage. The separate no-log/covariate
   raw overwrite was later fixed in the second audit; see preprocessing context. The historical
   assumption of prelogged input when no log step is configured is unchanged.
3. `covariate_p` and `covariate_q`: corrected local-variable collision with FT
   probabilities; stage-1 H0: slope=0 diagnostics are features x 1, FT tests remain
   features x contrasts. Original xfail converted and independent/contract tests
   added. Only diagnostic p/q reference fields amended (three covariate workflows).
   **Approved by user.** Ridge/flat statistical calibration is not newly
   established; no model, missingness or default changes were made.
4. ANCOVA free/nonnegative modes: feature-level suppression of entirely imputed
   backgrounds and contrast-local raw fallback implemented; independent two-stage
   numerical tests and two full pipeline/export runs added. Original crash xfails
   converted. **Approved by user.** No default fixed1 or reference changes.
   Two stale-BH diagnostic failures and one additional deferred Inmoose case were
   recorded separately; the diagnostic failures are repaired in follow-up 1 below.
5. Disconnected robust rollup: restored the historical least-squares initialization
   and solver return annotation removed in `eda73b1`. Original crash xfail moved
   to passing numerical/component tests; full workflow/export check added.
   **Approved by user.** Connected path and numerical references unchanged.
   A separately reproduced scaling defect was recorded as a strict expected
   failure, now addressed by the user's component-policy repair in follow-up 2.
6. Missing optional spectral counts: imported the existing warning helper and
   converted the crash xfail to an ordinary test. Added absent/zero/observed
   evidence, all three limma callers, and H5AD/XLSX omission/preservation checks.
   **Approved by user.** No numerical/reference/schema changes. Raw-input
   peptide/charge requirements are a separate boundary, not bypassed by this fix.
7. Obsolete PyArrow delimited loader: reproduced the undefined `pv_csv` failure
   and removed the obsolete PyArrow and pandas CSV/TSV branches. No template,
   example or inspected local config selects either. Polars is the sole
   delimited reader; omitted/explicit `load_method: polars` remains valid, other
   values fail before file I/O (including for Parquet). PyArrow's Parquet reader
   and package dependency remain. Replaced the obsolete loader xfail with
   passing container/value/missingness and invalid-configuration contracts using
   existing authored data; no new data or references. **Approved by user.**
   No numerical or output-schema changes for supported configurations.

The original InMoose deferral has now been compared with canonical R references.
The mismatch is confirmed; implementation repair remains upstream-only. Historical
upstream investigation is in [stabilization-1.10.4.md](stabilization-1.10.4.md).
All seven original numbered repairs were user-reviewed. Later API removals and
policy changes supersede their former supported-mode status.

## Historical follow-up repair order and xfail accounting

The user requested this sequence after the initial stabilization report.
Disposition before the subsequent explicit policy cleanup:

1. Fitted-mode diagnostic BH: confirmed correctness defect; recompute BH on the
   final neutralized p-vector, retaining all sites. The two original reproductions
   now pass in `test_covariate_statistics.py`, including H5AD round-trip checks.
   Nine additional independent checks cover partial/multiple/all-missing
   background and all three slope modes. Included in user commit `04a380a`.
2. Disconnected robust-rollup uniform scaling: the user's patch rejects
   disconnected proteins by default or fits components independently on explicit
   opt-in. The public preprocessing policy, metadata, numerical/configuration/
   workflow tests and public docs are now updated. Original scaling assertion
   converted to a passing test; old QR-dropped-sample assertions replaced by
   independent quantities retaining every supported sample. **Approved by user.**
   The user intentionally removed explicit robust policy keys from templates;
   CLI tests check the runtime default rather than demanding those keys.
3. Genuine no-log covariate raw overwrite: fixed in the second audit; passing
   independent raw/H5AD/workbook regression tests, without changing input-scale convention.
4. Peptide/charge input validation: classified as missing validation, fixed with
   an early actionable error; support for precursor-less input is a separate feature.

At that stage both Inmoose cases were unchanged pending the user's R reference.

The original suite at `238d3f9` had **nine collected xfail cases** (one function
was parameterized). Each originally lived in `tests/unit/test_known_defects.py`:

| Original case | Status/location after the historical follow-up sequence |
| --- | --- |
| `test_square_matrices_keep_sample_and_feature_labels` | Passing; same name in `tests/unit/test_matrix_orientation.py`. Explicit row orientation plus square/rectangular checks. |
| `test_scalar_and_list_log2_preserve_same_raw_scale` | Passing; same name in `tests/unit/test_raw_provenance.py`. Scalar log detection corrected. |
| `test_covariate_p_has_one_diagnostic_per_feature` | Passing; same name in `tests/unit/test_covariate_statistics.py`. FT/slope probability collision corrected. |
| `test_optional_covariate_slope_modes_can_run[free]` | Passing; same name/mode in `tests/integration/test_workflows.py`, strengthened with numerical/export assertions. |
| `test_optional_covariate_slope_modes_can_run[nonneg]` | Passing; same name/mode in `tests/integration/test_workflows.py`, strengthened with numerical/export assertions. |
| `test_disconnected_robust_rollup_produces_result` | Passing; same name in `tests/unit/test_robust_rollup.py`. Now explicitly opts into component mode and checks both independently derived singleton-component quantities. |
| `test_optional_spectral_counts_can_be_absent` | Passing; same name in `tests/unit/test_statistics.py`. Missing warning import repaired, assertions retained and strengthened. |
| `test_pyarrow_delimited_loader_can_read_supported_input` | Removed with obsolete loader by explicit user instruction. Replaced by `test_supported_file_containers_preserve_labels_values_and_missingness` and `test_only_polars_load_method_is_accepted_before_file_io` in `tests/contracts/test_inputs_and_config.py`. |
| `test_equal_residual_variances_still_allow_moderation` | Still xfail in `tests/unit/test_known_defects.py`; Inmoose `KeyError: -2`, original input/assertion unchanged. |

Four **new collected xfail cases** were added during the numbered repairs:

| Added case | Why added; status after the historical follow-up sequence |
| --- | --- |
| `test_fully_imputed_background_diagnostics_apply_bh_after_neutralization[free]` | Issue 4 exposed stale slope BH after p=1 neutralization. Passing after follow-up 1; moved from known defects to `tests/unit/test_covariate_statistics.py`, original assertions retained. |
| `test_fully_imputed_background_diagnostics_apply_bh_after_neutralization[nonneg]` | Same defect in the other fitted mode. Passing after follow-up 1 at the same new location, original assertions retained. |
| `test_fitted_modes_keep_global_slopes_and_contrast_local_background_support[nonneg-True]` | Issue 4's expanded pairwise/missing-background test reaches Inmoose's infinite-prior-df exception. Still xfail in `tests/unit/test_covariate_statistics.py`; inputs/assertions unchanged, no alternate fixture substituted. |
| `test_disconnected_robust_rollup_preserves_uniform_intensity_scaling` | After Issue 5 restored execution, a twofold input scaling produced fourfold output scaling. Passing after follow-up 2 in `tests/unit/test_robust_rollup.py`, with explicit component mode, the original twofold assertion and additional factors. |

Thus the previous report's count was **9 - 7 passing - 1 obsolete + 4 new = 5**.
Follow-up 1 resolved two BH cases, leaving three. Follow-up 2 resolves scaling,
leaving **2 Inmoose xfails**: the original case and the additional pairwise case.
The second audit adds one strict failure for undefined raw p-values contaminating
other features’ raw q-values; no missing-p family policy was chosen implicitly.
No reported no-log or peptide/charge finding is concealed in these counts.

The later policy cleanup removed fitted-slope modes and their mode-only tests,
including the pairwise nonneg xfail. Undefined-raw-p BH now passes. The current
three xfails are listed above: the unchanged original equal-variance case and
the two canonical R reference cases. Historical test locations in these tables
do not imply that removed modes remain supported.

## Current policy and independent-reference checks

- `tests/contracts/test_scientific_policy.py`: rejects incompatible scale and
  rollup paths, removed imputer names, invalid precursor values, and empty samples;
  verifies prelogged linear evidence including observed log2 zero.
- `tests/contracts/test_limma_reference.py`: labelled designs/contrasts and
  coefficient/raw/moderated statistics against five R cases; checks input hashes.
  Three fits pass and two upstream limitations xfail narrowly. Normal CI reads
  only committed CSVs. Generation/version details are in
  [the reference README](../../tests/reference/limma/README.md).
- `test_statistics.py`: independent finite-only BH ranks/nonmutation and raw versus
  moderated families, uniform selected-analysis pilot, missing batch values,
  reserved labels and invalid contrast selection.
- `test_covariate_statistics.py`: one selected model per cell, independent final
  BH, missing global diagnostics and batch-aware versus simple mean decomposition.
  Generic fixed1/association checks replace fitted-mode-only coverage.

The three remaining xfails are individually accounted for above: original equal
variance, canonical equal variance, canonical non-estimable feature. An unexpected
pass must meet the R numbers; references never change to match InMoose.
The ordinary skips remain optional local data and the two worker-count checks.

The nine non-PELSA workflow snapshots gain pilot metadata; three covariate snapshots
amend only coherent fallback SE/t/raw-p, final BH, missing slope diagnostics and
boolean provenance. Unrelated values remain byte-identical in the references;
new candidates match within the original tolerances. Matrix values/FC and kinase
scores did not intentionally change. Pilot-selection workflows use independent
assertions rather than extra snapshots. The pilot reference now includes A_vs_C
so that C participates in the selected analysis; only contrast metadata and the
new descriptive/count contrast column changed, with the original A_vs_B column
and every preprocessing value retained. Probability tolerances were not broadened.

## Running tests and CI

Use the project's intended environment. A reference environment can be installed
with `python -m pip install -c tests/constraints.txt -e '.[dev]'`. The constraints
pin the numerical packages used to establish references; they do not change
runtime/package defaults and are not a complete transitive lockfile. PhosX's exact
0.23.1 requirement is in production metadata/adapter code. Its original 0.23.0
reference provenance remains recorded. That resource update amended version
labels without changing scores. Later, separately documented amendments added
raw_p_ebayes and corrected the inference fields listed above.

```bash
python -m pytest -q --durations=15
python -m pytest tests/unit tests/contracts/test_inputs_and_config.py -q
python -m pytest tests/integration tests/contracts/test_results.py tests/contracts/test_reference_comparison.py -q
python -m pytest -m regression -q
python -m pytest tests/integration/test_slow_phosx.py --run-slow -q
python -m pytest tests/integration/test_slow_robust_rollup.py --run-slow -q
python -m pytest tests/unit/test_known_defects.py --runxfail -q
```

The final command is diagnostic and currently fails. For a fixed specific defect,
select its individual node rather than treating unrelated known failures as new.
For deterministic CI-style execution, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMBA_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`, `MPLBACKEND=Agg` before Python starts.
`conftest.py` supplies thread/backend defaults and temporary Numba/Matplotlib
caches. Explicit environment values take precedence. Fixture RNG seeds are local
and fixed; LC-ConMed row ordering and PhosX contrast ordering remain meaningful.

`.github/workflows/smoke-test.yml` now runs algorithm/configuration checks and
scientific workflow/serialization checks on pushes and PRs (retaining doc-only
path exclusions). It caches pip dependencies, uses CPU Torch, validates installed
dependencies and uploads JUnit reports. Manual dispatch can add worker-count
validation. The local/full suite is never selected by normal CI. Release
automation remains unchanged.

Python declarations now agree on >=3.10, the floor required by pinned Scanpy,
Numba and PhosX. CI retains 3.11/3.13. The constrained reference environment
requires >=3.11 (SciPy/InMoose); Python 3.10 is not scientifically validated.
Local measurements were on Linux/Python
3.12.7; the matrix still needs actual GitHub execution. The ordinary suite is
24.10 seconds at the second-audit baseline and 48.69 seconds after its additions,
excluding dependency installation;
first-run JIT costs and machines vary. Worker-count validation is short locally;
its process-spawning behavior is separately opt-in for PhosX and robust rollup. Numerous
existing Polars/dependency deprecations are reported rather than globally hidden.

## Optional full-data regression

Run a JSON manifest supplied explicitly by the developer:

```json
[
  {
    "name": "reviewed-experiment",
    "config": "experiment/config.yaml",
    "workdir": "experiment",
    "reference": "references/reviewed.h5ad",
    "provenance": "Describe input source, producing revision/config/environment and scientific review"
  }
]
```

```bash
python -m pytest tests/full --full-regression /path/to/manifest.json -q --durations=15
```

Manifest paths resolve relative to the manifest. Input paths in the YAML retain
production's working-directory interpretation. Each case redirects all exports
to pytest temporary space; config/source/reference files are never overwritten.
Missing explicitly requested files fail clearly; absent local data otherwise
cause no ordinary-suite failures. Optional case `rtol`/`atol` tune large-run
numeric comparisons (default 3e-6/2e-6); probability arrays/table columns retain
tighter 3e-5/1e-12 checks.

`tests/full_regression.py` compares labels, numeric matrices in chunks, metadata,
evidence, missingness and specialized tables without constructing giant JSON
snapshots. Both AnnData objects still load into memory; this is an intentionally
local mechanism, not a streaming multi-GB validator. It shares the documented QC
exclusions and includes corrected covariate diagnostics. References must be
selected with known provenance; old artifacts can expose schema drift and should
not be silently normalized to pass. The runner itself was exercised with the authored protein experiment and
checked not to modify original inputs/config/reference outputs; larger historical
datasets have not been run as part of this suite's establishment.

## Remaining weaknesses

- R comparisons cover five compact cases, not every limma option or platform.
  Infinite-prior-df moderation and non-estimable designs remain upstream gaps.
- Regression linear/LOESS methods now have known-offset and idempotence checks;
  most alternative imputers still lack strong independent numerical protection;
  the incompatible quantile path has been removed. Robust rollup now protects component-local
  quantities/scaling and default rejection, but component membership is not
  serialized and no downstream guard prevents unsupported cross-component
  inference after explicit opt-in. See the preprocessing context.
- The long/wide vendor adapters are checked with authored data; only Spectronaut
  has a reduced real-input workflow. Real DIA-NN/FragPipe data remain desirable
  once redistributable provenance is established.
- More multisite ambiguity patterns, very sparse or
  empty kinase outputs and production-size permutation runs need broader fixtures.
- PELSA extreme/outlier/multi-start/boundary behavior is not comprehensively
  checked. Successful parameter recovery and current fit outputs are covered.
- No PDF image/layout golden, viewer execution, exhaustive optional metadata
  schema, clustering topology/embedding baseline or multi-platform numeric audit.
- Two upstream InMoose limitations remain unresolved (three strict xfail cases).
  They are tracked separately from ProteoFlux repairs; passing reference cases
  do not validate those failing upstream paths.
