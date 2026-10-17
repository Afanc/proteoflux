# Second scientific audit: evidence and follow-up ledger

Scope: public documentation and adversarial scientific review on `dev/1.10.4`.
The user authorized small confirmed-defect repairs and tests, not unrelated
features/refactors. ProteoViewer was inspected read-only. No commits or remote
operations. The first stabilization sequence is historical; this audit does not
require stopping after each repair. At the start of this audit, InMoose was
deferred pending a canonical R limma reference. The subsequent policy pass
generated that reference; implementation repair remains upstream-only.

## Policy cleanup, 2026-10-13 — supersedes pending decisions below

The following ledger is a historical record of the adversarial audit. The user
subsequently supplied explicit policies and authorized implementation:

- Undefined raw p/q stay missing; common BH excludes nonfinite hypotheses.
- Log2 is the quantitative contract. log10/quantile and repeated/late log steps
  are rejected; prelogged input must be preaggregated, with no linear censoring.
  Prelogged evidence/counts are explicitly reconstructed in linear units.
- Phospho adjustment is fixed1 only. Fitted modes and their mode-only tests are
  removed. Generic regression/association tests remain.
- Fallback selects coherent model statistics; BH runs once on each final p family.
  Unestimable global slope diagnostics stay missing. Boolean covariate_adjusted
  records per-cell selection. cov_part is intentionally the descriptive difference
  in background condition means, not a batch-adjusted model coefficient; this is
  settled design, not a defect or a pending redesign.
- Pilot is uniform across the selected analysis: any singleton among conditions
  in requested contrasts makes all contrasts pilot. An untested singleton does
  not trigger pilot mode. Shared/per-pair variance-estimation scopes are unchanged.
  This is the user's later clarification; mixed statuses are unsupported.
- Empty samples fail before inference; iterative_hgb replaces randomforest and
  truthful hgb_* keys replace rf_* without aliases. Estimator math is unchanged.
- Missing batch values, reserved/colliding conditions and invalid precursor values
  fail clearly. PELSA is in the CLI registry; unused volcano_annotate_infinite
  is removed from shipped templates/examples.
- Canonical R 4.3.3/limma 3.65.1 results were generated locally for five compact
  cases. Balanced/shared/batch agree; equal-variance moderation and feature-specific
  rank deficiency expose InMoose exceptions. No dependency code/workaround changed.

See testing.md and the public 1.10.4 release notes for current interfaces/tests.
Remaining decisions: PELSA optimization/calibration, stochastic order invariance,
continuous covariates, robust component provenance/contrast restrictions,
and background confidence-column requirements. cov_part is not on this list:
the user confirmed its descriptive design, including with categorical batches.
These are not resolved by increasing test count or accepting snapshots as truth.

## Confirmed repairs

| Finding / classification | Reproduction and smallest correction |
| --- | --- |
| No-log background overwrites main raw: data provenance/correctness | `test_prelogged_background_cannot_overwrite_main_raw_or_exports` failed for both no normalization and median equalization. `_normalize_matrix` wrote both blocks into `raw_df`; now selects `raw_df` or `raw_df_covariate` explicitly. Tests compare main/background raw values and NaNs to original linear quantities through H5AD/XLSX. The prelogged-input convention is unchanged. |
| Condition sanitization merges groups: correctness/input validation | `drug-A`/`drug A`, and `1`/`C_1`, silently became the same model condition. `_get_condition_map` now rejects many-to-one original/sanitized names before grouping. |
| Required charge error: input validation | Missing peptide sequence already raised; missing charge failed deep in Polars peptide-table code. Preprocessing now validates both operational requirements early with mapping guidance. This does not make optional vendor spectral counts mandatory or add precursor-less input support. |
| `randomforest` fails in a clean interpreter: robustness | The factory lacked sklearn's experimental enabling import. A subprocess reproducer failed with ImportError; the import is now local to that factory branch. Observed values remain exact. No estimator/settings changed. |
| PhosX dependency/adapter contradiction: configuration/interface | Packaging required >=0.23.1 while adapter required exactly 0.23.0. Align both to exactly 0.23.1; contract test checks requirement membership and actual installation. All 20 non-cache package files in the available 0.23.0 and downloaded 0.23.1 distributions were byte-identical. Only resource-version strings changed in existing output references. |
| PELSA localization length: serialization/interface | `ACM[Oxidation (M)]K` reported length 19 rather than 4. Reuse existing `strip_mods`; IDs and fitted quantities stay unchanged. |
| Invalid PELSA doses: input validation/correctness | Negative, NaN and infinite non-control doses entered the ratio table and could be excluded by the fitter's finite-axis mask. Validate finite concentrations and positive non-control doses. An explicitly configured negative control sentinel remains supported. |
| Phospho RAW_PVALUE uses adjusted model: serialization/interface | Contract compared XLSX raw p/q against a separate unadjusted fit of identical processed values/design. 10/12 first-contrast p-values differed (maximum absolute difference ~0.1315). The covariate path computed raw p but omitted storage. Store `varm.raw_p_ebayes` beside raw q; the existing exporter already consumes it. No fitting/moderation arithmetic changed. Viewer `components/selection_export.py` already supports this optional field. |

The CLI-template baseline failure was an **incorrect test assumption**: the user
intentionally removed explicit robust-policy keys from templates. The test still
checks byte-for-byte template copying, then checks the effective default `raise`.
No template keys were restored. Python declarations were corrected from >=3.8
(packaging)/>=3.9 (README) to >=3.10, required by pinned Scanpy/Numba/PhosX;
scientific CI remains 3.11/3.13 and does not validate 3.10.

No private data or new data files were added. New cases are tiny matrices or
variants of the existing authored workflows. Three phospho references gained
`raw_p_ebayes`, derived from unadjusted fits to their *stored* X/labels/raw mask;
existing raw FC/q were independently checked against that fit. Existing numerical
reference values were not regenerated. Provenance annotations distinguish this
amendment from the original reference generation.

## Contracts challenged

| Area | Counterexample/invariant and result |
| --- | --- |
| Axes/provenance | Reviewed unique-valued square/rectangular orientation tests, buffer independence, evidence axes and H5AD/XLSX round trips. New prelogged background test found the overwrite above. Existing labels plus exact cell checks remain intact. |
| Normalization | Known unequal sample offsets distinguish global mean reference from median; global/local linear and LOESS recover the specified references and are idempotent on this example. Changing a sparse non-reference feature 1024-fold and reversing input rows leaves complete/tagged reference quantities unchanged. These checks do not validate arbitrary nonlinear normalization. |
| Imputation | LC-ConMed observed-mask/condition thresholds reviewed; adding 3 log units translates both MAR/MNAR fills by 3 with the same draws. Existing fractional thresholds, all-missing behavior and once-per-parent broadcast checks retained. Empty sklearn sample columns fail the intended shape contract (ledger below). |
| Rollup | Connected directLFQ and robust estimates checked against independent sum/geometric mean expectations under 13-fold scaling and label reversal. Existing sparse/disconnected tests verify rejection or explicit within-component estimates, never between-component ratios. |
| Differential/design | Contrast reversal changes FC/t signs but preserves SE/p/q; feature-specific log offsets preserve contrasts/probabilities. Existing independent least squares, design rank, batch confounding, singleton and global/separate-scope tests reviewed. Undefined raw p contamination exposed a new failure. |
| Phospho | Strengthened reordering from slope diagnostics alone to every varm field plus sample-labelled residuals. Existing independent slope regression, BH after neutralization, sparse background support, paired runs and parent-level imputation tests retained. Fitted modes are not thereby scientifically endorsed; see user rationale below. |
| KSEA | Strengthened the previous one-tested-kinase example to two tested kinases, with hand-derived means, sample variance, normal tails and BH ranks. Duplicate relationships still contribute once and insufficient support remains untested. |
| PhosX | Existing dense KS/null reference, family-wise BH and small resource workflow reviewed. New labelled reorder + perturbation of the other contrast leaves first-contrast results/assignments unchanged for uniquely ranked sites. Tie ordering and contrast-index seeds remain separate limitations. |
| PELSA | Label reorder, peptide-specific intensity scaling and 1000-fold dose-unit conversion preserve ratios and translate the log-dose/pEC50 axes as expected. Existing generating-EC50 recovery, derivative and dose eligibility tests retained. Recalibrated significance is a frozen distribution specification, not independent empirical calibration. |
| Serialization | Tests cover core and specialized H5AD payloads and workbook content; new raw-data, raw-model p/q and residue-count assertions prevent scientific label/meaning mismatches. No binary-file comparison. |

## Historical next-session ledger — superseded by the policy cleanup above

The user requested this list on 2026-10-12. It preserves the original findings
and open decisions, not the final unresolved-item list. The policy cleanup above
resolved the ProteoFlux defects/API decisions; remaining upstream limitations and
future validation work are identified separately. In particular, the former
cov_part redesign question was settled as intentional descriptive design.

| Original priority / type | Finding and decision recorded at that stage |
| --- | --- |
| High — confirmed correctness defect | **Undefined raw p poisons all raw q.** An observed feature with X=0, raw=1 yields p_raw=NaN; `stats_ops.bh_qvalues` passes the entire column to statsmodels and all 24 q_raw become NaN although 23 raw p-values and all moderated q-values are finite. New strict xfail protects finite q for otherwise testable features. Decide explicitly whether undefined hypotheses remain in the BH denominator or are excluded; preserve their missing p/q rather than inventing significance. No change/workaround to InMoose. |
| High — upstream/reference block | **Two InMoose failures unchanged.** Equal residual variances and pairwise nonnegative fitting both still reach `KeyError: -2` at `inmoose/limma/ebayes.py` (`t2[~Infdf]`). Both existing xfails and inputs remain intact; canonical R comparison is required. |
| Medium — scale contract defect/design decision | **Accepted log10/quantile versus log2-only consumers.** `log2fc`, PELSA exp2/control ratios and export labels assume log2, but preprocessing accepts other scales without downstream enforcement. Prefer explicit scale validation or a documented scale contract; do not silently relabel log10 as log2 or change default normalization. Also review already-logged precursor rollup: summing logs is not log of a sum. |
| Medium — scientific scope decision | **Experiment-wide pilot mode.** Any singleton forces pilot before selected/separate contrast fitting. Current tests intentionally protect that behavior, not its superiority. Decide whether pilot should be per contrast when separate fitting is requested; account for global preprocessing, residual df and downstream schema before changing it. |
| Medium — naming/API cleanup | **`randomforest` selects iterative histogram gradient boosting.** Runtime import repaired, estimator unchanged. Decide a truthful canonical method name and migration policy; do not replace it with a literal forest merely to match the old key. |
| Medium — scientific model/API decision | **Retire fitted phospho slopes?** User reports having compared fitted, zero and unit slopes; fixed slope one gave the best results, while fitted alternatives converged poorly. This is the empirical rationale for fixed1, not accidental code. User leans toward keeping only fixed1. Record/locate the benchmark evidence, then explicitly decide removal/deprecation of free/nonneg and affected configuration/tests. Do not remove them incidentally, or treat passing numerical checks as proof of calibration. |
| Medium — model/result consistency | **Selective no-FT fallback.** FC and moderated p/q fell back while SE/t/unmoderated fields retained residual-model values; q mixed separate families. The later policy pass corrected coherent selection and final-family BH. The separate cov_part question was reclassified: its descriptive background mean difference is intentional with or without batches; no exact batch-adjusted decomposition is required. |
| Medium — missing-data policy | **Empty samples in sklearn imputers.** Factory `knn`/`randomforest` on a 3x3 matrix with its middle sample entirely missing returns 3x2. Pipeline shape reconstruction cannot preserve sample labels. Reject early or define an explicit scientific fill/drop policy; never silently fabricate zero measurements. Mean/median and LC-ConMed also have numerical all-missing fallback conventions that are not evidence. |
| Lower — input/design hardening | **Partial missing batch factors and reserved labels.** Partial batch nulls become string categories; condition names containing the contrast separator or model-reserved names still need targeted validation. Sanitization collisions are now rejected, but null/invalid peptide sequences or charges need more than column-presence checks. |
| Lower — configuration usability | **PELSA template registry and distributed validation.** Packaged PELSA YAML is not in CLI `_TEMPLATES`; an explicit addition is small future work. Unknown keys can be accepted but ignored; options such as `volcano_annotate_infinite` in templates have stale/nonuniform consumers. Audit individual keys rather than inventing a global framework. |
| Lower — numerical interpretation | **PELSA optimizer diagnostics.** `fit_success` does not certify convergence/identifiability; best multi-start selection minimizes SSE even with robust loss; float64 optimizer input/bounds originate in float32; zero-SSE tests have missing significance. Independently validate these choices before modifying results. |
| Lower — reproducibility contract | **Ordering affects random draws.** LC-ConMed assigns row-major draws; PhosX includes contrast index in seed derivation. Tied rank order can affect KS/assignments. Current compact checks establish fixed-input/hash/worker behavior, not universal permutation invariance. |
| Lower — compatibility hardening | **Component metadata and backgrounds.** Robust component opt-in does not serialize membership or suppress unsupported between-component comparisons. Covariate attachment still has stronger q-value requirements than optional-pivot code suggests. Continuous batch regression would be a new feature, not a categorical-batch bug. |

Public docs describe supported operation and necessary interpretation. This historical
ledger preserves findings, then-pending policy decisions and the user's model history;
it is not a recommendation to preserve defects or add arbitrary compatibility
workarounds. Do not turn future features (e.g. continuous covariates) into release
blockers without an actual existing contract violation.

## Historical xfail accounting before policy cleanup

Before this audit: two InMoose xfails. Both were rerun with `--runxfail` and still
fail at the same upstream `KeyError: -2`; the other three surrounding parameter
cases pass. No R reference was added and no moderation code/expectation changed.
This audit adds exactly one xfail:
`tests/unit/test_statistics.py::test_undefined_raw_test_must_not_erase_other_features_qvalues`.
Its finite-p/moderated-q prerequisites pass, and the final finite-raw-q assertion
fails. An XPASS requires review of the missing-p BH policy, not automatic acceptance.
The original nine-case accounting remains in `testing.md`.

## Historical second-audit validation record

Linux, Python 3.12.7 and the existing reference numerical libraries were used.
The shell's default Python 3.13 environment lacked required test/runtime packages;
it was not used as validation evidence. Pytest and PhosX 0.23.1 were loaded from
scratch directories without modifying the user's environments. The initial
baseline used the available PhosX 0.23.0, exposing the packaging/adapter mismatch;
subsequent checks used 0.23.1. No private/full datasets or production permutation
counts were run.

| Check | Result |
| --- | --- |
| Initial normal suite, once | 252 passed, 1 failed (obsolete template-key expectation), 3 skipped, 2 xfailed; 24.10 s. |
| Initial new defect reproducers, before repairs | 8 failed, 1 passed; 3.43 s. |
| Initial repaired boundary tests | 10 passed; 3.47 s. |
| Scaling/reversal/dose-unit/missing-p challenges | 6 passed, 1 new xfail; 4.02 s. |
| Invalid-dose reproducer before fix | 4 failed, 1 passed; 1.89 s. |
| Related configuration, preprocessing, kinase, PELSA, provenance/workflow selection | 113 passed; 17.81 s. |
| Linear/LOESS offsets + expanded covariate reordering | 5 passed; 2.86 s. |
| PhosX order/contrast isolation | 1 passed; 9.05 s including fixture creation. |
| RAW_PVALUE export correction + surrounding covariate fits | 6 passed; 18.31 s. Before correction the new export assertion failed numerically. |
| Final normal suite, once (`python -m pytest -q --durations=15`, hash seed 0) | **279 passed, 3 skipped, 3 xfailed; 48.69 s.** |
| Compact `-m regression`, hash seed 1 | 23 passed; 30.44 s. |
| Compact `-m regression`, hash seed 8675309 | 23 passed; 30.46 s. |
| Small PhosX one/two-worker check, `--run-slow` | 1 passed; 2.46 s. |

The two hash runs were independent and concurrent, so their times include shared
machine load. Configuration/H5AD/workbook contracts are included in the final
normal suite and were not repeated separately. Normal skips are the explicit
local-manifest test and the optional PhosX/robust multiprocessing tests; PhosX was
then run explicitly. Existing deprecation warnings remain visible (2043 in the
final full run), not blanket-suppressed. CI itself was not run remotely.

A structural comparison against the initial working-tree snapshot confirmed all
pre-existing numerical reference values unchanged. Both InMoose reproductions,
markers and expectations are unchanged. Local documentation links and
`git diff --check` passed. New standalone documents are small text files; all
new scientific input examples are generated in pytest temporary directories.

Intentional output changes: main/background raw layers and their workbook values
are corrected for prelogged covariate runs (raw-based evidence masks can also
change downstream results); `raw_p_ebayes` is added to covariate H5AD outputs and
fixes existing XLSX RAW_PVALUE columns; PELSA localization residue counts are
corrected; PhosX resource provenance reports 0.23.1. Colliding conditions and
invalid PELSA doses now stop instead of yielding merged/reduced analyses. Required
charge failure is earlier/clearer. No existing field was renamed or removed.
