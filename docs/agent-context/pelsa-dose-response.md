# PELSA dose-response analysis

`main.run_pipeline` dispatches `analysis_type='pelsa'` to
`analysis/pelsa_pipeline.run_pelsa_pipeline`, which delegates curve fitting to
`analysis/pelsa_torch.py`. It bypasses pairwise limma/empirical Bayes, then uses
the shared QC and export pipeline. Do not add limma-shaped results merely to make
a downstream consumer accept PELSA.

## Peptide inputs and response construction

Preprocessing uses peptide indexing and `peptide_rollup_method`, not protein
rollup. The harmonizer preserves original protein INDEX as a PARENT_PROTEIN
fallback if UNIPROT is unavailable; missing both sources raises. Parent groups
are exact strings, not split into separate proteins. PTM collapse therefore
affects fitted features and parent-support counts.

The normal annotation path requires exact `Concentration`, castable to numeric.
`_get_condition_map` explicitly carries that name into `obs`. Analysis exposes
`analysis.pelsa.concentration_column`, but setting it alone does not change these
upstream requirements. Conditions/replicates are still needed for shared
preprocessing and QC; fitting uses concentrations rather than pairwise contrasts.

`_build_pelsa_ratio_table` defaults to `layers['normalized']`, preserving
pre-imputation missingness, and expects log2 input. Controls are selected by exact
numeric equality to `control_concentration`, default 0.0. Per peptide:

```text
control_ref = log2(mean(2 ** finite_control_log2_values))
log2_ratio = sample_log2_intensity - control_ref
ratio = 2 ** log2_ratio
```

This is a mean on the linear scale, not the mean/median log2 control. All finite
replicate points are retained; concentrations are not averaged before fitting.
Peptides without finite control reference produce no ratio points, even if other
doses are observed. They can remain in AnnData without a curve-result row.

Controls participate in fitting. `_control_log10_position` places them one
typical positive log-dose step below the smallest positive concentration:
median gap of distinct log10 positive concentrations, or one log10 unit when
only one positive concentration exists. Preserve this in both plots and fits.
`curve_points` includes concentration, pseudo and raw log10 concentration,
is_control, sample, peptide_id, log2_intensity, log2_ratio, ratio and optional
replicate. The control shift is applied to configured controls; the positive-dose
grid is still determined by `concentration > 0`.

Concentrations must be finite, and non-control concentrations must be positive;
invalid sample labels are included in the error. A finite configured control
sentinel may be nonpositive. The code does not impose a concentration unit;
EC50 is in the units supplied. Do not claim
molar pEC50 semantics without knowing input units, or silently rescale doses.

## Eligibility and current optimizer

`build_dense_torch_input` creates peptide x sample response and mask arrays,
retaining sample points. A peptide needs at least **four distinct finite x levels**
(including pseudo-controls) and at least `min_control_points` finite controls.
The runtime default is 1; the packaged PELSA template requests 2. Global control
sample count is validated first. Setting the minimum to zero does not bypass
the need for a finite reference to construct ratios.

Ineligible peptides represented in the ratio table get failure rows; if none
are eligible, fitting raises. `failure_reason` prioritizes insufficient controls
when both eligibility checks fail. There is no active SciPy backend selector,
despite stale text in `require_torch`'s error message.

The current backend is batched **CPU Torch Adam**. Defaults from
`TorchFitConfig`/`PelsaBounds` (templates may override):

| Option | Runtime default |
| --- | --- |
| `fit_steps`, `fit_lr` | 300, 0.03 |
| `fit_batch_size`, `fit_n_starts` | 4096, 1 |
| `fit_loss`, `fit_huber_delta` | `mse`, 1.0; also `huber` and `l1` |
| `fit_dtype` | `float32`; `float64` supported for Torch operations |
| `fit_betas`, `fit_eps`, `fit_weight_decay` | (0.9, 0.999), 1e-8, 0 |
| `fit_num_threads` | Unset; if set, calls process-wide `torch.set_num_threads` |
| `bounds.slope` | (0.01, 10) |
| `bounds.front`, `bounds.back` | (-15, 15), both must span zero |
| `bounds.pec50_margin_log10` | 0 |
| `bounds.response_margin_fraction` | 0.5 |

The template/example instead use Huber delta 0.5, five starts and zero response
margin. Compare configuration before interpreting numerical differences.

The model (`four_pl_numpy`, `four_pl_torch`) is:

```text
y = back + (front - back) / (1 + 10 ** (slope * (x + pec50)))
x = log10(concentration with pseudo-control)
y = log2_ratio_to_control
EC50 = 10 ** (-pec50)
```

The sign in `x + pec50` matters. Parameters are sigmoid-transformed from
unconstrained coordinates into bounds. pEC50 limits derive from the full fitting
x grid, including pseudo-control, extended by configured margin. Per-feature
plateau bounds intersect global limits with observed response range plus margin;
invalid intersections raise. The initial five starts reproduce fixed historical
choices, and extra starts follow deterministic grids.

Loss is summed over observed points and starts, normalized by mask count.
After fixed optimizer steps, `evaluate_model` selects the start with minimum
**squared error even for Huber/L1 fitting**. Fitted batches are labelled
`fit_success=True`; there is no convergence or quality threshold behind that
flag. `r2`, normalized RMSE, range/CI fields and failure metadata are separate
diagnostics. Dense inputs and bounds are built in float32 before optional Torch
float64 conversion, so selecting float64 does not recover original input precision.

## Statistics and interpretation

`curve_fold_change_log2` is fitted response at the global largest x minus fitted
response at global smallest x, not the asymptotic `back-front` amplitude. RMSE
uses finite point count; normalized RMSE divides by absolute plateau amplitude.
`pEC50_inside_range` compares with limits from the grid including pseudo-control.

`estimate_pec50_ci_batch` uses local nonlinear least-squares covariance
`SSE/(n-4) * pinv(J.T @ J)` and Student-t 95% intervals. The same approximation
is applied after robust-loss fitting; it is not a bootstrap or robust covariance
estimator. Fewer than five points or invalid calculations give missing intervals.

`add_curve_statistics` implements its own CurveCurator-style recalibration:

```text
F = max(0, ((SSE_null - SSE_curve) / SSE_curve) * n / 4)
adjust(n) = 1 / (((n - 4) ** 4) / n + 4)
dfn = 5
dfd = (0.8 - adjust(n)) * (n - 2.5)
p = scipy.stats.f.sf(F, dfn, dfd, loc=0.12, scale=1)
```

Only successful fits with finite SSEs, positive curve SSE and n>4 get this test;
BH is applied over finite peptide p-values. Perfect-zero SSE is excluded rather
than assigned p=0. Do not replace the recalibrated distribution with a textbook
nested-model F-test without a deliberate scientific change and regression data.
The repo does not establish external CurveCurator equivalence.

`_annotate_parent_significant_peptide_counts` counts distinct fitted peptide rows
with finite `curve_q_value < analysis.sign_threshold` (default 0.05) per exact
PARENT_PROTEIN string. It rejects duplicate curve peptide IDs and unknown IDs.
Counts are attached to curve results as `significant_peptides_per_parent` and to
all features as `var.PELSA_SIGNIFICANT_PEPTIDES_PER_PARENT` (unmapped rows get zero).
This is descriptive peptide support, not a protein-level combined significance
test or a filter on the original curves.

## Outputs and compatibility

`uns['analysis']` records `analysis_type='pelsa'` and
`analysis_method='pelsa_curve_fit'`. `uns['pelsa']` contains `curve_points`,
`curve_results`, formula/scales, source layer, control reference and position,
thresholds, concentration array, summary and localization metadata. No pairwise
`contrast_names`/limma matrices are generated by this branch.

Curve results carry snake-case `pec50` but mixed-case diagnostic names such as
`pEC50_inside_range`, optimizer metadata, fit/failure fields, SSEs and curve
statistics. `sanitize_curve_results_for_h5ad` converts missing text to strings,
boolean fields to bool, and counts to int64. Leading-underscore aliases at the
end of `pelsa_torch.py` explicitly preserve older internal imports; the old
`_build_dense_torch_input` wrapper returns four items, unlike the dataclass API.

`DEExporter._pelsa_curve_summary` derives EC50 and signed change fraction
`sign(fc)*(1-2**(-abs(fc)))` for the workbook, prefixing selected columns `PELSA_`.
H5AD retains more fit diagnostics than the workbook. PDF has a PELSA volcano path;
generic QC still uses normalized-layer data independently of curve fitting.

Optional peptide start/end/length and protein size metadata support localization
display. Preprocessing estimates amino-acid protein length by `weight/110`; units
must be checked against the vendor. `uns.pelsa.localization.peptide_length` is
computed using `utils.sequence_ops.strip_mods`, counting residues rather than
inline PTM annotation characters. The audit fixed the previous feature-ID string
length; the feature IDs themselves and curve quantities are unchanged.

## Validation targets

`tests/unit/test_pelsa.py` protects linear-mean controls, pseudo-dose placement,
distinct-dose eligibility, 4PL/Jacobian calculations and recalibrated significance.
An authored end-to-end experiment covers partial/absent controls, increasing/
decreasing curves, generating EC50 recovery, parent counts and H5AD/table
semantics. Its precise optimizer results are current-behavior references, not
independent validation of all fitted diagnostics. `examples/pelsa_small/` remains
a larger manual fixture. See [testing.md](testing.md) for tolerances and gaps,
especially extreme/outlier/boundary and best-start behavior. Optimizer, bounds,
dtype, batching and response preprocessing all deserve numerical comparison.
