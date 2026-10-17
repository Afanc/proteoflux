# ProteoFlux 1.10.4: behavior and configuration changes

Review these changes before reusing an earlier configuration or comparing results.
The [configuration guide](CONFIGURATION.md) and [output reference](RESULTS.md)
describe the current interfaces.

- Quantitative analysis uses log2 units. `log10` and uniform `quantile`
  normalization are rejected. Linear inputs require one initial `log2` step.
  Prelogged inputs must already contain one measurement per feature/sample;
  linear rollups and linear censoring cannot operate on logged values.
- `imputation.method: iterative_hgb` names the existing iterative histogram
  gradient-boosting imputer. `randomforest` and `rf_*` settings are removed,
  without aliases. Use `hgb_max_depth`, `hgb_learning_rate`,
  `hgb_estimator_max_iter` (formerly `rf_n_estimators`), `hgb_max_iter`, `hgb_tol`
  and `hgb_nearest_features`. The estimator and its defaults are unchanged.
- Phospho background adjustment supports only `ancova.beta_mode: fixed1`.
  `free` and `nonneg` are removed. Association diagnostics remain separate from
  the applied unit slope; unestimable diagnostics are missing, not p=1.
- Pilot status is uniform across selected contrasts: any participating singleton
  makes the whole analysis pilot. A singleton excluded by `only_contrasts` does
  not trigger pilot mode. `only_against` still includes all other conditions.
  Shared variance estimation retains all conditions; separate fits use each pair.
- Undefined raw p/q values remain missing. Finite raw p-values receive BH
  correction within their contrast, independently of moderated tests.
- Phospho fallback now selects a complete set of model statistics, then applies
  BH to the final p-vector. Corrected SE/t/raw-p and q-values can differ from
  earlier mixed-model outputs. `varm.covariate_adjusted` records selection.
- `cov_part` retains its descriptive meaning: the difference in mean log2
  background/protein abundance between conditions. It is not intended to exactly
  decompose batch-adjusted phosphosite fold changes; no model redesign is needed.
- Entirely empty quantitative samples fail with their names; imputers cannot
  drop them or manufacture measurements. Invalid precursor sequences/charges,
  missing batch values and ambiguous/reserved condition names are rejected.
- `spectronaut-pelsa` is selectable through the CLI template registry. Templates
  no longer advertise the unused `volcano_annotate_infinite` setting.

Earlier 1.10.4 corrections preserve square-matrix orientation, main/background raw
provenance and raw-model workbook probabilities. Robust rollup rejects disconnected
components by default; explicit `component` estimates only support within-component
comparisons. Spectral counts remain optional evidence. Polars is the supported
CSV/TSV reader; PyArrow remains used for Parquet.

The Python test suite includes committed references generated with R 4.3.3 and
limma 3.65.1. R is not needed to run ProteoFlux or its normal tests. Equal-variance
moderation and non-estimable feature designs still expose known InMoose limitations;
there is no local statistical workaround. See the
[reference provenance](../tests/reference/limma/README.md).
