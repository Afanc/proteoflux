# Architecture, configuration, and development

This context describes the checked-out implementation, not an idealized pipeline.
Source and executable branches take precedence over comments, templates, and the
public configuration reference. The initial survey used revision `297ca9b`.

## Execution and ownership

`pyproject.toml` exposes `proteoflux = proteoflux.cli:app`. The Typer CLI has
`templates`, `init TEMPLATE --path FILE`, and `run --config FILE`. Only `run`
imports the heavy pipeline and display setup; preserve lightweight template
commands. `init` copies a packaged YAML verbatim and overwrites its destination.

`main.run_pipeline(config)` is the programmatic entrypoint. It writes outputs
and has no explicit return value. Use `Dataset(**config).get_anndata()` when an
in-memory preprocessed object is needed; that object has not undergone statistics
or clustering. Construction of `Dataset` already performs loading/preprocessing.

The call chain is:

```text
cli.run: yaml.safe_load
  main.run_pipeline
    phospho: validate kinase method, optional FASTA and selected method resources
    workflow.dataset.Dataset(**config)
      load CSV/TSV/Parquet
      utils.harmonizer.DataHarmonizer.harmonize
      load/harmonize/concatenate optional injected runs
      workflow.preprocessing.Preprocessor.fit_transform
      Dataset._convert_to_anndata
    analysis.validation.validate_analysis_input (at least two samples)
    phospho: kinase_sequences.prepare_kinase_sequences
    pelsa: pelsa_pipeline.run_pelsa_pipeline
    otherwise: limma_pipeline.run_limma_pipeline
      phospho: optional ksea_pipeline or phosx_pipeline
    limma_pipeline.clustering_pipeline
    optional PDF -> required H5AD -> optional table
```

Configuration validation is distributed. There is no central schema, default
merger, general unknown-key rejection, or automatic path rebasing to the YAML's
directory. Relative input, annotation, database, and output paths normally resolve
from the process working directory. Some resource readers explicitly expand `~`;
do not assume all readers do.

## Module boundaries

| Area | Implementation and role |
| --- | --- |
| Input adapters | `utils/harmonizer.py`: configured columns to canonical long table; annotations, sample identity, peptide/site identity |
| Dataset orchestration | `workflow/dataset.py`: file readers, injected assays, AnnData assembly |
| Processing | `workflow/preprocessing.py`: filtering, pivoting, normalization, imputation and parent-protein broadcast |
| Intermediate containers | `dataset/intermediateresults.py`, `dataset/preprocessresults.py`: working artifacts and the return bundle; these are separate from `workflow.dataset.Dataset` |
| Quantification | `workflow/rollup/`, `workflow/peptide_tables.py`: protein summaries and evidence tables |
| Differential analysis | `analysis/limma_pipeline.py`, `stats_ops.py`: active modelling implementation |
| Specialized analysis | `analysis/pelsa_pipeline.py`, `pelsa_torch.py`; `kinase_sequences.py`, `ksea_pipeline.py`, `phosx_pipeline.py`, `phosx_adapter.py` |
| QC | `analysis/clustering.py`, `missingness.py`, `nrsc.py` |
| Consumers | `export/de_exporter.py`, `export/pdf_report_exporter.py`; external ProteoViewer consumes H5AD |

`analysis/statisticaltester.py` is not called by the active pipeline. Its generic
`t`/`p`/`q` output keys and missingness representation are not the current result
contract. Likewise, `utils.utils.debug_protein_view` still expects some old keys.
Trace callers before editing an apparently relevant helper.

## Configuration rules that cross modules

- Top-level blocks are `dataset`, `preprocessing`, `analysis`, `exports`; the
  covariate model also reads top-level `ancova`. Options belong where their
  consumer reads them. For example, `convert_numeric_ptms` is read from the
  dataset/harmonizer configuration; the public reference now places it there.
  The two collapse flags are read from preprocessing and
  explicitly forwarded to the harmonizer.
- `dataset.load_method` defaults to `polars`, its only accepted value. Dataset
  construction rejects other values before file I/O. CSV/TSV uses Polars;
  Parquet uses its existing PyArrow reader regardless of omission/explicit
  `polars`. The PyArrow dependency is still required for that format.
- `normalize_analysis_type` actually accepts `proteomics`, `peptidomics`,
  `phospho`, `pelsa`; missing/blank defaults to proteomics. Aliases include
  `proteo`, `protein`, `proteins`, `peptido`, `phosphoproteomics`. `DIA`/`DDA` are
  not accepted by its implementation despite its docstring. The tuple in
  `utils/semantics.py` omits PELSA; it is not a complete dispatch registry.
- Main `protein_rollup_method` and legacy `pivot_signal_method` cannot both be
  non-null. Default fallback is `sum`. Main peptidomics, phospho, and PELSA use
  `peptide_rollup_method`, default `sum`, regardless of protein rollup settings.
- `preprocessing.robust_regression_disconnected_policy` defaults to `raise`;
  only `raise`/`component` are accepted, including before input-file I/O through
  Dataset. Preprocessor forwards it to the rollup function's `disconnected_policy`
  argument and records it in quantification metadata for proteomics robust rollup.
  Read the preprocessing context before using component estimates downstream.
- `exports` takes precedence as a whole over legacy `analysis.exports` when
  truthy; these mappings are not merged. `path_h5ad` is mandatory; PDF and table
  paths may be null. Export validation occurs after analysis/clustering.
- `Dataset` derives whether covariates are enabled from active
  `inject_runs.*.is_covariate` entries; setting preprocessing covariates alone
  cannot enable the normal pipeline branch. Injection indexing does not change
  the main analysis type. See [phosphoproteomics-and-kinases.md](phosphoproteomics-and-kinases.md).
- `analysis.only_contrasts` takes precedence over `only_against`.
  `separate_contrasts` defaults to boolean `False` and rejects null/strings when
  read by limma. `batch_effect_columns` are categorical, including numeric-like
  annotations. Both settings change the statistical design, not preprocessing.
- Configuration is often accessed through `.get`, with inconsistent treatment
  of null versus omission. Do not add a global null/default cleanup without
  tracing each consumer. Several options validate locally, others fail later.

Templates are examples, not constructor defaults. For example, the Spectronaut
proteomics template selects directLFQ, eight cores, PEP 0.05, censoring at 2,
log2/median normalization and LC-ConMed; constructor defaults differ. The
phosphoproteomics template selects KSEA but leaves the database/mappings empty,
so it needs editing before running. `spectronaut-pelsa.yaml` is packaged and registered as `spectronaut-pelsa`
for `proteoflux init`.

Adding a configuration option requires checking the constructor/parser, callers,
metadata/export representation, templates, examples and configuration reference.
Validate the new key, define its default, propagate it consistently, and cover
default/nondefault behavior. Preserve existing keys and defaults unless changing
them is the actual task.

## Environment, build, and checks

Setuptools builds the package from `pyproject.toml`; templates are package data.
`MANIFEST.in` excludes tests, docs, examples and large result/data formats from
sdists. Distribution version comes from installed package metadata in
`proteoflux.__init__`, falling back to `0+unknown`, not a source constant there.

Use the intended conda/virtual environment and verify both `python` and the
`proteoflux` executable resolve into it. Running `python -m proteoflux.cli`
avoids accidentally using another environment's console script.

```bash
python -m pip install -e '.[dev]'
python -m proteoflux.cli templates
python -m proteoflux.cli init spectronaut-proteomics --path /tmp/proteoflux-agent-template.yaml
python -m proteoflux.cli run --config /path/to/config.yaml
```

The build workflow installs `build`/`twine`, runs `python -m build`, and checks
`python -m twine check dist/*`. It uses Python 3.11; scientific CI uses 3.11/3.13. Package
metadata and README now require Python >=3.10, matching the pinned Scanpy,
Numba and PhosX package floors and source syntax. Python 3.10 is not a tested
scientific environment; reference constraints include newer packages requiring
3.11. Use CI versions as validation evidence, not a promise for every interpreter.
Avoid mixing existing local `dist/` artifacts with a fresh build.

Important dependency contracts include pandas <3, NumPy <2.2,
`scanpy==1.11.0`, `numba==0.61.0`, `anndata==0.11.4`, `polars==1.36.1`.
`inmoose` and `directlfq[dask]` are unpinned; upstream changes can alter results.
PhosX is pinned to PyPI version 0.23.1 in `pyproject.toml` and the adapter requires
that exact version. The adversarial audit repaired a >=0.23.1 dependency versus
0.23.0 adapter contradiction; a contract test now links both requirements.
Torch is a declared dependency for
the CPU PELSA backend. There is no checked-in environment lockfile.

## Actual test coverage and validation map

Read [testing.md](testing.md) for the automated strategy, commands, fixtures,
temporary known-defect tests and remaining gaps. `tests/` now covers numerical
algorithms, configuration/input/result contracts and ten end-to-end scientific
references. `.github/workflows/smoke-test.yml` runs the core and workflow suites;
release CI retains its template/build checks. `tests/constraints.txt` pins the
reference numerical environment without changing package dependency defaults.

Existing examples remain useful manual integration inputs. The automated Searle
fixture reduces its tracked input at runtime; the larger examples themselves
have no full-data expected-output assertions:

| Fixture directory | Run from that directory | Coverage |
| --- | --- | --- |
| `examples/searle_small/` | `proteoflux run --config config.yaml` | Spectronaut protein workflow, directLFQ, tag normalization, LC-ConMed, limma and exports |
| `examples/phospho_small/` | `proteoflux run --config config_phospho.yaml` | Site mapping/localization, separate flowthrough injection and covariate analysis; no kinase method configured |
| `examples/pelsa_small/` | `proteoflux run --config config_pelsa.yaml` | Peptide indexing, concentrations, CPU 4PL fitting, PELSA exports |

For safe manual checks, copy the chosen fixture directory to scratch space or
redirect exports via an in-memory config. Examples write result files and are
not tiny unit tests. Untracked `proteoflux/config*.yaml`, local data, wheels and
comparison scripts are not a portable validation suite; do not overwrite them.

For numerical changes, retain independent checks and compare matrices, masks,
identifiers and statistical outputs; a successful CLI exit is insufficient.
Schema changes also need H5AD write/read and downstream checks. ProteoViewer is
external; the testing context records the read-only consumer audit and its limits.
