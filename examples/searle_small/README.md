# ProteoFlux Example Dataset

This dataset is a reduced subset (1000 precursors per sample) of a DIA proteomics experiment.

It is intended solely for demonstrating ProteoFlux configuration and execution.

## Run

Run from this example directory, because paths in YAML are relative to the
working directory. From the repository root:

```bash
cd examples/searle_small
proteoflux run --config config.yaml
```

Check the `exports` paths before running; the command writes H5AD, XLSX and PDF
outputs to those locations. These data demonstrate a workflow, not a reference
for statistical calibration.
