# Canonical R limma references

These are independent R results, not snapshots of InMoose. ProteoFlux and normal
pytest/CI need no R, Rscript or rpy2. Inputs are authored, deterministic log2
matrices with explicit sample, feature, design-term and contrast labels.

| Case | Purpose |
| --- | --- |
| balanced | Two groups, three replicates each, heterogeneous feature variance |
| shared | Three groups, shared residual-variance estimation, B−A and C−A |
| batch | Categorical three-level batch crossed with condition; explicit dummy design |
| equal_variance | Identical residual variances, infinite prior df; reproduces the original moderation failure |
| missing | One feature lacks all B observations; another has two missing samples. R retains estimable contrasts and feature-specific residual df. |

Each folder contains `expression.csv` (features × samples), `samples.csv`,
`design.csv` (samples × terms), `contrasts.csv` (terms × contrasts) and
`expected.csv` (labelled feature/contrast results). Twenty-four features provide
heterogeneous variances for empirical Bayes without large datasets. The equal
variance matrix matches the original constant-residual-variance unit reproducer.
No imputation, fully-missing contrast neutralization or phospho residualization
is applied in this reference layer.

`generate_reference.R` calls `lmFit`, `contrasts.fit`, and default `eBayes`.
Raw SE/t/p use residual variance/df; moderated probabilities come directly from
limma. BH is explicitly per contrast over finite p-values, matching the declared
ProteoFlux testing family. Posterior variance, prior df and total df are included
for diagnosing moderation, although ProteoFlux does not serialize all of them.

Regenerate **only when reviewing an intentional reference change** from the repo root:

```bash
TZ=UTC Rscript tests/reference/limma/generate_reference.R
python -m pytest tests/contracts/test_limma_reference.py -q
```

Requires an independently installed R with limma; the script installs nothing.
It writes the reference CSVs, input MD5s and `sessionInfo.txt`. Review numerical,
missingness and version diffs. Current generation: **R 4.3.3, limma 3.65.1**, using
the existing local installation. `sessionInfo.txt` records BLAS/platform and
loaded packages; `input_checksums.tsv` ties results to the input files. Package
source commit was not available from the installed limma DESCRIPTION.

Python comparisons use float64 `rtol=1e-7`, `atol=1e-12`, allowing independent
LAPACK/special-function rounding, with exact NaN positions and labelled axes.
Balanced, shared and batch cases pass with InMoose 0.9.0. Two strict expected
failures remain: infinite-prior-df `KeyError(-2)` and a feature-specific singular
fit (`LinAlgError`) where canonical R provides partially estimable results.
Do not alter R expectations or add a ProteoFlux workaround to resolve these.
