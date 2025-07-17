intro_preprocessing_text="""
Before any downstream analysis, raw proteomics intensity data must undergo rigorous preprocessing to reduce technical noise and improve biological interpretability. This involves several sequential steps:

---

#### Filtering

Low-confidence peptide-spectrum matches (PSMs) are removed based on the following criteria:

- **PEP (Posterior Error Probability):**
  The estimated probability that a PSM is incorrect.

- **Q-value:**
  The minimum false discovery rate at which a PSM is considered significant.

- **Minimum number of PSMs per feature**
  (sometimes referred to as `run_evidence_count`):
  Ensures that each peptide or protein is supported by enough independent observations across samples.

---

#### Normalization

Normalization is performed in **two steps** to reduce systematic variation:

1. **Log₂ Transformation:**
   Converts the typically right-skewed intensity values into a more symmetric distribution and stabilizes variance across the dynamic range.

2. **Scaling:**
   Aligns samples for comparability using one of several methods:
   - Median equalization
   - Linear fitting (local or global)
   - LOESS fitting (local or global)

---

#### Imputation

Missing values are imputed using methods such as **truncated k-nearest neighbors (KNN)**.
This fills in gaps by borrowing information from similar proteins or samples, without inflating variance.

---

### What Follows

Each of the following visualizations corresponds to a key step in this preprocessing pipeline — from raw data filtering to final imputed intensities.
Together, they help diagnose artifacts, monitor sample quality, and assess whether normalization behaved as expected.

"""
log_transform_text="""
Proteomics intensity values span several orders of magnitude and are typically right-skewed.
Log2 transformation compresses this range, reduces skewness, and makes the data more symmetric.
It also stabilizes variance and allows fold changes to be interpreted as additive differences.

"""

before_after_normalization_violins="""These violin plots show the distribution of log2-transformed intensities before and after normalization, grouped by condition. After normalization, the distributions should become more centered and aligned across conditions. The dashed horizontal line marks the median intensity, which should stabilize if normalization was successful. Deviations from this pattern may indicate remaining batch effects or incomplete correction.
Here, violin plots display log2 intensities for individual samples before and after normalization. This view is useful for identifying sample-specific anomalies. After normalization, distributions should be comparable across samples. Samples with unusually broad distributions or shifted medians may be outliers or poorly normalized and should be reviewed carefully."""

normalization_metrics="""Both CV and rMAD are measures of variability:

    CV (Coefficient of Variation) = (Standard Deviation / Mean)

    rMAD (Relative Median Absolute Deviation) = (MAD / Median)
    where MAD is the median of absolute deviations from the median.

These metrics are computed across replicates within each condition. They quantify the consistency of protein intensities. In a well-normalized dataset:

    CV and rMAD should not increase after normalization.

    High CV/rMAD may reflect technical noise or biological heterogeneity.

We include both metrics because:

    CV is sensitive to outliers and skewed distributions.

    rMAD is more robust to outliers and provides a complementary view of spread.

By comparing CV and rMAD before and after normalization, you can assess whether normalization preserved biological signals while reducing technical variance."""
