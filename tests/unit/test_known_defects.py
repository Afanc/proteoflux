"""Desired behavior blocked by specific defects; strict XPASS demands review.

These are not golden expectations approving the current defect. Keep each
expected failure narrow; unrelated exception types still fail the suite.
"""
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.stats import false_discovery_control

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.analysis.nrsc import compute_nrsc
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.workflow.rollup.robust_regression import pivot_df_robust_regression
from tests.unit.test_statistics import expression
from tests.unit.test_covariate_statistics import covariate_experiment

pytestmark = [pytest.mark.unit, pytest.mark.known_bug]


@pytest.mark.xfail(raises=KeyError, reason="inmoose 0.9.0 eBayes indexes t2 with ~True == -2 for infinite prior df")
def test_equal_residual_variances_still_allow_moderation():
    obj = expression()
    obj.X = np.tile(np.array([-1., 0., 1., 0., 1., 2., -2., -1., 0.])[:, None], (1, obj.n_vars))
    result = run_limma_pipeline(obj, {})
    assert np.isfinite(result.varm["p_ebayes"]).all()


@pytest.mark.xfail(raises=NameError, reason="nrsc missing-layer warning uses unimported log_warning")
def test_optional_spectral_counts_can_be_absent():
    obj = expression()
    del obj.layers["spectral_counts"]
    assert compute_nrsc(obj, ["A_vs_B"]) is None


@pytest.mark.xfail(raises=AssertionError, reason="Disconnected QR pruning makes absolute protein quantities fail uniform intensity scaling")
def test_disconnected_robust_rollup_preserves_uniform_intensity_scaling():
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    def rollup(frame):
        return pivot_df_robust_regression(pep_wide=frame, protein_col="protein", peptide_col="peptide",
                                         sample_cols=["a", "b"], num_cores=1, huber_t=1.345,
                                         max_iter=20, tol=1e-8, min_nonan=1).select("a", "b").to_numpy()
    baseline = rollup(peptides)
    scaled = rollup(peptides.with_columns(pl.col("a", "b") * 2))
    # Changing intensity units must not turn a twofold scale factor into fourfold.
    assert_allclose(scaled, 2*baseline, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize("mode", ["free", "nonneg"])
@pytest.mark.xfail(raises=AssertionError, reason="Fully imputed FT slope p-values are neutralized after BH, leaving other diagnostic q-values stale")
def test_fully_imputed_background_diagnostics_apply_bh_after_neutralization(covariate_experiment, mode):
    obj, config = covariate_experiment
    config["ancova"] = {"beta_mode": mode}
    baseline = run_limma_pipeline(obj, config)
    feature = np.argmin(baseline.varm["covariate_p"][:, 0])
    # A non-flat imputed analysis vector is not evidence of observed background.
    obj.layers["raw_covariate"][:, feature] = np.nan
    result = run_limma_pipeline(obj, config)
    p = result.varm["covariate_p"][:, 0]
    assert p[feature] == 1
    assert_allclose(result.varm["covariate_q"][:, 0], false_discovery_control(p), rtol=1e-12, atol=1e-14)


@pytest.mark.xfail(raises=NameError, reason="PyArrow CSV loader refers to unimported pv_csv")
def test_pyarrow_delimited_loader_can_read_supported_input(protein_config):
    protein_config["dataset"]["load_method"] = "pyarrow"
    assert Dataset(**protein_config).get_anndata().n_vars == 12
