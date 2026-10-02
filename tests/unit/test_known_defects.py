"""Desired behavior blocked by specific defects; strict XPASS demands review.

These are not golden expectations approving the current defect. Keep each
expected failure narrow; unrelated exception types still fail the suite.
"""
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.workflow.rollup.robust_regression import pivot_df_robust_regression
from tests.unit.test_statistics import expression

pytestmark = [pytest.mark.unit, pytest.mark.known_bug]


@pytest.mark.xfail(raises=KeyError, reason="inmoose 0.9.0 eBayes indexes t2 with ~True == -2 for infinite prior df")
def test_equal_residual_variances_still_allow_moderation():
    obj = expression()
    obj.X = np.tile(np.array([-1., 0., 1., 0., 1., 2., -2., -1., 0.])[:, None], (1, obj.n_vars))
    result = run_limma_pipeline(obj, {})
    assert np.isfinite(result.varm["p_ebayes"]).all()


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
