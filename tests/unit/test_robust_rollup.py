"""Robust solver numerics and identifiable effects in disconnected designs.

QR pruning does not establish a common scale between disconnected components.
These checks do not endorse the resulting absolute protein quantities; the
separate scale-equivariance failure remains in test_known_defects.py.
"""
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from proteoflux.workflow.rollup.robust_regression import (
    _fit_rlm_numpy_irls_explicit,
    pivot_df_robust_regression,
)

pytestmark = pytest.mark.unit


def test_disconnected_robust_rollup_produces_result():
    # Smallest disconnected graph: two peptides, two samples, two observations.
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    out = pivot_df_robust_regression(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                                    sample_cols=["a", "b"], num_cores=1, huber_t=1.345,
                                    max_iter=20, tol=1e-8, min_nonan=1)
    assert out.shape == (1, 3)  # original crash regression
    assert out.columns == ["protein", "a", "b"]
    assert out["protein"].to_list() == ["P"]
    values = out.select("a", "b").to_numpy()[0]
    # The retained design has one sample coefficient and one peptide effect.
    # The dropped sample coefficient must remain missing, not become 2**0 = 1.
    assert np.isnan(values).sum() == 1
    assert (values[np.isfinite(values)] > 0).all()


@pytest.mark.parametrize("sample_order", [["A", "B", "C", "D"], ["D", "C", "B", "A"], ["B", "D", "A", "C"]])
def test_disconnected_components_preserve_identifiable_within_component_ratios(sample_order):
    # Component 1 has a twofold B/A change; component 2 has a fourfold D/C
    # change. Each has two peptides differing by a constant twofold response.
    peptides = pl.DataFrame({"protein": ["P"]*4, "peptide": ["a", "b", "c", "d"],
                             "A": [8., 16., None, None], "B": [16., 32., None, None],
                             "C": [None, None, 32., 64.], "D": [None, None, 128., 256.]})
    out = pivot_df_robust_regression(pep_wide=peptides.reverse(), protein_col="protein", peptide_col="peptide",
                                    sample_cols=sample_order, num_cores=1, huber_t=1.345,
                                    max_iter=20, tol=1e-8, min_nonan=2)
    assert out.columns == ["protein", *sample_order]
    row = out.row(0, named=True)
    assert row["protein"] == "P"
    assert sum(row[s] is None for s in sample_order) == 1
    checked = 0
    for numerator, denominator, ratio in [("B", "A", 2.), ("D", "C", 4.)]:
        if row[numerator] is not None and row[denominator] is not None:
            assert_allclose(row[numerator]/row[denominator], ratio, rtol=1e-10)
            checked += 1
    assert checked >= 1


@pytest.mark.parametrize("max_iter", [0, 1, 20])
def test_explicit_solver_starts_from_ols_and_fits_retained_design(max_iter):
    # The smallest example's retained equations are a+p=3 and -p=5.
    # Coefficients are coordinates in that reduced basis, not independently
    # identified sample abundances in the original disconnected model.
    x = np.array([[1., 1.], [0., -1.]])
    y = np.array([3., 5.])
    beta, iterations = _fit_rlm_numpy_irls_explicit(x, y, huber_t=1.345, max_iter=max_iter, tol=1e-8)
    assert_allclose(beta, [8., -5.], atol=1e-12, rtol=0)
    assert_allclose(x @ beta, y, atol=1e-12, rtol=0)
    assert 0 <= iterations <= max_iter


def test_explicit_huber_fit_matches_independent_location_solution():
    y = np.array([-2., -1., 0., 1., 2., 100.])
    # With only an intercept, centering residuals leaves MAD(y)=1.5 fixed.
    # Five inlier residuals sum to -5*mu; the outlier contributes +1.345*scale.
    # Thus the Huber score is zero at mu=1.345*(1.4826*1.5)/5.
    expected = 1.345 * (1.4826 * 1.5) / 5
    beta, _ = _fit_rlm_numpy_irls_explicit(np.ones((6, 1)), y, huber_t=1.345, max_iter=100, tol=1e-12)
    assert_allclose(beta, [expected], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("min_nonan", [1, 2])
def test_disconnected_rollup_keeps_unobserved_samples_missing(min_nonan):
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.],
                             "zero": [0., 0.], "absent": [np.nan, np.nan]})
    out = pivot_df_robust_regression(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                                    sample_cols=["a", "b", "zero", "absent"], num_cores=1, huber_t=1.345,
                                    max_iter=20, tol=1e-8, min_nonan=min_nonan)
    assert out["zero"].item() is None
    assert out["absent"].item() is None
    if min_nonan == 2:
        assert out["a"].item() is None and out["b"].item() is None
