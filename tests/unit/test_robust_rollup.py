"""Robust fitting, explicit disconnected policy and component-local quantities.

Independent component estimates do not establish between-component fold changes.
"""
import warnings

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.workflow.rollup.robust_regression import (
    DisconnectedObservationError,
    DisconnectedObservationWarning,
    _fit_rlm_numpy_irls_explicit,
    pivot_df_robust_regression,
)

pytestmark = pytest.mark.unit


def rollup(peptides, sample_cols, **kwargs):
    options = dict(num_cores=1, huber_t=1.345, max_iter=20, tol=1e-8, min_nonan=1)
    options.update(kwargs)
    return pivot_df_robust_regression(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                                     sample_cols=sample_cols, **options)


@pytest.mark.parametrize("options", [{}, {"disconnected_policy": "raise"}])
def test_disconnected_rollup_rejects_unidentifiable_between_component_scale(options):
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    with pytest.raises(DisconnectedObservationError) as error:
        rollup(peptides, ["a", "b"], **options)
    assert "protein 'P'" in str(error.value)
    assert "[['a'], ['b']]" in str(error.value)
    assert "not identifiable" in str(error.value)


def test_disconnected_robust_rollup_produces_result():
    # Smallest disconnected graph: two peptides, two samples, two observations.
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    with pytest.warns(DisconnectedObservationWarning, match="compare samples only within"):
        out = rollup(peptides, ["a", "b"], disconnected_policy="component")
    assert out.shape == (1, 3)  # original crash regression
    assert out.columns == ["protein", "a", "b"]
    assert out["protein"].to_list() == ["P"]
    values = out.select("a", "b").to_numpy()[0]
    # Each singleton component has peptide effect zero, retaining both samples.
    assert_allclose(values, [8., 32.], rtol=1e-12)


@pytest.mark.parametrize("factor", [.125, 2., 13.])
def test_disconnected_robust_rollup_preserves_uniform_intensity_scaling(factor):
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    with pytest.warns(DisconnectedObservationWarning):
        baseline = rollup(peptides, ["a", "b"], disconnected_policy="component").select("a", "b").to_numpy()
    with pytest.warns(DisconnectedObservationWarning):
        scaled = rollup(peptides.with_columns(pl.col("a", "b") * factor), ["a", "b"],
                        disconnected_policy="component").select("a", "b").to_numpy()
    assert_allclose(baseline, [[8., 32.]], rtol=1e-12)
    # Original xfail assertion, now under the explicit component policy.
    assert_allclose(scaled, factor*baseline, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize("sample_order", [["A", "B", "C", "D"], ["D", "C", "B", "A"], ["B", "D", "A", "C"]])
def test_disconnected_components_preserve_identifiable_within_component_ratios(sample_order):
    # Component 1 has a twofold B/A change; component 2 has a fourfold D/C
    # change. Each has two peptides differing by a constant twofold response.
    peptides = pl.DataFrame({"protein": ["P"]*4, "peptide": ["a", "b", "c", "d"],
                             "A": [8., 16., None, None], "B": [16., 32., None, None],
                             "C": [None, None, 32., 64.], "D": [None, None, 128., 256.]})
    with pytest.warns(DisconnectedObservationWarning):
        out = rollup(peptides.reverse(), sample_order, min_nonan=2, disconnected_policy="component")
    assert out.columns == ["protein", *sample_order]
    row = out.row(0, named=True)
    assert row["protein"] == "P"
    expected = dict(zip("ABCD", np.sqrt([8*16, 16*32, 32*64, 128*256])))
    assert_allclose([row[s] for s in sample_order], [expected[s] for s in sample_order], rtol=1e-12)
    for numerator, denominator, ratio in [("B", "A", 2.), ("D", "C", 4.)]:
        assert_allclose(row[numerator]/row[denominator], ratio, rtol=1e-10)
    # Rescaling one component cannot change the other component's quantities.
    with pytest.warns(DisconnectedObservationWarning):
        scaled = rollup(peptides.with_columns(pl.col("A", "B") * 7), sample_order,
                        min_nonan=2, disconnected_policy="component").row(0, named=True)
    expected.update(A=expected["A"]*7, B=expected["B"]*7)
    assert_allclose([scaled[s] for s in sample_order], [expected[s] for s in sample_order], rtol=1e-12)


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
    with pytest.warns(DisconnectedObservationWarning):
        out = rollup(peptides, ["a", "b", "zero", "absent"],
                     min_nonan=min_nonan, disconnected_policy="component")
    assert out["zero"].item() is None
    assert out["absent"].item() is None
    if min_nonan == 2:
        assert out["a"].item() is None and out["b"].item() is None
    else:
        assert_allclose(out.select("a", "b").to_numpy(), [[8., 32.]], rtol=1e-12)


@pytest.mark.parametrize("min_nonan", [1, 2, 3])
def test_sparse_bridges_and_three_components_keep_labels_and_evidence_threshold(min_nonan):
    sample_log = np.array([4., 5., 7., 6., 8., 9., 0.])
    # Peptide effects sum to zero separately in each of three components.
    peptide_log = np.array([-2., .5, 1.5, -1., 1., 0.])
    values = np.full((6, 7), np.nan)
    for p, s in [(0,0), (1,0), (1,1), (2,1), (2,2), (3,3), (4,3), (3,4), (4,4), (5,5)]:
        values[p,s] = 2**(sample_log[s] + peptide_log[p])
    # Invalid observations must not create a fourth component or evidence.
    values[:,6] = [0., -1., np.inf, -np.inf, np.nan, 0.]
    samples = list("ABCDEFG")
    peptides = pl.DataFrame({"protein": ["P"]*6, "peptide": list("abcdef"),
                             **dict(zip(samples, values.T))})
    expected = np.exp2(sample_log)
    expected[np.sum(np.isfinite(values) & (values > 0), axis=0) < min_nonan] = np.nan
    with pytest.warns(DisconnectedObservationWarning, match="3 disconnected"):
        result = rollup(peptides, samples, disconnected_policy="component", min_nonan=min_nonan)
    assert_allclose(result.select(samples).to_numpy()[0], expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("policy", ["raise", "component"])
def test_unobserved_protein_remains_missing_without_disconnected_warning(policy):
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [0., -1.], "b": [np.nan, np.inf]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rollup(peptides, ["a", "b"], disconnected_policy=policy)
    assert not any(isinstance(w.message, DisconnectedObservationWarning) for w in caught)
    assert np.isnan(result.select("a", "b").to_numpy()).all()


@pytest.mark.parametrize("policy", [None, False, "", "ignore", ["component"]])
def test_invalid_policy_is_rejected_even_for_empty_input(policy):
    peptides = pl.DataFrame(schema={"protein": pl.String, "peptide": pl.String, "a": pl.Float64})
    with pytest.raises(ValueError, match="disconnected_policy must be"):
        rollup(peptides, ["a"], disconnected_policy=policy)


def test_component_robust_fit_agrees_with_separate_connected_fits():
    rng = np.random.default_rng(94)
    left = np.exp2(rng.normal(8, 1, (5, 4)))
    right = np.exp2(rng.normal(13, .5, (3, 2)))
    left[1,2] *= 1000
    right[2,0] *= 200

    def frame(values):
        samples = [f"s{i}" for i in range(values.shape[1])]
        return pl.DataFrame({"protein": ["P"]*len(values), "peptide": [f"p{i}" for i in range(len(values))],
                             **dict(zip(samples, values.T))}), samples

    expected = np.r_[rollup(*frame(left)).drop("protein").to_numpy()[0],
                     rollup(*frame(right)).drop("protein").to_numpy()[0]]
    values = np.full((8, 6), np.nan)
    values[:5,:4], values[5:,4:] = left, right
    with pytest.warns(DisconnectedObservationWarning):
        result = rollup(*frame(values), disconnected_policy="component")
    assert_array_equal(result.drop("protein").to_numpy()[0], expected)
