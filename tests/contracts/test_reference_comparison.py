"""Check that regression machinery catches scientific changes, including NaNs."""
import numpy as np
import pytest

from tests.full_regression import compare_full_results
from tests.scientific_reference import assert_reference, scientific_payload

pytestmark = pytest.mark.contract


def test_full_comparison_aligns_labels_and_rejects_numerical_drift(workflow_results):
    obj, _ = workflow_results("proteomics")
    compare_full_results(obj.copy(), obj)
    shuffled = obj[::-1, ::-1].copy()
    # This unlabelled uns vector must follow var order explicitly; evidence
    # tables have independent labels and must not be blindly sliced.
    shuffled.uns["residual_variance"] = np.asarray(obj.uns["residual_variance"])[::-1]
    compare_full_results(shuffled, obj)
    changed = obj.copy()
    changed.X[0, 0] += .1
    with pytest.raises(AssertionError, match="X"):
        compare_full_results(changed, obj)
    changed = obj.copy()
    index = tuple(np.argwhere(np.isfinite(changed.layers["raw"]))[0])
    changed.layers["raw"][index] = np.nan
    with pytest.raises(AssertionError):
        compare_full_results(changed, obj)


@pytest.mark.parametrize("name", ["phospho", "phosx", "pelsa"])
def test_full_comparison_handles_specialized_tables(name, workflow_results):
    obj, _ = workflow_results(name)
    compare_full_results(obj.copy(), obj)


@pytest.mark.parametrize("field", ["p_raw", "raw_p_ebayes", "ft_q_raw", "pvalue", "curve_q_value",
                                   "covariate_p", "covariate_q"])
def test_reference_comparison_does_not_use_intensity_tolerance_for_probabilities(field):
    with pytest.raises(AssertionError):
        assert_reference({field: [1.5e-10]}, {field: [1e-10]})


@pytest.mark.parametrize("field", ["covariate_p", "covariate_q"])
def test_covariate_probabilities_are_included_in_both_comparators(workflow_results, field):
    obj, _ = workflow_results("phospho")
    changed = obj.copy()
    changed.varm[field][0, 0] = .12345
    with pytest.raises(AssertionError, match=field):
        assert_reference(scientific_payload(changed), scientific_payload(obj))
    with pytest.raises(AssertionError, match=field):
        compare_full_results(changed, obj)


def test_significance_boundary_is_checked_even_within_float_tolerance(workflow_results):
    obj, _ = workflow_results("proteomics")
    obj = obj.copy()
    obj.varm["q_ebayes"][0, 0] = .0499999
    expected = scientific_payload(obj)
    obj.varm["q_ebayes"][0, 0] = .0500001
    with pytest.raises(AssertionError, match="q_ebayes_below_0.05"):
        assert_reference(scientific_payload(obj), expected)
