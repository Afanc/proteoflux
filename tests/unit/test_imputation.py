import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.workflow.imputer_factory import get_imputer
from proteoflux.workflow.imputers.lc_conmed_imputer import LC_ConMedImputer, _pooled_sd

pytestmark = pytest.mark.unit


def imputer(**options):
    return LC_ConMedImputer(
        condition_map=pd.DataFrame({"Sample": ["b2", "a2", "b1", "a1"],
                                    "CONDITION": ["B", "A", "B", "A"]}),
        sample_index=["a1", "b1", "a2", "b2"], random_state=42, **options)


def test_condition_labels_and_observed_mask_survive_imputation():
    x = np.array([[4., 10., np.nan, np.nan], [2., np.nan, 6., np.nan]])
    original = x.copy()
    actual = imputer(lod_k=2, lod_sd_width=0).fit_transform(x)
    assert_array_equal(x, original)
    assert_array_equal(actual[~np.isnan(x)], x[~np.isnan(x)])
    assert_allclose(actual[0], [4, 10, 4, 10])
    # The observed B group is empty in row 2; filling its first cell must not
    # turn it into observed evidence when filling the second cell.
    assert_allclose(actual[1, [1, 3]], [2.8, 2.8], atol=1e-7)


def test_integer_threshold_precedes_fraction_and_fraction_uses_ceiling():
    x = np.array([[10., 3., np.nan, 4.], [1., 2., 3., 4.]])
    integer = imputer(in_min_obs=1, frac_min_obs=1., lod_k=2).fit_transform(x)
    fraction = imputer(in_min_obs=None, frac_min_obs=.75, lod_k=2).fit_transform(x)
    assert integer[0, 2] == 10
    assert fraction[0, 2] <= 1.5


def test_seeded_draws_repeat_without_modifying_observed_values():
    x = np.array([[2., np.nan, 5., np.nan], [np.nan, 6., 7., 8.]])
    a = imputer().fit_transform(x)
    b = imputer().fit_transform(x)
    assert_array_equal(a, b)
    assert_allclose(a[~np.isnan(x)], x[~np.isnan(x)])
    assert a[1, 0] == 7  # The only observed A value clamps the draw to itself.


def test_all_missing_lod_and_pooled_sample_variance():
    out = imputer(lod_sd_width=0).fit_transform(np.full((2, 4), np.nan))
    assert_allclose(out, -.2, atol=1e-7)
    # Within-group sums of squares: 8 and 18, total df=2.
    assert _pooled_sd([np.array([1., 5.]), np.array([2., 8.])]) == pytest.approx(np.sqrt(13))


@pytest.mark.parametrize("options", [{"in_min_obs": 0}, {"in_min_obs": True},
                                      {"in_min_obs": None, "frac_min_obs": 0},
                                      {"in_min_obs": None, "frac_min_obs": 1.1}])
def test_invalid_support_thresholds_fail(options):
    with pytest.raises(ValueError, match="in_min_obs|frac_min_obs"):
        imputer(**options).fit(np.ones((2, 4)))


def test_factory_omission_and_explicit_none_string_are_distinct():
    assert get_imputer() is None
    with pytest.raises(ValueError, match="Invalid imputation method"):
        get_imputer(method="none")


@pytest.mark.parametrize("method, expected", [("mean", 6), ("median", 3)])
def test_rowwise_simple_imputation_including_empty_row(method, expected):
    x = np.array([[2., np.nan, 3., 13.], [np.nan]*4])
    result = get_imputer(method=method).fit_transform(x)
    assert_allclose(result[0], [2, expected, 3, 13])
    assert_allclose(result[1], 0)
