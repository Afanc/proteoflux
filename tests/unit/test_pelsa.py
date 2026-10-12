import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import f as f_distribution

from proteoflux.analysis.pelsa_pipeline import _build_pelsa_ratio_table, _control_log10_position
from proteoflux.analysis.pelsa_torch import (
    four_pl_numpy, four_pl_jacobian_numpy, four_pl_torch, add_curve_statistics,
    build_dense_torch_input, fit_4pl_torch_from_ratio_df,
)

pytestmark = pytest.mark.unit


def test_reference_is_linear_control_mean_and_normalized_layer():
    obj = ad.AnnData(np.zeros((4, 2)), obs=pd.DataFrame({"Concentration": [0., 0., .1, 1.]},
                                                      index=["c1", "c2", "d1", "d2"]),
                     var=pd.DataFrame(index=["peptide", "no_control"]))
    obj.layers["normalized"] = np.array([[1., np.nan], [3., np.nan], [4., 8.], [5., 9.]])
    result = _build_pelsa_ratio_table(obj)
    assert set(result.peptide_id) == {"peptide"}
    assert_allclose(result.log2_ratio, np.array([1., 3., 4., 5.])-np.log2(5))
    assert_allclose(result.ratio, [2/5, 8/5, 16/5, 32/5])
    assert_allclose(result.log10_concentration, [-2, -2, -1, 0])


def test_ratios_ignore_feature_scale_and_dose_units_shift_log_axis_only():
    obj = ad.AnnData(np.zeros((5, 2)), obs=pd.DataFrame(
        {"Concentration": [0., 0., .1, 1., 10.]}, index=list("abcde")),
        var=pd.DataFrame(index=["up", "down"]))
    obj.layers["normalized"] = np.array([[1., 5.], [3., 7.], [4., 4.], [5., 3.], [6., 2.]])
    baseline = _build_pelsa_ratio_table(obj).set_index(["peptide_id", "sample"]).sort_index()
    changed = obj[[4, 0, 2, 1, 3], ::-1].copy()
    changed.layers["normalized"] += np.array([7., -2.])
    changed.obs["Concentration"] *= 1000
    result = _build_pelsa_ratio_table(changed).set_index(["peptide_id", "sample"]).sort_index()
    pd.testing.assert_index_equal(result.index, baseline.index)
    assert_allclose(result.log2_ratio, baseline.log2_ratio, atol=1e-12)
    assert_allclose(result.ratio, baseline.ratio, atol=1e-12)
    assert_allclose(result.log10_concentration, baseline.log10_concentration+3, atol=1e-12)
    # EC50 is expressed in the supplied concentration units.
    x = np.array([-2., -1., 0., 1.])
    assert_allclose(four_pl_numpy(x+3, -2., 1.2, 0., -2.),
                    four_pl_numpy(x, 1., 1.2, 0., -2.), atol=1e-12)


@pytest.mark.parametrize("doses, expected", [([.1, 1, 10, 10], -2), ([10], 0), ([.01, .1, 100], -4)])
def test_control_pseudodose_uses_median_distinct_log_spacing(doses, expected):
    assert _control_log10_position(np.array(doses)) == expected


@pytest.mark.parametrize("invalid", [-1., np.nan, np.inf, -np.inf])
def test_invalid_noncontrol_doses_cannot_be_silently_dropped(invalid):
    obj = ad.AnnData(np.zeros((4, 1)), obs=pd.DataFrame(
        {"Concentration": [0., .1, 1., invalid]}, index=list("abcd")))
    obj.layers["normalized"] = np.ones(obj.shape)
    with pytest.raises(ValueError, match="finite.*positive|positive.*finite"):
        _build_pelsa_ratio_table(obj)


def test_explicit_negative_control_sentinel_remains_supported():
    obj = ad.AnnData(np.zeros((4, 1)), obs=pd.DataFrame(
        {"Concentration": [-1., .1, 1., 10.]}, index=list("abcd")))
    obj.layers["normalized"] = np.ones(obj.shape)
    result = _build_pelsa_ratio_table(obj, control_concentration=-1)
    assert result.is_control.tolist() == [True, False, False, False]
    assert_allclose(result.log10_concentration, [-2., -1., 0., 1.])


def test_4pl_sign_midpoint_and_jacobian():
    x = np.array([-4., -2., -1., 0., 2.])
    params = np.array([1., 1.2, .5, -2.])
    assert four_pl_numpy(-params[0], *params) == pytest.approx((params[2]+params[3])/2)
    expected = four_pl_numpy(x, *params)
    jac = four_pl_jacobian_numpy(x, *params)
    finite_difference = np.column_stack([
        (four_pl_numpy(x, *(params+np.eye(4)[i]*1e-6))-
         four_pl_numpy(x, *(params-np.eye(4)[i]*1e-6)))/2e-6 for i in range(4)])
    assert_allclose(jac, finite_difference, rtol=2e-7, atol=1e-9)
    import torch
    assert_allclose(four_pl_torch(torch, torch.tensor(x), *[torch.tensor(p) for p in params]), expected, rtol=1e-12)


def test_recalibrated_test_and_excluded_zero_sse():
    frame = pd.DataFrame({"fit_success": [True, True, True, False], "n_points": [8, 8, 4, 8],
                          "sse_null": [20., 20., 20., 20.], "sse_curve": [2., 0., 2., 2.]})
    result = add_curve_statistics(frame)
    assert result.curve_f_value.iloc[0] == 18
    # Frozen distribution specification, distinct from a textbook nested F test.
    expected_p = f_distribution.sf(18, 5, (.8-1/36)*5.5, loc=.12)
    assert result.curve_p_value.iloc[0] == pytest.approx(expected_p)
    assert result.curve_q_value.iloc[0] == pytest.approx(expected_p)
    assert result.curve_p_value.iloc[1:].isna().all()


def test_distinct_doses_not_replicate_count_determine_eligibility():
    frame = pd.DataFrame({"peptide_id": ["p"]*8, "sample": [f"s{i}" for i in range(8)],
                          "log10_concentration": [-2]*4+[-1]*4, "log2_ratio": np.linspace(0, 1, 8),
                          "is_control": [True]*4+[False]*4})
    dense = build_dense_torch_input(frame)
    assert dense.n_concentrations.tolist() == [2]
    with pytest.raises(ValueError, match="eligible|four|4|concentration"):
        fit_4pl_torch_from_ratio_df(frame, {"analysis": {"pelsa": {}}})


def test_localization_length_counts_residues_not_ptm_annotation_characters():
    from proteoflux.analysis.pelsa_pipeline import _build_pelsa_localization_metadata
    obj = ad.AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["ACMK", "ACM[Oxidation (M)]K"]))
    payload = _build_pelsa_localization_metadata(obj)
    assert payload["feature_ids"].tolist() == obj.var_names.tolist()
    assert payload["peptide_length"].tolist() == [4, 4]
