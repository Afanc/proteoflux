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
    actual = imputer(mode="standard", lod_k=2, lod_sd_width=0).fit_transform(x)
    assert_array_equal(x, original)
    assert_array_equal(actual[~np.isnan(x)], x[~np.isnan(x)])
    assert_allclose(actual[0], [4, 10, 4, 10])
    # The observed B group is empty in row 2; filling its first cell must not
    # turn it into observed evidence when filling the second cell.
    assert_allclose(actual[1, [1, 3]], [2.8, 2.8], atol=1e-7)


def test_integer_threshold_precedes_fraction_and_fraction_uses_ceiling():
    x = np.array([[10., 3., np.nan, 4.], [1., 2., 3., 4.]])
    integer = imputer(mode="standard", in_min_obs=1, frac_min_obs=1., lod_k=2).fit_transform(x)
    fraction = imputer(mode="standard", in_min_obs=None, frac_min_obs=.75, lod_k=2).fit_transform(x)
    assert integer[0, 2] == 10
    assert fraction[0, 2] <= 1.5


def test_seeded_draws_repeat_without_modifying_observed_values():
    x = np.array([[2., np.nan, 5., np.nan], [np.nan, 6., 7., 8.]])
    a = imputer(mode="standard").fit_transform(x)
    b = imputer(mode="standard").fit_transform(x)
    assert_array_equal(a, b)
    assert_allclose(a[~np.isnan(x)], x[~np.isnan(x)])
    assert a[1, 0] == 7  # The only observed A value clamps the draw to itself.


def test_lc_conmed_log_offset_equivariance_with_mar_and_mnar_draws():
    x = np.array([[2., np.nan, 5., np.nan], [np.nan, 6., 7., 8.],
                  [np.nan, 4., np.nan, 6.]])
    baseline = imputer(mode="standard").fit_transform(x)
    shifted = imputer(mode="standard").fit_transform(x + 3.)
    assert_allclose(shifted, baseline + 3., rtol=0, atol=1e-12)
    assert_array_equal(baseline[np.isfinite(x)], x[np.isfinite(x)])


def test_all_missing_lod_and_pooled_sample_variance():
    out = imputer(mode="standard", lod_sd_width=0).fit_transform(np.full((2, 4), np.nan))
    assert_allclose(out, -.2, atol=1e-7)
    # Within-group sums of squares: 8 and 18, total df=2.
    assert _pooled_sd([np.array([1., 5.]), np.array([2., 8.])]) == pytest.approx(np.sqrt(13))


@pytest.fixture
def three_replicate_conditions():
    samples = ["a1", "a2", "a3", "b1", "b2", "b3"]
    return {"sample_index": samples,
            "condition_map": pd.DataFrame({"Sample": samples,
                                            "CONDITION": ["A"]*3 + ["B"]*3}).iloc[::-1]}


@pytest.mark.parametrize("threshold", [{}, {"in_min_obs": 3},
                                        {"in_min_obs": None, "frac_min_obs": 1.}])
def test_mnar_only_preserves_partial_and_fully_missing_features(three_replicate_conditions, threshold):
    x = np.array([[np.nan, np.nan, 2.5, 3., 3., 3.],
                  [np.nan, np.nan, np.nan, 3., 3., 3.],
                  [np.nan]*6])
    before = x.copy()
    model = LC_ConMedImputer(**three_replicate_conditions, mode="mnar_only", lod_k=2, **threshold)
    actual = model.fit_transform(x)
    assert_array_equal(actual[0], x[0])
    assert_array_equal(actual[2], x[2])
    assert_array_equal(actual[~np.isnan(x)], x[~np.isnan(x)])
    # LoD = median(2.5, 3); only the absent condition consumes seeded draws.
    expected_lod = np.minimum(2.75 - .2 + np.random.default_rng(42).normal(0, .05, 3), 2.75)
    assert_array_equal(actual[1, :3], expected_lod)
    assert_array_equal(model.fit_transform(x), actual)
    assert_array_equal(x, before)


def test_mnar_only_leaves_an_entirely_missing_matrix_unchanged():
    x = np.full((2, 4), np.nan)
    assert_array_equal(imputer(mode="mnar_only").fit_transform(x), x)


def test_constructor_and_factory_default_to_mnar_only(three_replicate_conditions):
    x = np.array([[2., np.nan, 6., 8., np.nan, 14.],
                  [np.nan]*3 + [3., 4., 5.], [np.nan]*6])
    expected = LC_ConMedImputer(**three_replicate_conditions, mode="mnar_only", lod_k=2).fit_transform(x)
    constructor = LC_ConMedImputer(**three_replicate_conditions, lod_k=2).fit_transform(x)
    factory = get_imputer(method="lc_conmed", lc_conmed_lod_k=2,
                          **three_replicate_conditions).fit_transform(x)
    assert_array_equal(constructor, expected)
    assert_array_equal(factory, expected)


def test_standard_mode_preserves_original_seeded_values(three_replicate_conditions):
    x = np.array([[2., np.nan, 6., 8., np.nan, 14.],
                  [np.nan]*3 + [3., 4., 5.], [np.nan]*6])
    # Recorded before adding mode: MAR, absent-condition LoD, all-missing LoD.
    expected = np.array([
        [2., 4.219734611112851, 6., 8., 10.25005679585646, 14.],
        [2.337522559790323, 2.3470282358195607, 2.202448240567308, 3., 4., 5.],
        [2.234891024656884, 2.306392020158364, 2.2841878703828207,
         2.2991599421247852, 2.257347803621321, 2.343969898743141],
    ])
    standard = LC_ConMedImputer(**three_replicate_conditions, mode="standard", lod_k=2).fit_transform(x)
    explicit = get_imputer(method="lc_conmed", lc_conmed_mode="standard",
                           lc_conmed_lod_k=2, **three_replicate_conditions).fit_transform(x)
    assert_allclose(standard, expected, rtol=0, atol=1e-14)
    assert_array_equal(explicit, standard)


@pytest.mark.parametrize("mode", [None, "mnar", "STANDARD", True])
def test_invalid_lc_conmed_modes_fail(mode):
    with pytest.raises(ValueError, match="lc_conmed_mode"):
        imputer(mode=mode).fit(np.ones((2, 4)))


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_mnar_only_provenance_survives_preprocessing_and_h5ad(protein_config, tmp_path, explicit_mode):
    import anndata as ad
    from proteoflux.workflow.dataset import Dataset

    rows = pd.read_csv(protein_config["dataset"]["input_file"], sep="\t")
    rows.loc[rows.protein == "P011", "intensity"] = np.nan
    rows.to_csv(protein_config["dataset"]["input_file"], sep="\t", index=False)
    options = protein_config["preprocessing"]["imputation"]
    options["method"] = None
    before = Dataset(**protein_config).get_anndata()
    options.update(method="lc_conmed", lc_conmed_mode="mnar_only")
    if not explicit_mode:
        options.pop("lc_conmed_mode")
    dataset = Dataset(**protein_config)
    out = dataset.get_anndata()
    expected_fill = np.zeros(out.shape, dtype=bool)
    for feature, conditions in [("P001", ["A"]), ("P002", ["A", "B"])]:
        expected_fill[out.obs.CONDITION.isin(conditions), out.var_names.get_loc(feature)] = True
    assert_array_equal(np.isnan(before.X) & np.isfinite(out.X), expected_fill)
    assert_array_equal(out.X[~expected_fill], before.X[~expected_fill])
    imputed_only = dataset.preprocessor.intermediate_results.matrices["imputed_only"].T
    assert_array_equal(np.isfinite(imputed_only), expected_fill)
    assert_allclose(imputed_only[expected_fill], out.X[expected_fill], rtol=1e-7)
    path = tmp_path / "mnar_only.h5ad"
    out.write_h5ad(path)
    restored = ad.read_h5ad(path)
    assert_array_equal(restored.X, out.X)
    assert_array_equal(restored.obs_names, before.obs_names)
    assert_array_equal(restored.var_names, before.var_names)
    for layer in ("raw", "normalized", "qvalue"):
        assert_array_equal(restored.layers[layer], before.layers[layer])
    assert restored.uns["preprocessing"]["imputation"]["lc_conmed_mode"] == "mnar_only"


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("explicit_mode", [False, True])
def test_mnar_only_reaches_injected_phospho_background_and_adjustment(tmp_path, reuse, explicit_mode):
    from proteoflux.analysis.limma_pipeline import run_limma_pipeline
    from proteoflux.workflow.dataset import Dataset
    from tests.fixtures.workflows import phospho_workflow

    config = phospho_workflow(tmp_path, reuse=reuse, kinase=False)
    path = config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t", dtype={"sites": str, "positions": str, "probabilities": str})
    rows.loc[(rows.protein == "P000") & (rows.condition == "A")
             & (rows.replicate == 3) & rows.sites.notna(), "intensity"] = np.nan
    rows.to_csv(path, sep="\t", index=False)
    config["preprocessing"]["imputation"]["lc_conmed_mode"] = "mnar_only"
    if not explicit_mode:
        config["preprocessing"]["imputation"].pop("lc_conmed_mode")
    obj = Dataset(**config).get_anndata()
    for before, after in [(obj.layers["normalized"], obj.X),
                          (obj.layers["normalized_covariate"], obj.layers["processed_covariate"])]:
        expected_fill = np.zeros(obj.shape, dtype=bool)
        for condition in obj.obs.CONDITION.unique():
            selected = (obj.obs.CONDITION == condition).to_numpy()
            absent = np.isnan(before[selected]).all(axis=0) & np.isfinite(before).any(axis=0)
            expected_fill[selected] = absent
        assert expected_fill.any()
        assert_array_equal(np.isnan(before) & np.isfinite(after), expected_fill)
        assert_array_equal(after[~expected_fill], before[~expected_fill])
    s, t = [obj.var_names.get_loc(name) for name in ("P000|S6", "P000|T6")]
    assert_array_equal(obj.layers["processed_covariate"][:, s], obj.layers["processed_covariate"][:, t])
    assert np.isnan(obj.layers["processed_covariate"][obj.obs_names.get_loc("A_2"), s])
    assert np.isnan(obj.X[obj.obs_names.get_loc("A_3"), s])
    result = run_limma_pipeline(obj, config)
    paired = np.isfinite(obj.X) & np.isfinite(obj.layers["processed_covariate"])
    assert_array_equal(np.isfinite(result.layers["residuals_covariate"]), paired)
    assert np.isfinite(result.varm["log2fc"][[s, t]]).all()
    assert np.isfinite(result.varm["cov_part"][[s, t]]).all()


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


def test_iterative_hgb_factory_works_in_a_fresh_process_and_preserves_observations():
    import subprocess
    import sys
    # Other imports can enable sklearn's experimental class and hide this defect.
    script = '''
import numpy as np
from numpy.testing import assert_array_equal
from proteoflux.workflow.imputer_factory import get_imputer
x = np.array([[1., 2., 3.], [2., np.nan, 4.], [4., 5., 7.], [6., 8., 9.]])
result = get_imputer(method="iterative_hgb", hgb_max_iter=2, hgb_estimator_max_iter=2).fit_transform(x)
assert result.shape == x.shape and np.isfinite(result).all()
assert_array_equal(result[~np.isnan(x)], x[~np.isnan(x)])
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
