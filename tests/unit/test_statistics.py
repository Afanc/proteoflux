"""Independent linear-model checks and experiment/contrast scope invariants."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import t as student_t

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.analysis.missingness import compute_missingness
from proteoflux.analysis.nrsc import compute_nrsc, compute_nrsc_misalignment
from proteoflux.analysis.stats_ops import bh_qvalues

pytestmark = pytest.mark.unit


def expression(conditions=("A",)*3+("B",)*3+("C",)*3):
    # Different residual variances prevent degenerate empirical-Bayes priors.
    rng = np.random.default_rng(281)
    values = rng.normal(size=(len(conditions), 24)) * np.linspace(.1, 1.8, 24)
    values += np.array([{"A": 0, "B": 1.5, "C": -.5}[c] for c in conditions])[:, None]
    obj = ad.AnnData(values, obs=pd.DataFrame({"CONDITION": conditions},
                                             index=[f"run{i}" for i in range(len(conditions))]),
                     var=pd.DataFrame(index=[f"protein{i}" for i in range(24)]))
    obj.layers["raw"] = np.exp2(values)
    obj.layers["normalized"] = values.copy()
    obj.layers["spectral_counts"] = np.ones_like(values)
    obj.uns["analysis"] = {"de_method": "limma_ebayes"}
    return obj


def test_raw_limma_agrees_with_independent_least_squares():
    obj = expression()
    obj.obs["batch"] = [10, 20, 30]*3  # numeric values are categorical
    result = run_limma_pipeline(obj, {"analysis": {"only_contrasts": ["B_v_A"],
                                                   "batch_effect_columns": ["BATCH"]}})
    design = np.column_stack([obj.obs.CONDITION == c for c in "ABC"] +
                             [obj.obs.batch == b for b in (20, 30)]).astype(float)
    beta, *_ = np.linalg.lstsq(design, obj.X, rcond=None)
    contrast = np.array([-1., 1., 0., 0., 0.])
    expected_fc = contrast @ beta
    residual = obj.X - design @ beta
    df = len(obj.obs) - np.linalg.matrix_rank(design)
    variance = (residual**2).sum(axis=0)/df
    expected_se = np.sqrt(variance * (contrast @ np.linalg.inv(design.T @ design) @ contrast))
    assert list(result.uns["contrast_names"]) == ["B_vs_A"]
    assert result.uns["analysis"]["design_rank"] == 5
    assert_allclose(result.varm["log2fc"][:, 0], expected_fc, rtol=1e-10, atol=1e-10)
    assert_allclose(result.varm["se_raw"][:, 0], expected_se, rtol=1e-10)
    assert_allclose(result.varm["p_raw"][:, 0], 2*student_t.sf(abs(expected_fc/expected_se), df), rtol=1e-10)
    assert_allclose(result.uns["residual_variance"], variance, rtol=3e-7)


def test_bh_is_per_contrast_with_known_ranks():
    p = np.array([[.01, .5], [.04, .5], [.03, .5], [.2, .5]])
    assert_allclose(bh_qvalues(p), [[.04, .5], [4*.04/3, .5], [4*.04/3, .5], [.2, .5]])


def test_contrast_selection_does_not_restrict_global_variance():
    obj = expression()
    changed = obj.copy()
    changed.X[6:] *= 50
    config = {"analysis": {"only_contrasts": "A_vs_B"}}
    global_a, global_b = [run_limma_pipeline(x, config) for x in (obj, changed)]
    assert_allclose(global_a.varm["log2fc"], global_b.varm["log2fc"])
    assert not np.allclose(global_a.varm["se_raw"], global_b.varm["se_raw"])
    config["analysis"]["separate_contrasts"] = True
    pair_a, pair_b = [run_limma_pipeline(x, config) for x in (obj, changed)]
    for key in ("log2fc", "se_raw", "p_ebayes", "q_ebayes"):
        assert_allclose(pair_a.varm[key], pair_b.varm[key], rtol=1e-12)
    assert global_a.uns["residual_variance"].shape == (24,)
    assert pair_a.uns["residual_variance"].shape == (24, 1)
    assert pair_a.uns["analysis"]["design_formula"] == "per-contrast"


@pytest.mark.parametrize('separate', [False, True])
def test_unselected_singleton_does_not_make_analysis_pilot(separate):
    obj = expression(("A",)*3+("B",)*3+("C",))
    out = run_limma_pipeline(obj, {"analysis": {"only_contrasts": ["A_v_B"],
                                               "separate_contrasts": separate}})
    assert not out.uns["pilot_study_mode"]
    assert list(out.uns["analysis"]["pilot_contrasts"]) == []
    assert list(out.uns['contrast_names']) == ['A_vs_B']
    assert_array_equal(out.obs_names, obj.obs_names)
    assert_array_equal(out.var_names, obj.var_names)
    # Pilot selection does not change variance-estimation scope.
    fit_obj = obj[:6] if separate else obj
    levels = ['A', 'B'] if separate else ['A', 'B', 'C']
    design = np.column_stack([fit_obj.obs.CONDITION == c for c in levels]).astype(float)
    beta = np.linalg.lstsq(design, fit_obj.X, rcond=None)[0]
    residual = fit_obj.X - design @ beta
    contrast = np.array([1., -1.] + ([] if separate else [0.]))
    df = fit_obj.n_obs - np.linalg.matrix_rank(design)
    se = np.sqrt((residual**2).sum(0)/df * (contrast @ np.linalg.inv(design.T @ design) @ contrast))
    assert_allclose(out.varm['log2fc'][:, 0], contrast @ beta, atol=1e-12)
    assert_allclose(out.varm['se_raw'][:, 0], se, rtol=1e-12)
    assert_allclose(out.varm['p_raw'][:, 0], 2*student_t.sf(np.abs(contrast @ beta/se), df), rtol=1e-12)
    assert np.isfinite(out.varm['p_ebayes']).all()
    if separate:
        model = out.uns['analysis']['contrast_models']['A_vs_B']
        assert model['n_samples'] == 6 and not model['pilot']
    else:
        assert out.uns['analysis']['design_columns'] == ['A', 'B', 'C']
        assert out.uns['analysis']['residual_df'] == 4


def test_single_condition_has_no_fabricated_contrasts():
    out = run_limma_pipeline(expression(("A",)*3), {})
    assert out.uns["pilot_study_mode"]
    assert list(out.uns["contrast_names"]) == []
    assert "log2fc" not in out.varm


def test_both_side_raw_missingness_is_neutralized_before_bh():
    obj = expression()
    obj.layers["raw"][:6, 0] = np.nan  # A/B missing; C observed
    obj.layers["raw"][:3, 1] = np.nan  # A only missing
    obj.layers["qvalue"] = np.full(obj.shape, .001)  # identification remains observed
    result = run_limma_pipeline(obj, {"analysis": {"only_contrasts": ["A_vs_B", "A_vs_C"]}})
    assert result.uns["n_fully_imputed_cells"] == 1
    for key in ("p_raw", "q_raw", "p_ebayes", "q_ebayes"):
        assert result.varm[key][0, 0] == 1
    for key in ("log2fc", "t_raw", "t_ebayes"):
        assert result.varm[key][0, 0] == 0
    assert np.isinf(result.varm["se_ebayes"][0, 0])
    assert result.varm["p_ebayes"][1, 0] < 1
    assert result.varm["p_ebayes"][0, 1] < 1
    # Independent BH rank calculation includes the neutral p=1 in the universe.
    p = result.varm["p_ebayes"][:, 0]
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    assert_allclose(result.varm["q_ebayes"][order, 0], np.minimum(q, 1))
    assert result.uns["missingness"].to_numpy().sum() == 0


def test_label_reordering_preserves_inference():
    obj = expression()
    out = run_limma_pipeline(obj, {})
    permutation = [8, 0, 4, 1, 6, 3, 2, 7, 5]
    shuffled = run_limma_pipeline(obj[permutation, ::-1].copy(), {})
    aligned = shuffled[:, out.var_names]
    for key in ("log2fc", "se_raw", "p_ebayes", "q_ebayes"):
        assert_allclose(out.varm[key], aligned.varm[key], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("separate", [False, True])
def test_contrast_reversal_and_feature_specific_log_offsets(separate):
    obj = expression()
    config = {"analysis": {"only_contrasts": ["B_vs_A", "A_vs_B"],
                           "separate_contrasts": separate}}
    out = run_limma_pipeline(obj, config)
    assert list(out.uns["contrast_names"]) == ["B_vs_A", "A_vs_B"]
    for key in ("log2fc", "t_raw", "t_ebayes", "nrsc"):
        assert_allclose(out.varm[key][:, 0], -out.varm[key][:, 1], atol=1e-12)
    for key in ("se_raw", "se_ebayes", "p_raw", "q_raw", "p_ebayes", "q_ebayes"):
        assert_allclose(out.varm[key][:, 0], out.varm[key][:, 1], rtol=1e-10)
    # Multiplying each protein by its own constant cannot change its contrasts.
    shifted = obj.copy()
    offsets = np.linspace(-3., 7., obj.n_vars)
    shifted.X += offsets
    shifted.layers["normalized"] += offsets
    shifted.layers["raw"] *= np.exp2(offsets)
    result = run_limma_pipeline(shifted, config)
    for key in out.varm:
        assert_allclose(result.varm[key], out.varm[key], rtol=1e-10, atol=1e-12)


def test_undefined_raw_test_must_not_erase_other_features_qvalues():
    obj = expression()
    obj.X[:, 0] = 0
    obj.layers["raw"][:, 0] = 1  # genuinely observed, not fully imputed
    out = run_limma_pipeline(obj, {"analysis": {"only_contrasts": ["B_vs_A"]}})
    from scipy.stats import false_discovery_control
    assert np.isnan(out.varm["p_raw"][0, 0])
    assert np.isnan(out.varm["q_raw"][0, 0])
    assert list(out.var_names) == list(obj.var_names)
    assert_allclose(out.varm["q_raw"][1:, 0],
                    false_discovery_control(out.varm["p_raw"][1:, 0]), rtol=1e-12)
    # Independent family: moderation still tests all features including the
    # observed constant row. Raw missingness must not be copied to this family.
    assert np.isfinite(out.varm["p_ebayes"]).all()
    assert_allclose(out.varm["q_ebayes"][:, 0],
                    false_discovery_control(out.varm["p_ebayes"][:, 0]), rtol=1e-12)


def test_bh_excludes_undefined_values_and_restores_feature_positions():
    p = np.array([[.04, np.nan], [np.nan, np.nan], [.01, np.nan], [.03, np.nan]])
    before = p.copy()
    # Three testable hypotheses: sorted .01, .03, .04 -> .03, .04, .04.
    assert_allclose(bh_qvalues(p), [[.04, np.nan], [np.nan, np.nan],
                                    [.03, np.nan], [.04, np.nan]])
    assert_array_equal(p, before)



@pytest.mark.parametrize("analysis, message", [
    ({"only_contrasts": ["A-B"]}, "separator"),
    ({"only_contrasts": ["A_vs_A"]}, "different conditions"),
    ({"only_contrasts": ["A_vs_B", "A_v_B"]}, "Duplicate contrast"),
    ({"only_contrasts": ["A_vs_missing"]}, "conditions must be"),
    ({"only_against": "missing"}, "not found"),
    ({"separate_contrasts": "false"}, "true or false"),
    ({"separate_contrasts": None}, "true or false"),
    ({"batch_effect_columns": ["missing"]}, "missing"),
    ({"batch_effect_columns": ["batch", "batch"]}, "duplicates"),
])
def test_invalid_design_configuration_fails_clearly(analysis, message):
    with pytest.raises(ValueError, match=message):
        run_limma_pipeline(expression(), {"analysis": analysis})


def test_confounded_batch_is_rejected():
    obj = expression()
    obj.obs["batch"] = obj.obs.CONDITION
    with pytest.raises(ValueError, match="confounded|rank"):
        run_limma_pipeline(obj, {"analysis": {"batch_effect_columns": ["batch"]}})


def test_missingness_priority_and_full_feature_universe():
    obj = expression()
    obj.layers["raw"][0, 0] = np.nan
    raw = compute_missingness(obj, max_features=2)
    assert raw.df.shape[0] == obj.n_vars
    assert raw.df.to_numpy().sum() == 1
    obj.layers["qvalue"] = np.zeros(obj.shape)
    obj.layers["qvalue"][4:6, 1] = np.nan
    identification = compute_missingness(obj, max_features=2)
    assert identification.df.to_numpy().sum() == 2
    assert identification.df.loc["protein0"].sum() == 0


def test_nrsc_uses_condition_totals_not_per_sample_means():
    obj = expression(("A",)*2+("B",)*3)
    nrsc = compute_nrsc(obj, ["A_vs_B"])
    assert_allclose(nrsc, -.2)
    obj.varm["log2fc"] = np.full((obj.n_vars, 1), -1.)
    obj.uns["contrast_names"] = ["A_vs_B"]
    assert_allclose(compute_nrsc_misalignment(obj, ["A_vs_B"], nrsc=nrsc), .8)


def test_optional_spectral_counts_can_be_absent(caplog):
    obj = expression()
    del obj.layers["spectral_counts"]
    before = obj.X.copy()
    assert compute_nrsc(obj, ["A_vs_B"]) is None
    assert "spectral_counts" not in obj.layers
    assert "nrsc" not in obj.varm
    assert_array_equal(obj.X, before)
    assert any(record.levelname == "WARNING" and "Skipping nrSC" in record.message
               and "spectral_counts" in record.message for record in caplog.records)


def test_zero_spectral_counts_remain_distinct_from_absent_evidence(caplog):
    obj = expression(("A",)*2+("B",)*2)
    obj.layers["spectral_counts"][:, 0] = 0
    before = obj.layers["spectral_counts"].copy()
    nrsc = compute_nrsc(obj, ["A_vs_B", "B_vs_A"])
    assert nrsc is not None
    assert nrsc.shape == (obj.n_vars, 2)
    assert nrsc.dtype == np.float32
    assert np.isnan(nrsc[0]).all()  # zero total: undefined ratio, not absent layer
    assert_array_equal(nrsc[1:], 0)  # observed equal positive counts: zero ratio
    assert_array_equal(obj.layers["spectral_counts"], before)
    assert not caplog.records


@pytest.mark.parametrize("pilot", [False, True])
def test_missing_spectral_counts_do_not_change_limma_or_pilot_results(pilot):
    obj = expression(("A",)*3+("B",)*3+("C",) if pilot else ("A",)*3+("B",)*3+("C",)*3)
    config = {"analysis": {"only_contrasts": ["A_vs_B", "A_vs_C"]}}
    baseline = run_limma_pipeline(obj, config)
    del obj.layers["spectral_counts"]
    result = run_limma_pipeline(obj, config)
    assert result.uns["pilot_study_mode"] == pilot
    assert "spectral_counts" not in result.layers
    assert set(result.varm) == set(baseline.varm) - {"nrsc", "nrsc_misalignment"}
    for key in result.varm:
        assert_array_equal(result.varm[key], baseline.varm[key])
    assert_array_equal(result.X, baseline.X)
    assert_array_equal(result.layers["raw"], baseline.layers["raw"])


@pytest.mark.parametrize('separate', [False, True])
@pytest.mark.parametrize('selection,contrasts,pilot', [
    ({}, ['A_vs_B', 'A_vs_C', 'B_vs_C'], True),
    ({'only_against': 'A'}, ['B_vs_A', 'C_vs_A'], True),
    ({'only_contrasts': ['A_vs_B', 'A_vs_C']}, ['A_vs_B', 'A_vs_C'], True),
    ({'only_contrasts': ['A_vs_C', 'A_vs_B']}, ['A_vs_C', 'A_vs_B'], True),
    ({'only_contrasts': ['A_v_B'], 'only_against': 'A'}, ['A_vs_B'], False),
])
def test_pilot_is_uniform_across_selected_analysis(separate, selection, contrasts, pilot, tmp_path):
    obj = expression(('A',)*3+('B',)*3+('C',))
    config = {'analysis': {**selection, 'separate_contrasts': separate}}
    result = run_limma_pipeline(obj, config)
    assert result.uns['pilot_study_mode'] == pilot
    assert list(result.uns['contrast_names']) == contrasts
    assert list(result.uns['analysis']['pilot_contrasts']) == (contrasts if pilot else [])
    for j, contrast in enumerate(contrasts):
        a, b = contrast.split('_vs_')
        expected_fc = obj.X[obj.obs.CONDITION == a].mean(0)-obj.X[obj.obs.CONDITION == b].mean(0)
        assert_allclose(result.varm['log2fc'][:, j], expected_fc, atol=1e-12)
    for key in ('se_raw', 't_raw', 'p_raw', 'q_raw', 'se_ebayes', 't_ebayes', 'p_ebayes', 'q_ebayes'):
        if pilot:
            assert key not in result.varm
        else:
            assert np.isfinite(result.varm[key]).all()
    if separate:
        assert all(model['pilot'] == pilot for model in result.uns['analysis']['contrast_models'].values())
    path = tmp_path/'selected-pilot.h5ad'
    result.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    assert loaded.uns['pilot_study_mode'] == pilot
    assert set(loaded.varm) == set(result.varm)
    for key in result.varm:
        assert_array_equal(loaded.varm[key], result.varm[key])
    assert_array_equal(loaded.obs_names, obj.obs_names)
    assert_array_equal(loaded.var_names, obj.var_names)


@pytest.mark.parametrize('value', [None, np.nan, ''])
def test_partial_batch_values_are_rejected(value):
    obj = expression()
    obj.obs['batch'] = ['x', 'y', value]*3
    with pytest.raises(ValueError, match='batch.*missing|batch.*null'):
        run_limma_pipeline(obj, {'analysis': {'batch_effect_columns': ['batch']}})


@pytest.mark.parametrize('label', ['CONDITION', 'Q', 'A_vs_B', 'A_v_B', 'True', 'Intercept'])
def test_reserved_condition_names_are_rejected(label):
    obj = expression()
    obj.obs.loc[obj.obs.CONDITION == 'A', 'CONDITION'] = label
    with pytest.raises(ValueError, match='condition|Condition'):
        run_limma_pipeline(obj, {})


def test_condition_cannot_collide_with_generated_batch_term():
    obj = expression()
    obj.obs.loc[obj.obs.CONDITION == 'A', 'CONDITION'] = 'batch__y'
    obj.obs['batch'] = ['x', 'y', 'x']*3
    with pytest.raises(ValueError, match='collides with a batch'):
        run_limma_pipeline(obj, {'analysis': {'batch_effect_columns': ['batch']}})
