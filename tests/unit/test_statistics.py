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


def test_unselected_singleton_forces_experiment_wide_pilot():
    obj = expression(("A",)*3+("B",)*3+("C",))
    out = run_limma_pipeline(obj, {"analysis": {"only_contrasts": ["A_vs_B"]}})
    assert out.uns["pilot_study_mode"]
    assert not out.uns["has_covariate"]
    assert "p_ebayes" not in out.varm
    assert_allclose(out.varm["log2fc"][:, 0], obj.X[:3].mean(0)-obj.X[3:6].mean(0))


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


@pytest.mark.parametrize("analysis, message", [
    ({"only_contrasts": ["A-B"]}, "separator"),
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
    config = {"analysis": {"only_contrasts": ["A_vs_B"]}}
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
