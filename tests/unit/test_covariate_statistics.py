"""Stage-1 association diagnostics are distinct from FT contrast tests."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import false_discovery_control, linregress

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.workflow.dataset import Dataset
from tests.fixtures.workflows import phospho_workflow
from tests.unit.test_statistics import expression

pytestmark = pytest.mark.unit
DIAGNOSTICS = ["covariate_se", "covariate_t", "covariate_p", "covariate_q"]


@pytest.fixture
def covariate_experiment():
    obj = expression()
    rng = np.random.default_rng(813)
    cov = rng.normal(size=obj.shape) * np.linspace(.3, 2., obj.n_vars)
    cov -= cov.mean(axis=0)
    obj.X += cov * np.linspace(-2., 3., obj.n_vars)
    obj.layers["raw"] = np.exp2(obj.X)
    obj.layers["centered_covariate"] = cov
    obj.layers["raw_covariate"] = np.exp2(cov)
    # Analysis only inspects injection metadata here; no input file is loaded.
    config = {"dataset": {"inject_runs": {"flowthrough": {
        "is_covariate": True, "reuse_main_input_file": True}}},
        "analysis": {"only_contrasts": ["B_vs_A", "C_vs_A"]}}
    return obj, config


@pytest.mark.parametrize("contrasts", [["B_vs_A"], ["C_vs_A", "B_vs_A"]])
def test_slope_probabilities_agree_with_independent_linear_regression(covariate_experiment, contrasts):
    obj, config = covariate_experiment
    config["analysis"]["only_contrasts"] = contrasts
    config["ancova"] = {"ridge_lambda": 0}
    # With ridge disabled and non-flat C, the diagnostic is ordinary simple
    # regression. SciPy fits it independently of ProteoFlux's closed form.
    fits = [linregress(obj.layers["centered_covariate"][:, g], obj.X[:, g])
            for g in range(obj.n_vars)]
    p = np.array([fit.pvalue for fit in fits])
    out = run_limma_pipeline(obj, config)
    for key in DIAGNOSTICS:
        assert out.varm[key].shape == (obj.n_vars, 1)
    assert_allclose(out.varm["covariate_se"][:, 0], [fit.stderr for fit in fits], rtol=1e-10)
    assert_allclose(out.varm["covariate_t"][:, 0], [fit.slope/fit.stderr for fit in fits], rtol=1e-10)
    assert_allclose(out.varm["covariate_p"][:, 0], p, rtol=1e-10, atol=1e-14)
    assert_allclose(out.varm["covariate_q"][:, 0], false_discovery_control(p), rtol=1e-10, atol=1e-14)
    # fixed1 controls residualization; it does not replace the association test.
    assert_array_equal(out.varm["covariate_beta"], np.ones((obj.n_vars, 1)))
    assert list(out.uns["contrast_names"]) == contrasts
    for key in ("ft_p_ebayes", "ft_q_ebayes"):
        assert out.varm[key].shape == (obj.n_vars, len(contrasts))
    assert not np.allclose(out.varm["covariate_p"][:, 0], out.varm["ft_p_ebayes"][:, 0])


@pytest.mark.parametrize("separate", [False, True])
def test_slope_diagnostic_uses_all_samples_regardless_of_contrast_scope(covariate_experiment, separate):
    obj, config = covariate_experiment
    baseline = run_limma_pipeline(obj, config)
    selected = deepcopy(config)
    selected["analysis"] = {"only_contrasts": ["B_vs_A"], "separate_contrasts": separate}
    out = run_limma_pipeline(obj, selected)
    for key in DIAGNOSTICS:
        assert_array_equal(out.varm[key], baseline.varm[key])
    changed = obj.copy()
    changed.X[6:] += 4 * changed.layers["centered_covariate"][6:]
    altered = run_limma_pipeline(changed, selected)
    assert not np.allclose(altered.varm["covariate_p"], out.varm["covariate_p"])


@pytest.mark.parametrize("fully_imputed", [False, True])
def test_flat_covariate_diagnostic_is_neutral_including_imputed_background(covariate_experiment, fully_imputed):
    obj, config = covariate_experiment
    obj.layers["centered_covariate"][:, 0] *= .001
    obj.layers["raw_covariate"][:, 0] = (
        np.nan if fully_imputed else np.exp2(obj.layers["centered_covariate"][:, 0]))
    out = run_limma_pipeline(obj, config)
    assert out.varm["covariate_t"][0, 0] == 0
    assert out.varm["covariate_p"].shape == (obj.n_vars, 1)
    assert out.varm["covariate_p"][0, 0] == 1
    assert out.varm["covariate_q"][0, 0] == 1
    if fully_imputed:
        assert_array_equal(out.varm["ft_p_ebayes"][0], [1., 1.])


def test_slope_diagnostics_follow_feature_labels(covariate_experiment):
    obj, config = covariate_experiment
    baseline = run_limma_pipeline(obj, config)
    shuffled = run_limma_pipeline(obj[[8, 0, 4, 1, 6, 3, 2, 7, 5], ::-1].copy(), config)
    aligned = shuffled[:, baseline.var_names]
    for key in DIAGNOSTICS:
        assert_allclose(aligned.varm[key], baseline.varm[key], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("indices", [[0, 1, 2], [0, 1, 2, 3, 4, 5, 6]])
def test_pilot_does_not_fabricate_covariate_probabilities(covariate_experiment, indices):
    obj, config = covariate_experiment
    out = run_limma_pipeline(obj[indices].copy(), config)
    assert out.uns["pilot_study_mode"]
    assert not out.uns["has_covariate"]
    assert not set(DIAGNOSTICS + ["covariate_beta", "ft_p_ebayes", "ft_q_ebayes"]) & set(out.varm)


def test_covariate_p_has_one_diagnostic_per_feature(tmp_path):
    config = phospho_workflow(tmp_path, kinase=False)
    out = run_limma_pipeline(Dataset(**config).get_anndata(), config)
    assert out.varm["covariate_p"].shape == (out.n_vars, 1)
    assert out.varm["covariate_q"].shape == (out.n_vars, 1)
