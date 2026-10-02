"""Stage-1 association diagnostics are distinct from FT contrast tests."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import false_discovery_control, linregress, t as student_t

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


def test_missing_spectral_counts_do_not_change_covariate_results(covariate_experiment):
    obj, config = covariate_experiment
    baseline = run_limma_pipeline(obj, config)
    del obj.layers["spectral_counts"]
    result = run_limma_pipeline(obj, config)
    assert result.uns["has_covariate"]
    assert "spectral_counts" not in result.layers
    assert set(result.varm) == set(baseline.varm) - {"nrsc", "nrsc_misalignment"}
    for key in result.varm:
        assert_array_equal(result.varm[key], baseline.varm[key])
    assert_array_equal(result.layers["residuals_covariate"], baseline.layers["residuals_covariate"])


@pytest.mark.parametrize("mode", ["free", "nonneg"])
@pytest.mark.parametrize("separate", [False, True])
def test_fitted_modes_match_independent_two_stage_regression(covariate_experiment, mode, separate):
    obj, config = covariate_experiment
    config["ancova"] = {"beta_mode": mode, "ridge_lambda": 0}
    config["analysis"]["separate_contrasts"] = separate
    cov = obj.layers["centered_covariate"]
    fits = [linregress(cov[:, g], obj.X[:, g]) for g in range(obj.n_vars)]
    slopes = np.array([fit.slope for fit in fits])
    assert (slopes < 0).any() and (slopes > 0).any()
    applied = np.maximum(slopes, 0) if mode == "nonneg" else slopes
    residuals = obj.X - np.array([fit.intercept for fit in fits]) - cov*applied
    out = run_limma_pipeline(obj, config)
    assert_allclose(out.varm["covariate_beta"][:, 0], applied, rtol=1e-10, atol=1e-12)
    assert_allclose(out.layers["residuals_covariate"], residuals, rtol=1e-10, atol=1e-12)
    # Diagnostics describe the unconstrained slope, including negative slopes
    # clipped for residualization in nonneg mode.
    assert_allclose(out.varm["covariate_p"][:, 0], [fit.pvalue for fit in fits], rtol=1e-10, atol=1e-14)
    for j, contrast in enumerate(out.uns["contrast_names"]):
        a, b = contrast.split("_vs_")
        levels = [a, b] if separate else ["A", "B", "C"]
        selected = obj.obs.CONDITION.isin(levels)
        conditions = obj.obs.CONDITION[selected]
        design = np.column_stack([conditions == level for level in levels]).astype(float)
        beta, *_ = np.linalg.lstsq(design, residuals[selected], rcond=None)
        contrast_vector = np.array([float(level == a)-float(level == b) for level in levels])
        fc = contrast_vector @ beta
        error = residuals[selected] - design @ beta
        df = selected.sum() - np.linalg.matrix_rank(design)
        variance = (error**2).sum(axis=0)/df
        se = np.sqrt(variance * (contrast_vector @ np.linalg.inv(design.T @ design) @ contrast_vector))
        assert_allclose(out.varm["log2fc"][:, j], fc, rtol=1e-10, atol=1e-12)
        assert_allclose(out.varm["se_raw"][:, j], se, rtol=1e-10)
        p = 2*student_t.sf(abs(fc/se), df)
        assert_allclose(out.varm["p_raw"][:, j], p, rtol=1e-10, atol=1e-14)
        assert_allclose(out.varm["q_raw"][:, j], false_discovery_control(p), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("mode,separate", [
    ("free", False), ("nonneg", False), ("free", True),
    pytest.param("nonneg", True, marks=[pytest.mark.known_bug,
        pytest.mark.xfail(raises=KeyError, reason="Deferred Inmoose infinite-prior-df KeyError -2 in pairwise nonneg fit")]),
])
def test_fitted_modes_keep_global_slopes_and_contrast_local_background_support(covariate_experiment, mode, separate):
    obj, config = covariate_experiment
    config["ancova"] = {"beta_mode": mode}
    config["analysis"] = {"only_contrasts": ["B_vs_A", "C_vs_B"], "separate_contrasts": separate}
    baseline = run_limma_pipeline(obj, config)
    # Last four features have positive slopes, including under nonneg.
    assert (baseline.varm["covariate_beta"][-4:, 0] > 0).all()
    obj.layers["raw_covariate"][:3, -1] = np.nan  # A absent; B/C still supported
    obj.layers["raw_covariate"][:, -2] = np.nan   # no observed FT in the experiment
    obj.layers["raw_covariate"][0, -3] = np.nan   # partial A; both contrasts supported
    obj.layers["raw_covariate"][:6, -4] = np.nan  # only C observed
    before_raw = obj.layers["raw_covariate"].copy()
    out = run_limma_pipeline(obj, config)
    raw = run_limma_pipeline(obj, {"analysis": config["analysis"]})
    # Only an entirely imputed background suppresses the global slope.
    keep = np.arange(obj.n_vars) != obj.n_vars-2
    assert_array_equal(out.varm["covariate_beta"][keep], baseline.varm["covariate_beta"][keep])
    assert out.varm["covariate_beta"][-2, 0] == 0
    assert out.varm["covariate_t"][-2, 0] == 0
    assert out.varm["covariate_p"][-2, 0] == 1
    assert out.varm["covariate_q"][-2, 0] == 1
    assert_allclose(out.layers["residuals_covariate"][:, -2], obj.X[:, -2]-obj.X[:, -2].mean(), atol=1e-12)
    unsupported = np.zeros((obj.n_vars, 2), dtype=bool)
    unsupported[-1, 0] = True
    unsupported[[-2, -4], :] = True
    for key in ("log2fc", "p_ebayes", "q_ebayes"):
        assert_array_equal(out.varm[key][unsupported], raw.varm[key][unsupported])
    assert_array_equal(out.varm["cov_part"][unsupported], 0)
    assert_allclose(out.varm["log2fc"][~unsupported], baseline.varm["log2fc"][~unsupported], rtol=1e-10, atol=1e-12)
    assert_allclose(out.varm["raw_log2fc"], out.varm["log2fc"]+out.varm["cov_part"], rtol=1e-10, atol=1e-12)
    assert_array_equal(out.layers["raw_covariate"], before_raw)
    assert_array_equal(obj.layers["raw_covariate"], before_raw)
