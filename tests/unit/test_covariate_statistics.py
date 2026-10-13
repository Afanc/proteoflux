"""Stage-1 association diagnostics are distinct from FT contrast tests."""
from copy import deepcopy

import anndata as ad
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import false_discovery_control, linregress, t as student_t

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.export.de_exporter import DEExporter
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
def test_flat_covariate_diagnostic_is_missing_including_imputed_background(covariate_experiment, fully_imputed):
    obj, config = covariate_experiment
    obj.layers["centered_covariate"][:, 0] *= .001
    obj.layers["raw_covariate"][:, 0] = (
        np.nan if fully_imputed else np.exp2(obj.layers["centered_covariate"][:, 0]))
    out = run_limma_pipeline(obj, config)
    assert np.isnan(out.varm["covariate_t"][0, 0])
    assert out.varm["covariate_p"].shape == (obj.n_vars, 1)
    assert np.isnan(out.varm["covariate_p"][0, 0])
    assert np.isnan(out.varm["covariate_q"][0, 0])
    if fully_imputed:
        assert_array_equal(out.varm["ft_p_ebayes"][0], [1., 1.])


def test_slope_diagnostics_follow_feature_labels(covariate_experiment):
    obj, config = covariate_experiment
    baseline = run_limma_pipeline(obj, config)
    shuffled = run_limma_pipeline(obj[[8, 0, 4, 1, 6, 3, 2, 7, 5], ::-1].copy(), config)
    aligned = shuffled[:, baseline.var_names]
    for key in baseline.varm:
        assert_allclose(aligned.varm[key], baseline.varm[key], rtol=1e-10, atol=1e-14)
    # Residual sample axes must also follow labels, not merely keep their shape.
    aligned = aligned[baseline.obs_names, :]
    assert_allclose(aligned.layers["residuals_covariate"], baseline.layers["residuals_covariate"],
                    rtol=1e-10, atol=1e-14)


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


@pytest.mark.contract
def test_unestimable_background_diagnostics_apply_bh_to_testable_sites(covariate_experiment, tmp_path):
    obj, config = covariate_experiment
    baseline = run_limma_pipeline(obj, config)
    feature = np.argmin(baseline.varm["covariate_p"][:, 0])
    # A non-flat imputed analysis vector is not evidence of observed background.
    obj.layers["raw_covariate"][:, feature] = np.nan
    result = run_limma_pipeline(obj, config)
    p = result.varm["covariate_p"][:, 0]
    assert np.isnan(p[feature])
    expected_p = baseline.varm["covariate_p"][:, 0].copy()
    expected_p[feature] = np.nan
    assert_array_equal(p, expected_p)
    expected_q = np.full_like(p, np.nan)
    expected_q[np.isfinite(p)] = false_discovery_control(p[np.isfinite(p)])
    assert_allclose(result.varm["covariate_q"][:, 0], expected_q, rtol=1e-12, atol=1e-14)
    # Correct diagnostics must survive the actual H5AD exporter with their labels.
    path = tmp_path/"slope-diagnostics.h5ad"
    DEExporter(result, output_path=None, config=config).export_adata(str(path))
    restored = ad.read_h5ad(path)
    assert_array_equal(restored.var_names, obj.var_names)
    assert restored.varm["covariate_q"].shape == (obj.n_vars, 1)
    assert_array_equal(restored.varm["covariate_p"][:, 0], p)
    assert_allclose(restored.varm["covariate_q"][:, 0], expected_q, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("missing", ["partial", "multiple", "all"])
def test_slope_diagnostic_bh_excludes_unestimable_sites(covariate_experiment, missing):
    obj, config = covariate_experiment
    config["ancova"] = {"ridge_lambda": 0}
    expected_p = np.array([linregress(obj.layers["centered_covariate"][:, g], obj.X[:, g]).pvalue
                           for g in range(obj.n_vars)])
    if missing == "partial":
        obj.layers["raw_covariate"][0, [0, -1]] = np.nan
    elif missing == "multiple":
        obj.layers["raw_covariate"][:, [0, -1]] = np.nan
    else:
        obj.layers["raw_covariate"][:] = np.nan
    expected_p[np.isnan(obj.layers["raw_covariate"]).all(axis=0)] = np.nan
    expected_q = np.full_like(expected_p, np.nan)
    finite = np.isfinite(expected_p)
    if finite.any():
        expected_q[finite] = false_discovery_control(expected_p[finite])
    result = run_limma_pipeline(obj, config)
    assert_allclose(result.varm["covariate_p"][:, 0], expected_p, rtol=1e-10, atol=1e-14)
    assert_allclose(result.varm["covariate_q"][:, 0], expected_q, rtol=1e-10, atol=1e-14)


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


@pytest.mark.parametrize("separate", [False, True])
def test_fixed1_matches_independent_residual_regression(covariate_experiment, separate):
    obj, config = covariate_experiment
    config["ancova"] = {"ridge_lambda": 0}
    config["analysis"]["separate_contrasts"] = separate
    cov = obj.layers["centered_covariate"]
    fits = [linregress(cov[:, g], obj.X[:, g]) for g in range(obj.n_vars)]
    slopes = np.array([fit.slope for fit in fits])
    assert (slopes < 0).any() and (slopes > 0).any()
    applied = np.ones_like(slopes)
    residuals = obj.X - np.array([fit.intercept for fit in fits]) - cov*applied
    out = run_limma_pipeline(obj, config)
    assert_allclose(out.varm["covariate_beta"][:, 0], applied, rtol=1e-10, atol=1e-12)
    assert_allclose(out.layers["residuals_covariate"], residuals, rtol=1e-10, atol=1e-12)
    # Diagnostics describe the unconstrained slope, not the applied unit slope.
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


def test_selective_fallback_uses_coherent_model_and_one_final_bh_family(covariate_experiment):
    obj, config = covariate_experiment
    obj.layers['raw_covariate'][3:6, [0, 5, 11]] = np.nan
    result = run_limma_pipeline(obj, config)
    raw = run_limma_pipeline(obj, {'analysis': config['analysis']})
    adjusted_obj = obj.copy()
    cov = obj.layers['centered_covariate']
    adjusted_obj.X = obj.X - obj.X.mean(0) - (cov-cov.mean(0))
    adjusted = run_limma_pipeline(adjusted_obj, {'analysis': config['analysis']})
    fallback = np.zeros((obj.n_vars, 2), dtype=bool)
    fallback[[0, 5, 11], 0] = True
    for key in ('log2fc', 'se_raw', 't_raw', 'p_raw', 'se_ebayes', 't_ebayes', 'p_ebayes'):
        expected = np.where(fallback, raw.varm[key], adjusted.varm[key])
        assert_allclose(result.varm[key], expected, rtol=1e-10, atol=1e-14)
    for family in ('raw', 'ebayes'):
        for j in range(2):
            assert_allclose(result.varm['q_'+family][:, j],
                            false_discovery_control(result.varm['p_'+family][:, j]), rtol=1e-12)
    assert_array_equal(result.varm['covariate_adjusted'], ~fallback)


@pytest.mark.parametrize('mode', ['free', 'nonneg', 'nonnegative'])
def test_unsupported_slopes_rejected_even_without_covariate(mode):
    with pytest.raises(ValueError, match='fixed1'):
        run_limma_pipeline(expression(), {'ancova': {'beta_mode': mode}})


def test_cov_part_remains_a_mean_difference_not_a_batch_decomposition(covariate_experiment):
    obj, config = covariate_experiment
    obj.obs['batch'] = ['x', 'x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']
    config['analysis']['batch_effect_columns'] = ['batch']
    obj.layers['centered_covariate'] += 3 * (obj.obs.batch == 'y').to_numpy()[:, None]
    result = run_limma_pipeline(obj, config)
    cov = obj.layers['centered_covariate']
    assert_allclose(result.varm['cov_part'][:, 0], cov[3:6].mean(0)-cov[:3].mean(0), atol=1e-12)
    # Linearity of least squares with one common design gives the actual
    # batch-adjusted difference. cov_part intentionally reports another quantity.
    assert_allclose(result.varm['raw_log2fc']-result.varm['log2fc'], result.varm['ft_log2fc'], atol=1e-12)
    assert not np.allclose(result.varm['cov_part'], result.varm['ft_log2fc'])
    assert 'not a batch-adjusted decomposition' in result.uns['decomposition_rule']
