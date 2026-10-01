"""Desired behavior blocked by specific defects; strict XPASS demands review.

These are not golden expectations approving the current defect. Keep each
expected failure narrow; unrelated exception types still fail the suite.
"""
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.analysis.nrsc import compute_nrsc
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.workflow.rollup.robust_regression import pivot_df_robust_regression
from tests.unit.test_statistics import expression

pytestmark = [pytest.mark.unit, pytest.mark.known_bug]


@pytest.mark.xfail(raises=KeyError, reason="inmoose 0.9.0 eBayes indexes t2 with ~True == -2 for infinite prior df")
def test_equal_residual_variances_still_allow_moderation():
    obj = expression()
    obj.X = np.tile(np.array([-1., 0., 1., 0., 1., 2., -2., -1., 0.])[:, None], (1, obj.n_vars))
    result = run_limma_pipeline(obj, {})
    assert np.isfinite(result.varm["p_ebayes"]).all()


@pytest.mark.xfail(raises=NameError, reason="nrsc missing-layer warning uses unimported log_warning")
def test_optional_spectral_counts_can_be_absent():
    obj = expression()
    del obj.layers["spectral_counts"]
    assert compute_nrsc(obj, ["A_vs_B"]) is None


@pytest.mark.xfail(raises=UnboundLocalError, reason="disconnected robust-regression IRLS reads beta before initialization")
def test_disconnected_robust_rollup_produces_result():
    peptides = pl.DataFrame({"protein": ["P", "P"], "peptide": ["first", "second"],
                             "a": [8., None], "b": [None, 32.]})
    out = pivot_df_robust_regression(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                                    sample_cols=["a", "b"], num_cores=1, huber_t=1.345,
                                    max_iter=20, tol=1e-8, min_nonan=1)
    assert out.shape == (1, 3)


@pytest.mark.xfail(raises=AssertionError, reason="scalar log2 setting is misclassified as no-log when exporting raw")
def test_scalar_and_list_log2_preserve_same_raw_scale(protein_config):
    protein_config["preprocessing"]["normalization"]["method"] = ["log2"]
    expected = Dataset(**protein_config).get_anndata()
    protein_config["preprocessing"]["normalization"]["method"] = "log2"
    actual = Dataset(**protein_config).get_anndata()
    assert_allclose(actual.layers["raw"], expected.layers["raw"], equal_nan=True)


@pytest.mark.parametrize("mode", ["free", "nonneg"])
@pytest.mark.xfail(raises=UnboundLocalError, reason="non-fixed covariate modes use contrast_noft before assignment")
def test_optional_covariate_slope_modes_can_run(tmp_path, mode):
    from tests.fixtures.workflows import phospho_workflow
    config = phospho_workflow(tmp_path, kinase=False)
    config["ancova"] = {"beta_mode": mode}
    result = run_limma_pipeline(Dataset(**config).get_anndata(), config)
    assert result.varm["log2fc"].shape[1] == 2


@pytest.mark.xfail(raises=AssertionError, reason="covariate_p is overwritten with FT contrast p-values and gains a third axis")
def test_covariate_p_has_one_diagnostic_per_feature(tmp_path):
    from tests.fixtures.workflows import phospho_workflow
    config = phospho_workflow(tmp_path, kinase=False)
    out = run_limma_pipeline(Dataset(**config).get_anndata(), config)
    assert out.varm["covariate_p"].shape == (out.n_vars, 1)


@pytest.mark.xfail(raises=NameError, reason="PyArrow CSV loader refers to unimported pv_csv")
def test_pyarrow_delimited_loader_can_read_supported_input(protein_config):
    protein_config["dataset"]["load_method"] = "pyarrow"
    assert Dataset(**protein_config).get_anndata().n_vars == 12
