"""Regression tests for previously demonstrated defects."""
import numpy as np
import pytest

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.workflow.preprocessing import Preprocessor
from tests.unit.test_statistics import expression

pytestmark = pytest.mark.unit


def test_equal_residual_variances_still_allow_moderation():
    obj = expression()
    obj.X = np.tile(np.array([-1., 0., 1., 0., 1., 2., -2., -1., 0.])[:, None], (1, obj.n_vars))
    result = run_limma_pipeline(obj, {})
    assert np.isfinite(result.varm["p_ebayes"]).all()
