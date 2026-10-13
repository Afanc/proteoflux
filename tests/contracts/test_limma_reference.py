"""Canonical R limma outputs; pytest never executes R or regenerates answers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.analysis.limma_pipeline import _build_limma_fit_plan, _fit_limma_from_plan

pytestmark = [pytest.mark.contract, pytest.mark.regression]
ROOT = Path(__file__).parents[1] / 'reference' / 'limma'


@pytest.mark.parametrize('case', [
    'balanced', 'shared', 'batch',
    pytest.param('equal_variance', marks=[pytest.mark.known_bug, pytest.mark.xfail(
        raises=KeyError, reason='InMoose 0.9.0 infinite-prior-df KeyError -2; R limma 3.65.1 yields finite moderated statistics')]),
    pytest.param('missing', marks=[pytest.mark.known_bug, pytest.mark.xfail(
        raises=np.linalg.LinAlgError, reason='InMoose lmFit singular feature-specific design; R limma retains estimable contrasts with NA for the unsupported contrast')]),
])
def test_limma_adapter_against_canonical_r(case):
    folder = ROOT / case
    expression = pd.read_csv(folder/'expression.csv', index_col=0)
    obs = pd.read_csv(folder/'samples.csv', index_col=0)
    design = pd.read_csv(folder/'design.csv', index_col=0)
    contrasts = pd.read_csv(folder/'contrasts.csv', index_col=0)
    expected = pd.read_csv(folder/'expected.csv').set_index(['feature', 'contrast'])
    config = {'analysis': {'only_contrasts': contrasts.columns.tolist(),
                           'batch_effect_columns': ['batch'] if case == 'batch' else []}}
    plan = _build_limma_fit_plan(obs, sorted(obs.CONDITION.unique()), config)
    entry = plan['entries'][0]
    assert_array_equal(expression.columns, obs.index)
    assert_array_equal(entry['sample_names'], design.index)
    assert_array_equal(plan['contrast_names'], contrasts.columns)
    assert_array_equal(np.asarray(entry['design']), design.to_numpy())
    assert_array_equal(entry['contrasts'].to_numpy(), contrasts.to_numpy())
    result = _fit_limma_from_plan(df_GxN=expression, fit_plan=plan)
    for key in ['coefs', 'df_res', 'se_raw', 't_raw', 'p_raw', 'q_raw',
                'se_ebayes', 't_ebayes', 'p_ebayes', 'q_ebayes']:
        field = {'coefs': 'log2fc', 'df_res': 'df_residual'}.get(key, key)
        reference = expected[field].unstack('contrast').loc[expression.index, contrasts.columns]
        actual = result[key]
        if key == 'df_res':
            actual = np.repeat(np.asarray(actual)[:, None], len(contrasts.columns), axis=1)
        # Float64 fits, independent R/Python LAPACK and special-function paths.
        # Exact missingness is protected by assert_allclose(equal_nan=True).
        assert_allclose(actual, reference.to_numpy(), rtol=1e-7, atol=1e-12, err_msg=f'{case}: {key}')


def test_reference_inputs_match_generation_record():
    import hashlib
    checksums = pd.read_csv(ROOT/'input_checksums.tsv', sep='\t')
    for record in checksums.itertuples(index=False):
        assert hashlib.md5((ROOT/record.file).read_bytes()).hexdigest() == record.md5, record.file
    session = (ROOT/'sessionInfo.txt').read_text()
    assert 'limma: 3.65.1' in session and 'R version 4.3.3' in session
