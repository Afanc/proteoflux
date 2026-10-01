import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from proteoflux.analysis.phosx_adapter import _fast_empirical_distribution

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_phosx_permutation_stream_is_independent_of_worker_count():
    rng = np.random.default_rng(15)
    assignments = pd.DataFrame(rng.random((40, 8)) < .2)
    scores = pd.Series(np.sort(rng.normal(size=40))[::-1])
    options = dict(n_permutations=2047, random_seed=71, family_label="worker determinism")
    serial = _fast_empirical_distribution(assignments, scores, n_processes=1, **options)
    parallel = _fast_empirical_distribution(assignments, scores, n_processes=2, **options)
    assert_array_equal(serial, parallel)
