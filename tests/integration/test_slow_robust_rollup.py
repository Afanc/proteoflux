"""Opt-in worker process check, matching the PhosX worker-test convention."""
import polars as pl
import pytest
from numpy.testing import assert_array_equal

from proteoflux.workflow.rollup.robust_regression import (
    DisconnectedObservationError,
    DisconnectedObservationWarning,
    pivot_df_robust_regression,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_disconnected_rollup_worker_results_and_errors_match_serial():
    peptides = pl.DataFrame({"protein": ["P", "P", "Q", "Q"],
                             "peptide": ["first", "second", "first", "second"],
                             "a": [8., None, 16., None], "b": [None, 32., None, 64.]})
    options = dict(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                   sample_cols=["a", "b"], huber_t=1.345, max_iter=20, tol=1e-8, min_nonan=1)
    with pytest.warns(DisconnectedObservationWarning):
        serial = pivot_df_robust_regression(num_cores=1, disconnected_policy="component", **options)
    # Warnings originate in the workers; the caller still receives ordered results.
    parallel = pivot_df_robust_regression(num_cores=2, disconnected_policy="component", **options)
    assert parallel.columns == serial.columns
    assert parallel["protein"].to_list() == ["P", "Q"]
    assert_array_equal(parallel.select("a", "b").to_numpy(), serial.select("a", "b").to_numpy())
    for workers in (1, 2):
        with pytest.raises(DisconnectedObservationError, match="disconnected observation components"):
            pivot_df_robust_regression(num_cores=workers, **options)
