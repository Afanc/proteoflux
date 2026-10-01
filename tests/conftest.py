"""Deterministic test execution; optional data are never required by default."""
import os
import tempfile

# Set before scientific imports. CI sets these too, including before pytest plugins.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
_cache = tempfile.TemporaryDirectory(prefix="proteoflux-test-cache-")
os.environ.setdefault("NUMBA_CACHE_DIR", _cache.name + "/numba")
os.environ.setdefault("MPLCONFIGDIR", _cache.name + "/matplotlib")

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run expensive validation")
    parser.addoption("--full-regression", metavar="MANIFEST", help="Local regression manifest JSON")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="enable with --run-slow"))
        if "full" in item.keywords and not config.getoption("--full-regression"):
            item.add_marker(pytest.mark.skip(reason="requires --full-regression MANIFEST"))


@pytest.fixture
def protein_config(tmp_path):
    from tests.fixtures.workflows import protein_workflow
    return protein_workflow(tmp_path)


@pytest.fixture(scope="session")
def workflow_results(tmp_path_factory):
    """Run each requested real entrypoint once; callers must copy before mutation."""
    from proteoflux.main import run_pipeline
    import anndata as ad
    from tests.fixtures.workflows import make_workflow

    cache = {}

    def run(name):
        if name not in cache:
            directory = tmp_path_factory.mktemp(name)
            config = make_workflow(name, directory)
            run_pipeline(config)
            cache[name] = (ad.read_h5ad(config["exports"]["path_h5ad"]), config)
        return cache[name]

    return run
