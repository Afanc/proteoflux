"""Opt-in full experiments; paths and reference provenance are user supplied."""
import json
from pathlib import Path

import anndata as ad
import pytest
import yaml

from proteoflux.main import run_pipeline
from tests.full_regression import compare_full_results

pytestmark = pytest.mark.full


def pytest_generate_tests(metafunc):
    if "local_case" not in metafunc.fixturenames:
        return
    path = metafunc.config.getoption("--full-regression")
    if path is None:
        metafunc.parametrize("local_case", [None], ids=["disabled"])
        return
    manifest = Path(path).resolve()
    cases = json.loads(manifest.read_text())
    if not isinstance(cases, list) or not cases:
        raise pytest.UsageError("Full-regression manifest must be a nonempty JSON list")
    for case in cases:
        if not {"name", "config", "workdir", "reference", "provenance"} <= set(case):
            raise pytest.UsageError("Each local case requires name, config, workdir, reference, provenance")
        for key in ("config", "workdir", "reference"):
            case[key] = (manifest.parent / case[key]).resolve()
            if not case[key].exists():
                raise pytest.UsageError(f"Explicit local case {case['name']}: missing {key}: {case[key]}")
    metafunc.parametrize("local_case", cases, ids=[case["name"] for case in cases])


def test_local_scientific_regression(local_case, tmp_path, monkeypatch):
    config = yaml.safe_load(local_case["config"].read_text())
    # Production resolves relative input paths from cwd. Never write through the
    # original config's output locations, even when legacy exports are present.
    config["exports"] = {"path_h5ad": str(tmp_path/"actual.h5ad"),
                         "path_table": None, "path_pdf_report": None}
    monkeypatch.chdir(local_case["workdir"])
    run_pipeline(config)
    compare_full_results(ad.read_h5ad(config["exports"]["path_h5ad"]),
                         ad.read_h5ad(local_case["reference"]),
                         rtol=float(local_case.get("rtol", 3e-6)),
                         atol=float(local_case.get("atol", 2e-6)))
