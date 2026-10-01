"""Generate candidate references for review, never update files during pytest.

Usage: python -m tests.generate_references --output-dir /tmp/candidate-references
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import tempfile

# Apply the same execution settings before importing scientific libraries.
from tests import conftest  # noqa: F401
import anndata as ad
from proteoflux.main import run_pipeline
from tests.fixtures.workflows import WORKFLOWS, make_workflow
from tests.scientific_reference import REFERENCE_DIR, scientific_payload, reference_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workflow", choices=WORKFLOWS, action="append")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination == REFERENCE_DIR.resolve():
        parser.error("Generate candidates outside tests/references, review the diff, then copy explicitly.")
    destination.mkdir(parents=True, exist_ok=True)
    dependencies = {name: version(name) for name in ("numpy", "scipy", "pandas", "polars", "anndata",
                                                    "inmoose", "directlfq", "torch", "phosx", "scikit-learn")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in args.workflow or WORKFLOWS:
        with tempfile.TemporaryDirectory(prefix="proteoflux-reference-") as temporary:
            config = make_workflow(name, Path(temporary))
            run_pipeline(config)
            obj = ad.read_h5ad(config["exports"]["path_h5ad"])
            payload = {"kind": "current-behavior snapshot; not external scientific validation",
                       "source_revision": revision, "dependencies": dependencies,
                       "fixture": name, "result": scientific_payload(obj)}
            (destination/f"{name}.json").write_text(reference_json(payload)+"\n")


if __name__ == "__main__":
    main()
