from copy import deepcopy
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_allclose
from typer.testing import CliRunner

from proteoflux.cli import app
from proteoflux.utils.analysis_type import normalize_analysis_type
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Preprocessor
from tests.fixtures.workflows import base_config, protein_rows

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet", ".pq"])
@pytest.mark.parametrize("explicit_polars", [False, True], ids=["default", "explicit-polars"])
def test_supported_file_containers_preserve_labels_values_and_missingness(tmp_path, suffix, explicit_polars):
    config = base_config(tmp_path)
    frame = protein_rows()
    canonical = tmp_path/"canonical.tsv"
    frame.to_csv(canonical, sep="\t", index=False)
    config["dataset"]["input_file"] = str(canonical)
    expected = Dataset(**config).get_anndata()
    path = tmp_path/("input"+suffix)
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, sep="\t" if suffix == ".tsv" else ",", index=False)
    config["dataset"]["input_file"] = str(path)
    if explicit_polars:
        config["dataset"]["load_method"] = "polars"
    actual = Dataset(**config).get_anndata()[expected.obs_names, expected.var_names]
    for key in ("raw", "qvalue", "pep", "normalized", "spectral_counts"):
        assert_allclose(actual.layers[key], expected.layers[key], rtol=3e-7, equal_nan=True)
    assert_allclose(actual.X, expected.X, rtol=3e-7)


@pytest.mark.parametrize("method", ["pyarrow", "pandas", "unknown", None])
@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_only_polars_load_method_is_accepted_before_file_io(tmp_path, method, suffix):
    config = base_config(tmp_path)
    config["dataset"].update(input_file=str(tmp_path/("does-not-exist"+suffix)), load_method=method)
    # Invalid config must not silently select another backend, even for Parquet
    # (whose format-specific reader remains PyArrow).
    with pytest.raises(ValueError, match=r"dataset\.load_method.*polars"):
        Dataset(**config)


@pytest.mark.parametrize("template", ["spectronaut-proteomics", "diann-proteomics", "fragpipe-peptidomics"])
def test_vendor_column_mappings_on_same_authored_experiment(tmp_path, template):
    config = yaml.safe_load(resources.files("proteoflux.templates").joinpath(template+".yaml").read_text())
    base = base_config(tmp_path, config["dataset"]["analysis_type"])
    rows = protein_rows()
    # Wide FragPipe lacks these confidence fields; compare the same already
    # filtered experiment for all adapters, rather than invent vendor scores.
    rows = rows[(rows.qvalue <= .01) & (rows.pep <= .05)].copy()
    source = tmp_path/"canonical.tsv"
    rows.to_csv(source, sep="\t", index=False)
    base["dataset"]["input_file"] = str(source)
    expected = Dataset(**base).get_anndata()
    mapping = {base["dataset"][key]: target for key, target in config["dataset"].items()
               if key.endswith("_column") and target and base["dataset"].get(key)}
    # Repeated targets must be supplied from one canonical source.
    converted = rows[list(mapping)].rename(columns=mapping)
    if config["dataset"]["input_layout"] == "wide":
        columns = [config["dataset"][key] for key in ("peptide_seq_column", "charge_column", "uniprot_column", "gene_names_column")]
        wide_rows = []
        for _, group in converted.groupby(columns, sort=False):
            first = group.iloc[0]
            record = {col: first[col] for col in columns}
            indices = group.index
            for idx in indices:
                sample = rows.loc[idx, "run"].removesuffix(".raw")
                record[sample+" Intensity"] = rows.loc[idx, "intensity"]
                record[sample+" Spectral Count"] = 1
            wide_rows.append(record)
        converted = pd.DataFrame(wide_rows)
    else:
        # DIA-NN needs annotations; use them for both long vendor mappings.
        ann = rows[["run", "condition", "replicate"]].drop_duplicates()
        ann.columns = ["File Name", "Condition", "Replicate"]
        annotation = tmp_path/"annotation.tsv"
        ann.to_csv(annotation, sep="\t", index=False)
        config["dataset"]["annotation_file"] = str(annotation)
    destination = tmp_path/"vendor.tsv"
    converted.to_csv(destination, sep="\t", index=False)
    config["dataset"]["input_file"] = str(destination)
    config["preprocessing"] = deepcopy(base["preprocessing"])
    actual = Dataset(**config).get_anndata()[expected.obs_names, expected.var_names]
    assert_allclose(actual.layers["raw"], expected.layers["raw"], rtol=3e-7, equal_nan=True)
    assert_allclose(actual.layers["normalized"], expected.layers["normalized"], rtol=3e-7, equal_nan=True)


def test_annotation_overrides_labels_normalizes_extensions_and_respects_exclusion(protein_config, tmp_path):
    ann = pd.DataFrame({"File Name": [f"{c}_{r}.RAW" for c in "ABC" for r in (1, 2, 3)],
                        "Condition": ["1 control"]*3+["drug-dose"]*3+["extra"]*3})
    # Excluded A_3 need not appear in annotations. Missing Replicate is inferred.
    ann = ann[ann["File Name"] != "A_3.RAW"].sample(frac=1, random_state=3)
    path = tmp_path/"annotation.tsv"
    ann.to_csv(path, sep="\t", index=False)
    protein_config["dataset"].update(annotation_file=str(path), exclude_runs=["A_3.mzML"])
    obj = Dataset(**protein_config).get_anndata()
    assert "A_3" not in obj.obs_names
    assert obj.obs.loc["A_1", "CONDITION"] == "C_1_control"
    assert obj.obs.loc["A_1", "CONDITION_ORIG"] == "1 control"
    assert obj.obs.loc["B_2", "ALIGN_KEY"] == "drug_dose#R2"


def test_unannotated_kept_run_fails(protein_config, tmp_path):
    path = tmp_path/"annotation.tsv"
    pd.DataFrame({"File Name": ["A_1"], "Condition": ["control"]}).to_csv(path, sep="\t", index=False)
    protein_config["dataset"]["annotation_file"] = str(path)
    with pytest.raises(ValueError, match="mismatch|not.*annotat|missing"):
        Dataset(**protein_config)


@pytest.mark.parametrize("column", ["condition", "replicate"])
def test_required_sample_metadata_without_annotation_fails(protein_config, column):
    path = protein_config["dataset"]["input_file"]
    pd.read_csv(path, sep="\t").drop(columns=column).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match=f"Missing required columns.*{column.upper()}.*annotation_file"):
        Dataset(**protein_config)


def test_annotation_requires_condition_column(protein_config, tmp_path):
    path = tmp_path/"annotation.tsv"
    pd.DataFrame({"File Name": ["A_1"]}).to_csv(path, sep="\t", index=False)
    protein_config["dataset"]["annotation_file"] = str(path)
    with pytest.raises(ValueError, match="missing required columns.*Condition"):
        Dataset(**protein_config)


def test_absent_optional_confidence_columns_do_not_fabricate_scores(protein_config):
    path = protein_config["dataset"]["input_file"]
    pd.read_csv(path, sep="\t").drop(columns=["qvalue", "pep"]).to_csv(path, sep="\t", index=False)
    obj = Dataset(**protein_config).get_anndata()
    assert set(obj.var_names) == {f"P{i:03}" for i in range(14)}
    assert "qvalue" not in obj.layers and "pep" not in obj.layers
    # A zero intensity is still censored without an identification score.
    assert np.isnan(obj.layers["raw"][obj.obs_names.get_loc("A_1"), obj.var_names.get_loc("P003")])


def test_runtime_defaults_are_distinct_from_template_settings():
    runtime = Preprocessor()
    template = yaml.safe_load(resources.files("proteoflux.templates").joinpath("spectronaut-proteomics.yaml").read_text())
    configured = Preprocessor(template["preprocessing"])
    assert runtime.protein_rollup_method == "sum"
    assert runtime.filter_pep == .2 and runtime.min_linear_intensity is None
    assert runtime.directlfq_cores == 4
    assert configured.protein_rollup_method == "directlfq"
    assert configured.filter_pep == .05 and configured.min_linear_intensity == 2
    assert configured.directlfq_cores == 8


@pytest.mark.parametrize("config, message", [
    ({"protein_rollup_method": "sum", "pivot_signal_method": "sum"}, "both"),
    ({"protein_rollup_method": "unknown"}, "Invalid protein_rollup"),
    ({"peptide_rollup_method": "top3"}, "Invalid peptide_rollup"),
    ({"filtering": {"pep_direction": "backwards"}}, "pep_direction"),
    ({"phospho": {"localization_filter_mode": "all"}}, "localization_mode"),
])
def test_invalid_preprocessing_options_fail_at_construction(config, message):
    with pytest.raises(ValueError, match=message):
        Preprocessor(config)


@pytest.mark.parametrize("value, expected", [(None, "proteomics"), ("proteo", "proteomics"),
                                             ("phosphoproteomics", "phospho"), ("pelsa", "pelsa"),
                                             ("peptido", "peptidomics")])
def test_analysis_aliases(value, expected):
    assert normalize_analysis_type(value) == expected


def test_cli_templates_copy_packaged_config_verbatim(tmp_path):
    runner = CliRunner()
    listing = runner.invoke(app, ["templates"])
    assert listing.exit_code == 0
    names = listing.stdout.splitlines()
    assert set(names) == {"spectronaut-proteomics", "spectronaut-peptidomics", "spectronaut-phosphoproteomics",
                          "diann-proteomics", "diann-peptidomics", "fragpipe-proteomics", "fragpipe-peptidomics"}
    for name in names:
        path = tmp_path/(name+".yaml")
        result = runner.invoke(app, ["init", name, "--path", str(path)])
        assert result.exit_code == 0, result.output
        assert path.read_bytes() == resources.files("proteoflux.templates").joinpath(name+".yaml").read_bytes()
