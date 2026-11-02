from importlib import resources

import anndata as ad
import pandas as pd
import polars as pl
import pytest
import yaml
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.analysis.pelsa_pipeline import run_pelsa_pipeline
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.workflow.dataset import Dataset
from tests.fixtures.workflows import protein_workflow, phospho_workflow, pelsa_workflow

pytestmark = pytest.mark.contract
VENDOR_COLUMN = "PEP.AllOccurringProteinAccessions"


def test_other_proteins_mapping_preserves_vendor_value():
    harmonizer = DataHarmonizer({"other_proteins_column": VENDOR_COLUMN})
    result = harmonizer._rename_columns_safely(pl.DataFrame({VENDOR_COLUMN: ["P1;P2;P3"]}))
    assert result.columns == ["OTHER_PROTEINS"]
    assert result["OTHER_PROTEINS"].to_list() == ["P1;P2;P3"]
    assert DataHarmonizer({"other_proteins_column": None}).column_map == {}


@pytest.mark.parametrize("analysis", ["proteomics", "peptidomics", "phosphoproteomics", "pelsa"])
def test_spectronaut_templates_map_other_proteins(analysis):
    template = resources.files("proteoflux.templates").joinpath(f"spectronaut-{analysis}.yaml")
    assert yaml.safe_load(template.read_text())["dataset"]["other_proteins_column"] == VENDOR_COLUMN


@pytest.mark.parametrize("analysis, rollup", [
    ("proteomics", "sum"), ("proteomics", "directlfq"), ("proteomics", "robust_regression"),
    ("peptidomics", "sum"), ("phospho", "sum"), ("pelsa", "sum"),
])
def test_other_proteins_exports_preserve_quantification_and_statistics(tmp_path, analysis, rollup):
    if analysis == "phospho":
        config = phospho_workflow(tmp_path, reuse=True, kinase=False)
    elif analysis == "pelsa":
        config = pelsa_workflow(tmp_path)
    else:
        config = protein_workflow(tmp_path, analysis)
    config["preprocessing"]["protein_rollup_method"] = rollup
    config["dataset"]["protein_descriptions_column"] = "description"
    path = config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t", dtype={"sites": str, "positions": str, "probabilities": str})
    rows["description"] = "Protein " + rows.protein
    rows[VENDOR_COLUMN] = "P1;P2;P3"
    if analysis == "proteomics":
        # Different peptides assigned to the same protein retain distinct lists.
        rows.loc[rows.sequence.str.endswith("A"), VENDOR_COLUMN] = "P3;P1;P12345-2"
    if analysis == "phospho":
        selected = rows.index[(rows.protein == "P000") & (rows.locations == "S6")]
        rows.loc[selected[0], VENDOR_COLUMN] = "P3;P1;P2"
        rows.loc[selected[1:], VENDOR_COLUMN] = "P2;Q67890;P12345-2"
        rows.loc[rows.sites.isna(), VENDOR_COLUMN] = "BACKGROUND_ONLY"
    rows.to_csv(path, sep="\t", index=False)

    baseline = Dataset(**config).get_anndata()
    config["dataset"]["other_proteins_column"] = VENDOR_COLUMN
    result = Dataset(**config).get_anndata()
    assert_array_equal(result.X, baseline.X)
    assert_frame_equal(result.obs, baseline.obs)
    metadata = result.var.drop(columns="OTHER_PROTEINS", errors="ignore")
    # Existing per-condition consistency columns may arrive in a different order.
    # Compare them by name; feature/sample order and all values remain exact.
    assert set(metadata.columns) == set(baseline.var.columns)
    assert_frame_equal(metadata[baseline.var.columns], baseline.var)
    for key in baseline.layers:
        assert_array_equal(result.layers[key], baseline.layers[key])
    if analysis == "proteomics":
        for key in baseline.uns["peptides"]:
            assert_array_equal(result.uns["peptides"][key], baseline.uns["peptides"][key])

    analyze = run_pelsa_pipeline if analysis == "pelsa" else run_limma_pipeline
    baseline = analyze(baseline, config)
    result = analyze(result, config)
    for key in baseline.varm:
        assert_array_equal(result.varm[key], baseline.varm[key])
    if analysis == "pelsa":
        assert_frame_equal(result.uns["pelsa"]["curve_results"], baseline.uns["pelsa"]["curve_results"])

    h5ad = tmp_path / "metadata.h5ad"
    exporter = DEExporter(result, tmp_path / "metadata.xlsx", config=config)
    exporter.export_adata(str(h5ad))
    summary = pd.read_excel(exporter.export(), sheet_name="Summary", index_col=0)
    restored = ad.read_h5ad(h5ad)
    if analysis == "proteomics":
        assert "OTHER_PROTEINS" not in restored.var and "OTHER_PROTEINS" not in summary
        original_path = DEExporter(baseline, tmp_path / "baseline.xlsx", config=config).export()
        original_summary = pd.read_excel(original_path, sheet_name="Summary", index_col=0)
        assert set(summary.columns) == set(original_summary.columns)
        assert_frame_equal(summary[original_summary.columns], original_summary)
        peptides = restored.uns["peptides"]
        expected = ["P3;P1;P12345-2" if seq.endswith("A") else "P1;P2;P3"
                    for seq in peptides["peptide_seq"]]
        assert_array_equal(peptides["other_proteins"], expected)
        table = pd.read_excel(tmp_path / "metadata.xlsx", sheet_name="Peptides (raw)")
        assert_array_equal(table.PEPTIDE_SEQ, peptides["peptide_seq"])
        assert_array_equal(table.OTHER_PROTEINS, expected)
        assert table.columns.get_loc("OTHER_PROTEINS") == table.columns.get_loc("PROTEIN_DESCRIPTIONS") + 1
    else:
        assert_array_equal(restored.var.OTHER_PROTEINS, result.var.OTHER_PROTEINS)
        assert_array_equal(summary.loc[result.var_names, "OTHER_PROTEINS"], result.var.OTHER_PROTEINS)
        assert restored.var.columns.get_loc("OTHER_PROTEINS") == restored.var.columns.get_loc("PROTEIN_DESCRIPTIONS") + 1
        assert summary.columns.get_loc("OTHER_PROTEINS") == summary.columns.get_loc("PROTEIN_DESCRIPTIONS") + 1
        if analysis == "phospho":
            assert result.var.loc["P000|S6", "OTHER_PROTEINS"] == "P3;P1;P2;Q67890;P12345-2"
            assert "BACKGROUND_ONLY" not in ";".join(result.var.OTHER_PROTEINS)
        else:
            assert set(result.var.OTHER_PROTEINS) == {"P1;P2;P3"}


@pytest.mark.parametrize("mapping", [None, "missing_column", VENDOR_COLUMN])
def test_absent_or_empty_other_proteins_does_not_invent_accessions(protein_config, mapping):
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t")
    rows[VENDOR_COLUMN] = [None if i % 2 else " ; " for i in range(len(rows))]
    rows.to_csv(path, sep="\t", index=False)
    protein_config["dataset"]["other_proteins_column"] = mapping
    data = Dataset(**protein_config).get_anndata()
    assert "OTHER_PROTEINS" not in data.var
    if mapping == VENDOR_COLUMN:
        assert set(data.uns["peptides"]["other_proteins"]) == {""}
    else:
        assert "other_proteins" not in data.uns["peptides"]
