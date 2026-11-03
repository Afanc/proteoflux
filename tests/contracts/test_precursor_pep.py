from importlib import resources

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.export.de_exporter import DEExporter
from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.workflow.dataset import Dataset

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("filter_column", ["PG.PEP", "EG.PEP"])
def test_precursor_pep_mapping_preserves_separate_and_shared_filter_sources(filter_column):
    source = pl.DataFrame({"EG.PEP": [.1, None, .9], "PG.PEP": [.001, .002, .003]})
    harmonizer = DataHarmonizer({"pep_column": filter_column, "precursor_pep_column": "EG.PEP"})
    result = harmonizer._rename_columns_safely(source)
    assert result["PRECURSOR_PEP"].to_list() == source["EG.PEP"].to_list()
    assert result["PEP"].to_list() == source[filter_column].to_list()
    with pytest.raises(ValueError, match="PRECURSOR_PEP.*already exists"):
        harmonizer._rename_columns_safely(source.with_columns(pl.lit(.5).alias("PRECURSOR_PEP")))


def test_templates_map_vendor_precursor_metadata():
    for path in resources.files("proteoflux.templates").iterdir():
        if path.name.endswith(".yaml"):
            dataset = yaml.safe_load(path.read_text())["dataset"]
            vendor = path.name.split("-", 1)[0]
            assert dataset["precursor_pep_column"] == {
                "spectronaut": "EG.PEP", "diann": "PEP", "fragpipe": None,
            }[vendor], path.name
            assert dataset["other_proteins_column"] == {
                "spectronaut": "PEP.AllOccurringProteinAccessions", "diann": "Protein.Ids", "fragpipe": None,
            }[vendor], path.name
            if vendor == "diann":
                assert dataset["qvalue_column"] == "PG.Q.Value", path.name
                assert dataset["pep_column"] == "PG.PEP", path.name
                source = pl.DataFrame({"PG.PEP": [.002], "PEP": [.8], "Protein.Ids": ["P1;P2"], "Protein.Group": ["P1"]})
                result = DataHarmonizer(dataset)._rename_columns_safely(source)
                assert result["PEP"].to_list() == [.002]
                assert result["PRECURSOR_PEP"].to_list() == [.8]
                assert result["OTHER_PROTEINS"].to_list() == ["P1;P2"]


@pytest.mark.parametrize("mapping", [None, "missing_column"])
def test_unmapped_or_absent_precursor_pep_is_not_exported(protein_config, tmp_path, mapping):
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t").assign(**{"EG.PEP": .9})
    rows.to_csv(path, sep="\t", index=False)
    protein_config["dataset"]["precursor_pep_column"] = mapping
    result = Dataset(**protein_config).get_anndata()
    assert "best_precursor_pep" not in result.uns["peptides"]
    exporter = DEExporter(result, tmp_path / "absent.xlsx", config=protein_config)
    table = pd.read_excel(exporter.export(), sheet_name="Peptides (raw)")
    assert "BEST_PRECURSOR_PEP" not in table
    assert "OTHER_PROTEINS" not in table


@pytest.mark.parametrize("analysis", ["proteomics", "peptidomics"])
def test_shared_precursor_pep_mapping_keeps_existing_filtering(protein_config, analysis):
    protein_config["dataset"]["analysis_type"] = analysis
    baseline = Dataset(**protein_config).get_anndata()
    protein_config["dataset"]["precursor_pep_column"] = "pep"
    result = Dataset(**protein_config).get_anndata()[baseline.obs_names, baseline.var_names]
    assert_frame_equal(result.var.sort_index(axis=1), baseline.var.sort_index(axis=1))
    assert_array_equal(result.X, baseline.X)
    for key in baseline.layers:
        assert_array_equal(result.layers[key], baseline.layers[key])
    if analysis == "proteomics":
        assert_allclose(result.uns["peptides"]["best_precursor_pep"], .002)
    else:
        assert "peptides" not in result.uns


@pytest.mark.parametrize("other_proteins", [True, False])
def test_precursor_pep_h5ad_and_excel_metadata_only(protein_config, tmp_path, other_proteins):
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t")
    rows["EG.PEP"] = np.where(rows.sequence.str.endswith("A"), .5, .4)
    extra = rows.copy()
    extra["charge"] += 2
    extra["EG.PEP"] = np.where(extra.sequence.str.endswith("A"), .8, .7)
    # Existing methionine-oxidation collapse also defines exported peptide rows.
    extra.loc[extra.protein == "P004", "sequence"] = extra.loc[
        extra.protein == "P004", "sequence"
    ].str.replace("M", "M[Oxidation (M)]", regex=False)
    rows = pd.concat([rows, extra], ignore_index=True)
    rows.loc[rows.protein == "P000", "EG.PEP"] = .2  # identical contributing values
    rows.loc[rows.protein == "P001", "EG.PEP"] = np.nan  # entirely missing
    rows.loc[(rows.protein == "P002") & (rows.charge < 4), "EG.PEP"] = np.nan
    rows.loc[rows.protein == "P003", "EG.PEP"] = 0.
    rows.loc[rows.protein == "P005", "EG.PEP"] = np.where(
        rows.loc[rows.protein == "P005", "charge"] < 4, -np.inf, np.inf
    )  # no finite evidence
    rows.loc[(rows.protein == "P006") & (rows.charge >= 4), "EG.PEP"] = -np.inf
    rows.loc[rows.qvalue > .01, "EG.PEP"] = .001  # excluded rows must not contribute
    rows["matches"] = "P000;P001"
    rows.to_csv(path, sep="\t", index=False)
    if other_proteins:
        protein_config["dataset"]["other_proteins_column"] = "matches"
    baseline = Dataset(**protein_config).get_anndata()
    protein_config["dataset"]["precursor_pep_column"] = "EG.PEP"
    result = Dataset(**protein_config).get_anndata()

    assert_frame_equal(result.var.sort_index(axis=1), baseline.var.sort_index(axis=1))
    assert_array_equal(result.X, baseline.X)
    for key in baseline.layers:
        assert_array_equal(result.layers[key], baseline.layers[key])
    assert result.uns.keys() == baseline.uns.keys()
    for key in baseline.uns["peptides"]:
        assert_array_equal(result.uns["peptides"][key], baseline.uns["peptides"][key])
    baseline = run_limma_pipeline(baseline, protein_config)
    result = run_limma_pipeline(result, protein_config)
    for key in baseline.varm:
        assert_array_equal(result.varm[key], baseline.varm[key])

    exporter = DEExporter(result, tmp_path / "metadata.xlsx", config=protein_config)
    h5ad = tmp_path / "metadata.h5ad"
    exporter.export_adata(str(h5ad))
    restored = ad.read_h5ad(h5ad)
    peptides = restored.uns["peptides"]
    expected = [{"P000": .2, "P001": np.nan, "P003": 0., "P005": np.nan,
                 "P002": .8 if sequence.endswith("A") else .7}.get(
                    protein, .5 if sequence.endswith("A") else .4)
                for protein, sequence in zip(peptides["protein_index"], peptides["peptide_seq"])]
    assert_allclose(peptides["best_precursor_pep"], expected, rtol=0, atol=0, equal_nan=True)
    assert_array_equal(peptides["rows"], result.uns["peptides"]["rows"])
    assert "BEST_PRECURSOR_PEP" not in restored.var

    workbook = exporter.export()
    readme = pd.read_excel(workbook, sheet_name="README")
    assert not readme.README.str.contains("BEST_PRECURSOR", na=False).any()
    table = pd.read_excel(workbook, sheet_name="Peptides (raw)")
    assert_array_equal(table.PEPTIDE_SEQ, peptides["peptide_seq"])
    assert_array_equal(table.PROTEIN_UNIPROT_AC, peptides["protein_index"])
    assert_allclose(table.BEST_PRECURSOR_PEP, expected, rtol=0, atol=0, equal_nan=True)
    if other_proteins:
        assert table.columns.get_loc("OTHER_PROTEINS") == table.columns.get_loc("BEST_PRECURSOR_PEP") + 1
        assert table.columns.get_loc("OTHER_PROTEINS") == table.columns.get_loc(peptides["cols"][0]) - 1
        assert_array_equal(table.OTHER_PROTEINS, peptides["other_proteins"])
    else:
        assert "OTHER_PROTEINS" not in table
        assert table.columns.get_loc("BEST_PRECURSOR_PEP") == table.columns.get_loc(peptides["cols"][0]) - 1
        assert "other_proteins" not in result.uns["peptides"]
    assert_allclose(table[list(peptides["cols"])], peptides["raw"], equal_nan=True)
    summary = pd.read_excel(workbook, sheet_name="Summary", index_col=0)
    assert "BEST_PRECURSOR_PEP" not in summary
    original = DEExporter(baseline, tmp_path / "baseline.xlsx", config=protein_config).export()
    original_summary = pd.read_excel(original, sheet_name="Summary", index_col=0)
    assert_frame_equal(summary[original_summary.columns], original_summary)
