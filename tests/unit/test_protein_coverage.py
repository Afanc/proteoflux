from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from proteoflux.export.de_exporter import DEExporter
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.protein_coverage import load_protein_coverage_fasta, protein_coverage_metadata
from tests.fixtures.workflows import SEQUENCES

pytestmark = pytest.mark.unit


@pytest.fixture
def coverage_fasta(tmp_path):
    path = tmp_path / "proteome.fasta"
    path.write_text(
        ">sp|P12345|CANONICAL\nMACDKRPEPTIDERAA\n"
        ">sp|P12345-2|ISOFORM\nGGACDKRPEPTIDER\n"
        ">P23456\nMACDKRPEPTIDERAA\n"
        ">REPEAT\nAAAAA\n"
    )
    return load_protein_coverage_fasta({"protein_coverage": {"fasta": path}}, "proteomics")


@pytest.mark.parametrize("accession, peptide, status, reference, length, start, end, missed", [
    ("P12345", "ACDKRPEPTIDER", "mapped", "P12345", 16, 2, 14, 1),
    ("P12345-2", "ACDKRPEPTIDER", "mapped", "P12345-2", 15, 3, 15, 1),
    ("P12345-3", "ACDKRPEPTIDER", "unmapped_reference", "", np.nan, np.nan, np.nan, 1),
    ("P12345;P23456", "ACDKRPEPTIDER", "ambiguous_protein_group", "", np.nan, np.nan, np.nan, 1),
    ("P12345;ABSENT", "ACDKRPEPTIDER", "ambiguous_protein_group", "", np.nan, np.nan, np.nan, 1),
    ("REPEAT", "AAA", "ambiguous_peptide", "REPEAT", 5, np.nan, np.nan, 0),
    ("P12345", "TTTT", "unmapped_peptide", "P12345", 16, np.nan, np.nan, 0),
    ("P12345", "_AC[Carbamidomethyl]DKRPEPTIDER_", "mapped", "P12345", 16, 2, 14, 1),
    ("ABSENT", "KRKPRAK", "unmapped_reference", "", np.nan, np.nan, np.nan, 3),
])
def test_exact_reference_and_unique_occurrence_policy(
    coverage_fasta, accession, peptide, status, reference, length, start, end, missed,
):
    result = protein_coverage_metadata([accession], [peptide], coverage_fasta)
    assert result["reference_accession"] == [reference]
    assert result["peptide_mapping_status"] == [status]
    for key, expected in [("protein_length_aa", length), ("peptide_start", start),
                          ("peptide_end", end), ("tryptic_missed_cleavages", missed)]:
        assert_array_equal(result[key], [expected])


@pytest.mark.contract
@pytest.mark.parametrize("multiple_files", [False, True])
def test_optional_coverage_preserves_peptide_axes_intensities_and_h5ad(protein_config, tmp_path, multiple_files):
    # Keep isoforms and ambiguous groups intact through the real input path.
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t")
    replacements = {"P000": "P000-2", "P001": "P001;P002"}
    for column in ("protein", "accession"):
        rows[column] = rows[column].replace(replacements)
    rows.to_csv(path, sep="\t", index=False)
    baseline = Dataset(**protein_config).get_anndata()
    fasta = tmp_path / "coverage.fasta"
    records = [
        f">sp|{replacements.get(f'P{g:03}', f'P{g:03}')}|REFERENCE\nM{sequence}AAA\n"
        for g, sequence in enumerate(SEQUENCES) if g != 1
    ]
    fasta.write_text("".join(records))
    fasta_value = str(fasta)
    if multiple_files:
        other = tmp_path / "other-species.fasta"
        fasta.write_text("".join(records[::2]))
        other.write_text("".join(records[1::2]))
        fasta_value = [str(fasta), str(other)]
    config = deepcopy(protein_config)
    config["analysis"]["protein_coverage"] = {"fasta": fasta_value}
    # This separate option must neither supply nor override the coverage FASTA.
    config["analysis"]["kinase_activity"] = {"fasta": str(tmp_path / "unused.fasta")}
    result = Dataset(**config).get_anndata()
    assert_array_equal(result.X, baseline.X)
    for key in baseline.layers:
        assert_array_equal(result.layers[key], baseline.layers[key])
    original = baseline.uns["peptides"]
    payload = result.uns["peptides"]
    for key in original:
        assert_array_equal(payload[key], original[key])
    assert len(set(payload) - set(original)) == 6
    assert payload["reference_accession"] == [
        protein if ";" not in protein else "" for protein in payload["protein_index"]
    ]
    for i, protein in enumerate(payload["protein_index"]):
        if protein == "P000-2":
            assert payload["reference_accession"][i] == "P000-2"
            assert payload["peptide_start"][i] == 2
        if protein == "P001;P002":
            assert payload["peptide_mapping_status"][i] == "ambiguous_protein_group"
            assert np.isnan(payload["peptide_start"][i])
    output = tmp_path / "coverage.h5ad"
    DEExporter(result, output_path=None, config=config).export_adata(str(output))
    restored = ad.read_h5ad(output)
    assert_array_equal(restored.obs_names, baseline.obs_names)
    assert_array_equal(restored.var_names, baseline.var_names)
    for key in payload:
        assert_array_equal(restored.uns["peptides"][key], payload[key])


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_accessions_across_fastas_are_order_independent(tmp_path, conflicting):
    first, second = tmp_path / "first.fasta", tmp_path / "second.fasta"
    first.write_text(">sp|P12345|FIRST\nACDEK\n")
    duplicate = "ACDER" if conflicting else "ACDEK"
    second.write_text(f">sp|P12345|SECOND\n{duplicate}\n>P23456\nFGHIR\n")
    original = load_protein_coverage_fasta({"protein_coverage": {"fasta": first}}, "proteomics")
    merged = []
    for paths in ([first, str(second)], [str(second), first]):
        config = {"protein_coverage": {"fasta": paths}}
        if conflicting:
            with pytest.raises(ValueError, match="P12345.*conflicting sequences"):
                load_protein_coverage_fasta(config, "proteomics")
        else:
            merged.append(load_protein_coverage_fasta(config, "proteomics"))
    if not conflicting:
        assert merged[0] == merged[1]
        assert merged[0]["P12345"] == "ACDEK"
        assert merged[0]["P23456"] == "FGHIR"
    # Merging must not modify the existing reader's cached single-file records.
    assert load_protein_coverage_fasta({"protein_coverage": {"fasta": first}}, "proteomics") == original
    assert "P23456" not in original


@pytest.mark.parametrize("settings, error", [
    ([], ValueError), ({"fasta": 5}, ValueError), ({"fasta": "/missing/proteome.fasta"}, FileNotFoundError),
    ({"fasta": []}, ValueError), ({"fasta": [None]}, ValueError),
    ({"fasta": [""]}, ValueError), ({"fasta": [5]}, ValueError),
    ({"fasta": ["/missing/proteome.fasta"]}, FileNotFoundError),
])
def test_invalid_coverage_config_fails_before_input_io(protein_config, settings, error):
    protein_config["dataset"]["input_file"] = "/missing/input.tsv"
    protein_config["analysis"]["protein_coverage"] = settings
    with pytest.raises(error, match="protein_coverage|Protein-coverage"):
        Dataset(**protein_config)


@pytest.mark.parametrize("settings", [None, {}, {"fasta": None}, {"fasta": ""}])
def test_coverage_is_optional_and_independent_of_kinase_fasta(settings):
    assert load_protein_coverage_fasta({
        "protein_coverage": settings, "kinase_activity": {"fasta": "/unused.fasta"},
    }, "proteomics") is None
