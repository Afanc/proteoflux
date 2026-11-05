from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from proteoflux.analysis.limma_pipeline import run_limma_pipeline
from proteoflux.export.de_exporter import DEExporter
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.protein_coverage import (
    load_protein_coverage_fasta, protein_coverage_metadata, protein_coverage_payload,
)
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


@pytest.mark.parametrize("prefix", ["ma", "custom"])
def test_three_part_fasta_header_maps_accession_regardless_of_prefix(tmp_path, prefix):
    path = tmp_path / "manual.fasta"
    path.write_text(
        f">{prefix}|G12345|VHHGFP_NANO MANUAL entry - VhH GFPtrap-Biotin-6HIS Nanobody GN=VhH\n"
        "MDQVQLVESGGALVQPGGSLR\n"
    )
    fasta = load_protein_coverage_fasta({"protein_coverage": {"fasta": path}}, "proteomics")
    assert fasta["G12345"] == fasta[f"{prefix.upper()}|G12345|VHHGFP_NANO"]
    metadata = protein_coverage_metadata(["G12345"], ["MDQVQLVESGGALVQPGGSLR"], fasta)
    assert metadata["peptide_mapping_status"] == ["mapped"]
    assert_array_equal(metadata["peptide_start"], [1])
    assert_array_equal(metadata["peptide_end"], [21])


def test_three_part_fasta_headers_reject_conflicting_accession_aliases(tmp_path):
    path = tmp_path / "manual.fasta"
    path.write_text(
        ">ma|G12345|FIRST\nACDEK\n"
        ">custom|G12345|SECOND\nFGHIR\n"
    )
    with pytest.raises(ValueError, match="G12345.*conflicting sequences"):
        load_protein_coverage_fasta({"protein_coverage": {"fasta": path}}, "proteomics")


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
    if ";" not in accession:
        payload = protein_coverage_payload(["peptide-id"], [accession], [peptide], coverage_fasta)
        assert payload["version"] == 1
        references = payload["references"]
        assert references["protein_index"] == [accession]
        assert references["accession"] == [accession]
        assert_array_equal(references["protein_length_aa"], [length])
        assert_array_equal(references["accession_order"], [0])
        mappings = payload["peptide_mappings"]
        assert mappings["protein_index"] == [accession]
        assert mappings["peptide_id"] == ["peptide-id"]
        assert mappings["accession"] == [accession]
        for key in ("peptide_start", "peptide_end", "peptide_mapping_status", "tryptic_missed_cleavages"):
            assert_array_equal(mappings[key], result[key])


@pytest.mark.parametrize("group, accessions, lengths, starts", [
    ("A;B;C", ["A", "B", "C"], [9, 11, 13], [2, 3, 4]),
    ("C;A;B", ["C", "A", "B"], [13, 9, 11], [4, 2, 3]),
])
def test_group_payload_keeps_each_reference_and_peptide_coordinates(group, accessions, lengths, starts):
    # FASTA order differs from group order; one accession also has its own feature.
    fasta = {"B": "GGPEPTIDEQQ", "C": "VVVPEPTIDEQQQ", "A": "MPEPTIDEQ"}
    groups = [group, "A", group]
    peptide_ids = ["group-peptide", "single-peptide", "group-fragment"]
    sequences = ["PEPTIDE", "PEPTIDE", "TIDE"]
    payload = protein_coverage_payload(peptide_ids, groups, sequences, fasta)
    assert set(payload) == {"version", "references", "peptide_mappings"}
    references = payload["references"]
    assert set(references) == {"protein_index", "accession", "protein_length_aa", "accession_order"}
    assert references["protein_index"] == [group] * 3 + ["A"]
    assert references["accession"] == accessions + ["A"]
    assert_array_equal(references["protein_length_aa"], lengths + [9])
    assert_array_equal(references["accession_order"], [0, 1, 2, 0])
    assert references["protein_length_aa"].dtype == np.float64
    assert references["accession_order"].dtype == np.int64
    mappings = payload["peptide_mappings"]
    assert set(mappings) == {
        "protein_index", "peptide_id", "accession", "peptide_start", "peptide_end",
        "peptide_mapping_status", "tryptic_missed_cleavages",
    }
    assert mappings["protein_index"] == [group] * 3 + ["A"] + [group] * 3
    assert mappings["peptide_id"] == ["group-peptide"] * 3 + ["single-peptide"] + ["group-fragment"] * 3
    assert mappings["accession"] == accessions + ["A"] + accessions
    assert_array_equal(mappings["peptide_start"], starts + [2] + [start + 3 for start in starts])
    assert_array_equal(mappings["peptide_end"], [start + 6 for start in starts] + [8]
                       + [start + 6 for start in starts])
    assert mappings["peptide_mapping_status"] == ["mapped"] * 7
    assert_array_equal(mappings["tryptic_missed_cleavages"], [0] * 7)
    assert mappings["peptide_start"].dtype == mappings["peptide_end"].dtype == np.float64
    assert mappings["tryptic_missed_cleavages"].dtype == np.int64
    assert groups == [group, "A", group]
    assert peptide_ids == ["group-peptide", "single-peptide", "group-fragment"]
    assert sequences == ["PEPTIDE", "PEPTIDE", "TIDE"]


def test_missing_group_members_and_exact_isoforms(coverage_fasta):
    group = "P12345-3;P12345-2;ABSENT;P12345"
    payload = protein_coverage_payload(["p"], [group], ["ACDKRPEPTIDER"], coverage_fasta)
    references, mappings = payload["references"], payload["peptide_mappings"]
    assert references["accession"] == mappings["accession"] == ["P12345-3", "P12345-2", "ABSENT", "P12345"]
    assert references["protein_index"] == mappings["protein_index"] == [group] * 4
    assert_array_equal(references["accession_order"], [0, 1, 2, 3])
    assert_array_equal(references["protein_length_aa"], [np.nan, 15, np.nan, 16])
    assert mappings["peptide_mapping_status"] == ["unmapped_reference", "mapped", "unmapped_reference", "mapped"]
    assert_array_equal(mappings["peptide_start"], [np.nan, 3, np.nan, 2])
    assert_array_equal(mappings["peptide_end"], [np.nan, 15, np.nan, 14])
    assert_array_equal(mappings["tryptic_missed_cleavages"], [1] * 4)


def test_repeated_peptide_is_ambiguous_only_in_its_reference(coverage_fasta):
    fasta = {**coverage_fasta, "UNIQUE": "MAAAK"}
    payload = protein_coverage_payload(["p"], ["REPEAT;UNIQUE;P12345"], ["AAA"], fasta)
    mappings = payload["peptide_mappings"]
    assert mappings["accession"] == ["REPEAT", "UNIQUE", "P12345"]
    assert mappings["peptide_mapping_status"] == ["ambiguous_peptide", "mapped", "unmapped_peptide"]
    assert_array_equal(mappings["peptide_start"], [np.nan, 2, np.nan])
    assert_array_equal(mappings["peptide_end"], [np.nan, 4, np.nan])


def test_group_tokens_preserve_original_id_and_first_accession_positions():
    group = " B ;A;;B; C "
    payload = protein_coverage_payload(["p"], [group], ["PEP"], {"A": "PEP", "B": "PEP", "C": "PEP"})
    assert payload["references"]["protein_index"] == [group] * 3
    assert payload["references"]["accession"] == ["B", "A", "C"]
    assert_array_equal(payload["references"]["accession_order"], [0, 1, 4])
    assert payload["peptide_mappings"]["protein_index"] == [group] * 3
    assert payload["peptide_mappings"]["accession"] == ["B", "A", "C"]


@pytest.mark.contract
@pytest.mark.parametrize("multiple_files", [False, True])
def test_optional_coverage_preserves_peptide_axes_intensities_and_h5ad(protein_config, tmp_path, multiple_files):
    # Keep isoforms and ambiguous groups intact through the real input path.
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t")
    replacements = {"P000": "P000-2", "P001": "A;B;C", "P004": "ABSENT;P004"}
    for column in ("protein", "accession"):
        rows[column] = rows[column].replace(replacements)
    rows.to_csv(path, sep="\t", index=False)
    baseline = Dataset(**protein_config).get_anndata()
    fasta = tmp_path / "coverage.fasta"
    records = []
    for g, sequence in enumerate(SEQUENCES):
        if g != 1:
            accession = "P000-2" if g == 0 else f"P{g:03}"
            records.append(f">sp|{accession}|REFERENCE\nM{sequence}AAA\n")
    records.extend(f">{accession}\n{prefix}{SEQUENCES[1]}AAA\n"
                   for accession, prefix in [("C", "VVV"), ("A", "M"), ("B", "GG")])
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
    assert "protein_coverage" not in baseline.uns
    assert_array_equal(result.X, baseline.X)
    assert_frame_equal(result.obs, baseline.obs)
    assert set(result.var.columns) == set(baseline.var.columns)
    assert_frame_equal(result.var[baseline.var.columns], baseline.var)
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
        if ";" in protein:
            assert payload["peptide_mapping_status"][i] == "ambiguous_protein_group"
            assert np.isnan(payload["peptide_start"][i])

    coverage = result.uns["protein_coverage"]
    references = pd.DataFrame(coverage["references"])
    group_references = references[references.protein_index == "A;B;C"]
    assert_array_equal(group_references.accession, ["A", "B", "C"])
    assert_array_equal(group_references.accession_order, [0, 1, 2])
    assert_array_equal(group_references.protein_length_aa, [13, 14, 15])
    missing = references[references.protein_index == "ABSENT;P004"]
    assert_array_equal(missing.accession, ["ABSENT", "P004"])
    assert_array_equal(missing.protein_length_aa, [np.nan, 13])
    mappings = pd.DataFrame(coverage["peptide_mappings"])
    for peptide_id, protein, sequence in zip(payload["rows"], payload["protein_index"], payload["peptide_seq"]):
        selected = mappings[mappings.peptide_id == peptide_id]
        if protein == "A;B;C":
            assert_array_equal(selected.accession, ["A", "B", "C"])
            assert_array_equal(selected.peptide_start, [2, 3, 4])
            assert_array_equal(selected.peptide_end, [1 + len(sequence), 2 + len(sequence), 3 + len(sequence)])
            assert_array_equal(selected.peptide_mapping_status, ["mapped"] * 3)
        elif protein == "ABSENT;P004":
            assert_array_equal(selected.accession, ["ABSENT", "P004"])
            assert_array_equal(selected.peptide_mapping_status, ["unmapped_reference", "mapped"])
            assert_array_equal(selected.peptide_start, [np.nan, 2])

    baseline = run_limma_pipeline(baseline, config)
    result = run_limma_pipeline(result, config)
    assert set(result.varm) == set(baseline.varm)
    for key in baseline.varm:
        assert_array_equal(result.varm[key], baseline.varm[key])
    output = tmp_path / "coverage.h5ad"
    exporter = DEExporter(result, tmp_path / "coverage.xlsx", config=config)
    exporter.export_adata(str(output))
    restored = ad.read_h5ad(output)
    assert_array_equal(restored.obs_names, baseline.obs_names)
    assert_array_equal(restored.var_names, baseline.var_names)
    for key in payload:
        assert_array_equal(restored.uns["peptides"][key], payload[key])
    assert restored.uns["protein_coverage"]["version"] == 1
    for table in ("references", "peptide_mappings"):
        for key, values in coverage[table].items():
            assert_array_equal(restored.uns["protein_coverage"][table][key], values)
    assert_array_equal(restored.X, baseline.X)
    for key in baseline.layers:
        assert_array_equal(restored.layers[key], baseline.layers[key])
    baseline_export = DEExporter(baseline, tmp_path / "baseline.xlsx", config=config).export()
    coverage_export = exporter.export()
    for sheet in ("Summary", "Peptides (raw)"):
        original_table = pd.read_excel(baseline_export, sheet_name=sheet)
        coverage_table = pd.read_excel(coverage_export, sheet_name=sheet)
        assert set(coverage_table.columns) == set(original_table.columns)
        assert_frame_equal(coverage_table[original_table.columns], original_table)


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
