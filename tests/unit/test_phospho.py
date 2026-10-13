import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.utils.harmonizer import DataHarmonizer
from proteoflux.workflow.dataset import Dataset
from proteoflux.analysis.kinase_sequences import prepare_kinase_sequences
from tests.fixtures.workflows import phospho_workflow

pytestmark = pytest.mark.unit


def site_frame(probabilities="95;80;99"):
    return pl.DataFrame({"UNIPROT": ["P1;P2"], "PEP_SEQUENCE": ["AASATAYK"],
                         "PEPTIDE_LSEQ": ["AAS[Phospho (STY)]AT[Phospho (STY)]AYK"],
                         "PEPTIDE_START": [100], "_PEPTIDE_STARTS_STR": ["100;200"],
                         "PTM_SITES_STR": ["2"], "PTM_POSITIONS_STR": ["3;5;7"],
                         "PTM_PROBS_STR": [probabilities],
                         "PTM_PROTEINLOCATIONS": ["S102,T104;S202,T204"], "SIGNAL": [100.]})


def test_declared_sites_percent_probabilities_and_accession_pairing():
    out = DataHarmonizer({"analysis_type": "phospho"})._build_phospho_index(site_frame()).sort("INDEX")
    assert out["INDEX"].to_list() == ["P1;P2|S102", "P1;P2|T104"]
    assert out["PEPTIDE_INDEX"].to_list() == ["AASATAYK|p3", "AASATAYK|p5"]
    assert_allclose(out["LOC_PROB"], [.95, .8])
    assert_allclose(out["SIGNAL"], [100, 100])  # duplicate, not intensity splitting
    assert out["ACCESSION_SITES"].to_list() == ["P1:S102;P2:S202", "P1:T104;P2:T204"]
    assert out["PHOSPHOFORM_POSITIONS"].to_list() == ["3;5", "3;5"]


def test_retain_uses_minimum_localization_and_mismatch_stays_unscored():
    harmonizer = DataHarmonizer({"analysis_type": "phospho", "phospho_multisite_collapse_policy": "retain"})
    out = harmonizer._build_phospho_index(site_frame())
    assert out.height == 1
    assert out["LOC_PROB"].item() == .8
    assert out["SIGNAL"].item() == 100
    unmatched = harmonizer._build_phospho_index(site_frame("95"))
    assert unmatched["LOC_PROB"].item() is None


def test_background_alignment_and_parent_shared_imputation(tmp_path):
    config = phospho_workflow(tmp_path, kinase=False)
    data = Dataset(**config).get_anndata()
    source = pd.read_csv(config["dataset"]["inject_runs"]["flowthrough"]["input_file"], sep="\t")
    source["sample"] = source.condition + "_" + source.replicate.astype(str)
    expected = source.pivot(index="sample", columns="protein", values="intensity")
    for parent in (f"P{i:03}" for i in range(6)):
        indices = [data.var_names.get_loc(parent+"|"+site) for site in ("S6", "T6")]
        for i in indices:
            assert_allclose(data.layers["raw_covariate"][:, i], expected.loc[data.obs_names, parent], rtol=3e-7, equal_nan=True)
        # Includes missing values filled stochastically: two site-specific draws
        # would generally disagree and would be the wrong unit of imputation.
        assert_array_equal(data.layers["processed_covariate"][:, indices[0]],
                           data.layers["processed_covariate"][:, indices[1]])
    processed = data.layers["processed_covariate"]
    assert_allclose(data.layers["centered_covariate"], processed-np.median(processed, axis=0), atol=1e-6)
    assert not any("background" in name for name in data.obs_names)


@pytest.mark.parametrize("empty_sample", [False, True])
def test_localization_drops_features_and_orphan_precursors(tmp_path, empty_sample):
    config = phospho_workflow(tmp_path, kinase=False)
    # This test isolates localization. Retaining only P000 would also leave an
    # empty aligned background sample; that independent policy is tested below.
    config["dataset"].pop("inject_runs")
    config["preprocessing"]["phospho"].update(localization_filter_mode="strict", localization_filter_threshold=.95)
    frame = pd.read_csv(config["dataset"]["input_file"], sep="\t")
    # Strict ignores absent runs; equality passes. Retain only P000 with all
    # finite probabilities >= threshold, while other sites contain .8.
    frame.loc[frame.accession == "P000", "probabilities"] = .95
    missing = (frame.accession == "P000") & (frame.run == "A_1.raw")
    if not empty_sample:
        # Keep T6 observed: a missing S6 is allowed, an empty sample is not.
        missing &= frame.locations == "S6"
    frame = frame[~missing]
    frame.to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    if empty_sample:
        with pytest.raises(ValueError, match="No observed quantitative values.*A_1"):
            Dataset(**config)
        return
    data = Dataset(**config).get_anndata()
    assert np.isfinite(data.layers["raw"][data.obs_names.get_loc("A_1")]).any()
    assert set(data.var_names) == {"P000|S6", "P000|T6"}
    assert set(data.uns["precursors"]["protein_index"]) == set(data.var_names)


def test_background_duplicate_and_unmatched_pairs_fail(tmp_path):
    config = phospho_workflow(tmp_path, kinase=False)
    path = config["dataset"]["inject_runs"]["flowthrough"]["input_file"]
    frame = pd.read_csv(path, sep="\t")
    frame.loc[frame.run == "background_A_1.raw", "replicate"] = 9
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="alignment|pair|ALIGN_KEY"):
        Dataset(**config)


def test_site_windows_use_local_positions_and_mark_auxiliary_phosphorylation(tmp_path):
    config = phospho_workflow(tmp_path, kinase=False)
    data = prepare_kinase_sequences(Dataset(**config).get_anndata(), config)
    assert data.var.loc["P000|S6", "KINASE_10MER"] == "AACDESPEPK"
    assert data.var.loc["P000|T6", "KINASE_10MER"] == "AACDETPEPK"
    assert set(data.var.KINASE_10MER.str.len()) == {10}
    assert set(data.var.SITE_15MER.str.len()) == {15}


def test_auxiliary_sites_and_fasta_extension_preserve_focal_site(tmp_path):
    import anndata as ad
    obj = ad.AnnData(np.ones((2, 2)), var=pd.DataFrame({"ACCESSION_SITES": ["P:S3", "P:S3;P:T5"]},
                                                     index=["P|S3", "P|S3,T5"]))
    obj.uns["precursors"] = {"protein_index": ["P|S3"], "peptide_seq": ["AASATAYK|p3"],
                              "phosphoform_positions": ["3;5"]}
    prepared = prepare_kinase_sequences(obj.copy(), {})
    assert prepared.var.loc["P|S3", "KINASE_10MER"] == "___AASAtAY"
    assert prepared.var.loc["P|S3,T5", "KINASE_10MER"] == "__________"
    fasta = tmp_path/"synthetic.fasta"
    fasta.write_text(">sp|P|SYNTHETIC\nAASATAYKLMNPQ\n")
    enriched = prepare_kinase_sequences(obj.copy(), {"analysis": {"kinase_activity": {"fasta": str(fasta)}}})
    assert enriched.var.loc["P|S3", "SITE_15MER"] == "_____AASAtAYKLM"
    assert enriched.var.loc["P|S3", "KINASE_10MER"] == prepared.var.loc["P|S3", "KINASE_10MER"]


def test_entirely_missing_background_sample_is_not_imputed(tmp_path):
    config = phospho_workflow(tmp_path, kinase=False)
    path = config["dataset"]["inject_runs"]["flowthrough"]["input_file"]
    frame = pd.read_csv(path, sep="\t")
    frame.loc[frame.run == "background_B_2.raw", "intensity"] = np.nan
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="No observed quantitative values.*covariate.*B_2"):
        Dataset(**config)
