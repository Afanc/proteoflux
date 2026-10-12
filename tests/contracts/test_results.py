import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.stats import false_discovery_control, t as student_t

from proteoflux.export.de_exporter import DEExporter
from proteoflux.analysis.clustering import run_clustering, run_clustering_missingness
from tests.fixtures.workflows import WORKFLOWS
from tests.scientific_reference import scientific_payload, assert_reference
from tests.unit.test_statistics import expression

pytestmark = pytest.mark.contract
CONTRACT = json.loads((Path(__file__).parents[1]/"fixtures/viewer_contract.json").read_text())


def test_phospho_raw_workbook_probabilities_are_from_the_unadjusted_model(workflow_results):
    from proteoflux.analysis.limma_pipeline import run_limma_pipeline
    obj, config = workflow_results("phospho")
    # Same processed phospho values/design, explicitly without covariate fitting.
    # This checks model identity, not independent validation of InMoose itself.
    unadjusted = run_limma_pipeline(obj, {"analysis": config["analysis"]})
    assert not unadjusted.uns["has_covariate"]
    assert not np.allclose(unadjusted.varm["p_ebayes"], obj.varm["p_ebayes"])
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    for j, contrast in enumerate(obj.uns["contrast_names"]):
        for label, key in [("PVALUE", "p_ebayes"), ("QVALUE", "q_ebayes")]:
            assert_allclose(summary.loc[obj.var_names, f"RAW_{label}_{contrast}"],
                            unadjusted.varm[key][:, j], rtol=1e-12, atol=1e-14)
    assert obj.varm["raw_p_ebayes"].shape == obj.varm["raw_q_ebayes"].shape
    assert_allclose(obj.varm["raw_p_ebayes"], unadjusted.varm["p_ebayes"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", WORKFLOWS)
def test_serialized_viewer_contract(name, workflow_results):
    obj, _ = workflow_results(name)
    assert obj.X.shape == (len(obj.obs), len(obj.var))
    assert obj.obs_names.is_unique and obj.var_names.is_unique
    assert {"CONDITION", "REPLICATE", "ALIGN_KEY"} <= set(obj.obs.columns)
    assert {"raw", "lognorm", "normalized", "spectral_counts"} <= set(obj.layers)
    for layer in obj.layers.values():
        assert layer.shape == obj.shape
    if name != "pelsa":
        contrasts = obj.uns["contrast_names"]
        assert isinstance(contrasts, np.ndarray)  # existing viewer calls .tolist()
        fields = ["log2fc"] if obj.uns.get("pilot_study_mode", False) else CONTRACT["differential_varm"]
        for key in fields:
            assert obj.varm[key].shape == (obj.n_vars, len(contrasts))
        assert list(obj.uns["missingness"].index) == list(obj.var_names)
    evidence_name = "peptides" if obj.uns["preprocessing"]["analysis_type"] == "proteomics" else "precursors"
    evidence = obj.uns[evidence_name]
    assert set(CONTRACT["evidence"]) <= set(evidence)
    assert len(set(evidence["rows"])) == len(evidence["rows"])
    assert set(evidence["cols"]) == set(obj.obs_names)
    assert np.asarray(evidence["raw"]).shape == (len(evidence["rows"]), obj.n_obs)
    assert len(evidence["protein_index"]) == len(evidence["rows"])
    assert_array_equal(np.isnan(evidence["raw"]), np.isnan(evidence["centered"]))
    if obj.uns["preprocessing"]["analysis_type"] == "phospho":
        assert set(evidence["protein_index"]) <= set(obj.var_names)
        if obj.uns["has_covariate"]:
            assert {"raw_covariate", "processed_covariate", "qval_covariate", "sc_covariate"} <= set(obj.layers)
        else:
            assert not any("covariate" in name for name in obj.layers)
        assert {"ACCESSION_SITES", "KINASE_10MER", "SITE_15MER"} <= set(obj.var)
        for table in ("results", "substrates", "condition_kinases"):
            frame = obj.uns["kinase_activity"][table]
            assert isinstance(frame, pd.DataFrame)
            assert set(CONTRACT["kinase_"+table]) <= set(frame.columns)
    if name == "pelsa":
        curves = obj.uns["pelsa"]["curve_results"]
        assert set(CONTRACT["pelsa_curve_results"]) <= set(curves.columns)
        assert set(curves.peptide_id) <= set(obj.var_names)


def test_h5ad_export_does_not_mutate_analysis_object(tmp_path, workflow_results):
    obj, config = workflow_results("phospho")
    obj = obj.copy()
    before = scientific_payload(obj)
    obs, var = obj.obs.copy(deep=True), obj.var.copy(deep=True)
    path = tmp_path/"roundtrip.h5ad"
    DEExporter(obj, output_path=None, config=config).export_adata(str(path))
    assert_frame_equal(obj.obs, obs)
    assert_frame_equal(obj.var, var)
    assert_reference(scientific_payload(obj), before)
    assert_reference(scientific_payload(ad.read_h5ad(path)), before)


@pytest.mark.parametrize("evidence", ["absent", "zero", "observed"])
def test_optional_spectral_evidence_through_analysis_and_exports(protein_config, tmp_path, evidence):
    from proteoflux.analysis.limma_pipeline import run_limma_pipeline
    from proteoflux.workflow.dataset import Dataset

    # Exercise the optional AnnData analysis interface, not a claim that source
    # files may omit peptide/charge metadata required by other preprocessing QC.
    obj = Dataset(**protein_config).get_anndata()
    if evidence == "absent":
        del obj.layers["spectral_counts"]
    elif evidence == "zero":
        obj.layers["spectral_counts"][:] = 0
    expected_counts = None if evidence == "absent" else obj.layers["spectral_counts"].copy()
    result = run_limma_pipeline(obj, protein_config)
    exporter = DEExporter(result, output_path=tmp_path/"counts.xlsx", config=protein_config)
    table_path = exporter.export()
    h5ad_path = tmp_path/"counts.h5ad"
    exporter.export_adata(str(h5ad_path))
    restored = ad.read_h5ad(h5ad_path)
    with pd.ExcelFile(table_path) as workbook:
        summary = pd.read_excel(workbook, sheet_name="Summary", index_col=0)
        if evidence == "absent":
            assert "spectral_counts" not in restored.layers
            assert not {"nrsc", "nrsc_misalignment"} & set(restored.varm)
            assert "Spectral Counts" not in workbook.sheet_names
            assert not any(column.startswith("NRSC") for column in summary.columns)
        else:
            assert_array_equal(restored.layers["spectral_counts"], expected_counts)
            counts = pd.read_excel(workbook, sheet_name="Spectral Counts", index_col=0)
            assert_array_equal(counts.loc[obj.var_names, obj.obs_names].to_numpy().T, expected_counts)
            assert {"nrsc", "nrsc_misalignment"} <= set(restored.varm)
            assert any(column.startswith("NRSC") for column in summary.columns)
            if evidence == "zero":
                assert np.isnan(restored.varm["nrsc"]).all()
                assert np.isnan(restored.varm["nrsc_misalignment"]).all()
    assert_array_equal(restored.X, obj.X)
    assert_array_equal(restored.layers["raw"], obj.layers["raw"])


@pytest.mark.parametrize("name", ["phospho", "phospho_reuse", "phosx"])
def test_serialized_slope_diagnostics_are_distinct_from_ft_contrasts(name, workflow_results):
    # These objects have already passed through the real pipeline and H5AD writer.
    obj, config = workflow_results(name)
    for key in ("covariate_beta", "covariate_se", "covariate_t", "covariate_p", "covariate_q"):
        assert obj.varm[key].shape == (obj.n_vars, 1)
    p = 2 * student_t.sf(abs(obj.varm["covariate_t"][:, 0]), obj.n_obs-2)
    assert_allclose(obj.varm["covariate_p"][:, 0], p, rtol=1e-12, atol=1e-14)
    assert_allclose(obj.varm["covariate_q"][:, 0], false_discovery_control(p), rtol=1e-12, atol=1e-14)
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    for key, prefix in (("ft_p_ebayes", "FT_PVALUE_"), ("ft_q_ebayes", "FT_QVALUE_")):
        assert obj.varm[key].shape == (obj.n_vars, len(obj.uns["contrast_names"]))
        for j, contrast in enumerate(obj.uns["contrast_names"]):
            assert_allclose(summary.loc[obj.var_names, prefix+contrast], obj.varm[key][:, j], rtol=1e-12, atol=1e-14)


def test_workbook_observed_counts_and_sample_column_contract(workflow_results):
    obj, config = workflow_results("proteomics")
    frame = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0).loc[obj.var_names]
    for sample in obj.obs_names:
        assert "Processed_log2_Intensities"+sample in frame
        assert "Raw_Intensities"+sample in frame
    for contrast in obj.uns["contrast_names"]:
        a, b = contrast.split("_vs_")
        counts = [np.sum(~np.isnan(obj.layers["raw"][obj.obs.CONDITION == c]), axis=0) for c in (a, b)]
        assert_array_equal(frame["Observed_"+contrast], np.maximum(*counts))
    assert "MAX_OBSERVED" not in frame


def test_qc_capping_preserves_analysis_values_and_full_missingness():
    obj = expression(("A", "A", "B"))
    obj.layers["normalized"][0, 0] = np.nan
    before = obj.X.copy()
    normalized = obj.layers["normalized"].copy()
    out = run_clustering(obj, n_pcs=2, max_features=8, layer="normalized")
    out = run_clustering_missingness(out, max_features=5)
    assert_array_equal(out.X, before)
    assert_array_equal(out.layers["normalized"], normalized)
    assert out.layers["centered"].shape == obj.shape
    assert np.isnan(out.layers["centered"]).all(0).sum() == obj.n_vars-8
    assert out.obsm["X_pca"].shape[0] == 3
    assert "X_umap" not in out.obsm and "X_mds" not in out.obsm
    assert len(out.uns["intensity_feature_order"]) == 8


def test_pdf_viewer_selection_contract_distinguishes_observed_and_imputed():
    from proteoflux.export.pdf_report_exporter import ReportPlotter
    obj = expression(("A", "A", "B", "B", "B"))[:, :8].copy()
    obj.var["GENE_NAMES"] = obj.var_names
    obj.uns["preprocessing"] = {"analysis_type": "proteomics"}
    obj.uns["contrast_names"] = ["A_vs_B"]
    obj.varm["log2fc"] = np.zeros((8, 1))
    raw = np.ones((5, 8))
    raw[0, 1] = np.nan
    raw[2, 1] = np.nan  # one side meeting the threshold is sufficient
    raw[:2, 2] = np.nan
    raw[2:, 3] = np.nan
    raw[:, 4] = np.nan
    raw[[0, 2, 3], 5] = np.nan
    obj.layers["raw"] = raw
    missing = pd.DataFrame(0, index=obj.var_names, columns=["A", "B"])
    missing.loc["protein6", "A"] = 2  # confidence missingness also affects display
    obj.uns["missingness"] = missing
    obj.varm["nrsc_misalignment"] = np.array([0]*7+[1.6])[:, None]
    before = obj.X.copy()
    report = ReportPlotter(obj, {"dataset": {"analysis_type": "proteomics"}})
    measured, missing_a, missing_b = report._default_volcano_masks("A_vs_B", 0, {})
    assert_array_equal(measured, [True, True, False, False, False, False, False, False])
    assert_array_equal(missing_a, [False, False, True, False, False, False, True, False])
    assert_array_equal(missing_b, [False, False, False, True, False, False, False, False])
    assert_array_equal(obj.X, before)
    obj.uns["has_covariate"] = True
    obj.layers["raw_covariate"] = np.ones(obj.shape)
    obj.layers["raw_covariate"][2, 0] = np.nan  # A=2 B=2, below default max(group size)=3
    assert not report._default_volcano_masks("A_vs_B", 0, {})[0][0]
    assert report._default_volcano_masks("A_vs_B", 0, {"volcano_min_ft_per_condition": 2})[0][0]
