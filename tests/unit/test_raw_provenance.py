"""Raw is filtered/rolled-up evidence, independent of analysis transforms."""
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.testing import assert_allclose, assert_array_equal
from polars.testing import assert_frame_equal as assert_polars_equal

from proteoflux.export.de_exporter import DEExporter
from proteoflux.workflow.dataset import Dataset
from tests.fixtures.workflows import base_config, phospho_workflow

pytestmark = pytest.mark.unit


@pytest.fixture
def raw_experiment(tmp_path):
    config = base_config(tmp_path)
    samples = ["run_z", "run_a", "run_m"]
    proteins = ["P_Z", "P_A", "P_M", "P_B"]
    sequences = ["ACDEK", "FGHIR", "LMNPK", "QRSTK"]
    totals = [[8., 16., 32.], [64., np.nan, 256.], [0., 4., 8.], [256., 512., 1024.]]
    rows = [{"protein": protein, "accession": protein, "sequence": sequences[g]+"A"*p,
             "charge": p+2, "intensity": totals[g][s]*weight, "qvalue": .001, "pep": .002,
             "run": sample+".raw", "condition": "A", "replicate": s+1,
             "gene": protein, "fasta": "synthetic", "n_precursors": 2}
            for g, protein in enumerate(proteins) for s, sample in enumerate(samples)
            for p, weight in enumerate((.25, .75))]
    pd.DataFrame(rows).sample(frac=1, random_state=7).to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    # Two precursors per protein. Threshold 2 censors both zeros, and 1 of
    # the 1+3 pair, so that quantity is 3. Raw is not the original vendor table.
    expected = pd.DataFrame([[8., 16., 32.], [64., np.nan, 256.],
                             [np.nan, 3., 8.], [256., 512., 1024.]],
                            index=proteins, columns=samples).T
    return config, expected


@pytest.mark.parametrize("method", ["log2", "LoG2", "log10", "LOG10"])
def test_yaml_scalar_and_list_logs_preserve_raw_evidence(raw_experiment, method):
    config, expected = raw_experiment
    datasets = []
    for setting in (method, [method]):
        config["preprocessing"]["normalization"]["method"] = setting
        config["preprocessing"]["imputation"]["method"] = None
        # The CLI uses safe_load; neither YAML nor Dataset should reinterpret a scalar.
        dataset = Dataset(**yaml.safe_load(yaml.safe_dump(config)))
        datasets.append(dataset)
        obj = dataset.get_anndata()
        raw = expected.loc[obj.obs_names, obj.var_names].to_numpy()
        assert_array_equal(obj.layers["raw"], raw)
        log = np.log2 if method.lower() == "log2" else np.log10
        assert_allclose(obj.X, log(raw), rtol=3e-7, equal_nan=True)
        ir = dataset.preprocessor.intermediate_results
        assert_array_equal(ir.matrices["raw_mat"], raw.T)
        assert_polars_equal(dataset.preprocessed_data.filtered, ir.dfs["raw_df"])
    assert_polars_equal(datasets[0].rawinput, datasets[1].rawinput)
    assert_array_equal(datasets[0].get_anndata().uns["peptides"]["raw"],
                       datasets[1].get_anndata().uns["peptides"]["raw"])


@pytest.mark.parametrize("methods", ["log2", ["log2"], ["log2", "median_equalization"]])
@pytest.mark.parametrize("imputation", [None, "mean", "lc_conmed"])
def test_analysis_steps_preserve_raw_values_missingness_and_evidence(raw_experiment, methods, imputation):
    config, expected = raw_experiment
    path = Path(config["dataset"]["input_file"])
    input_bytes = path.read_bytes()
    control = Dataset(**config)
    before_input = control.rawinput.clone()
    before_raw = control.preprocessed_data.filtered.clone()
    before_evidence = deepcopy(control.get_anndata().uns["peptides"])
    config["preprocessing"]["normalization"]["method"] = methods
    config["preprocessing"]["imputation"]["method"] = imputation
    dataset = Dataset(**config)
    obj = dataset.get_anndata()
    raw = expected.loc[obj.obs_names, obj.var_names].to_numpy()
    assert_array_equal(obj.layers["raw"], raw)
    assert_polars_equal(dataset.rawinput, before_input)
    assert_polars_equal(dataset.preprocessed_data.filtered, before_raw)
    for key, value in before_evidence.items():
        assert_array_equal(obj.uns["peptides"][key], value)
    assert path.read_bytes() == input_bytes
    # Imputation changes analysis values, never the observation mask in raw.
    if imputation is None:
        assert_array_equal(np.isnan(obj.X), np.isnan(raw))
    else:
        assert np.isfinite(obj.X).all()
    # A later analysis consumer may modify these buffers. They must not alias raw.
    obj.X[:] = -99
    obj.layers["normalized"][:] = -88
    assert_array_equal(obj.layers["raw"], raw)
    assert_polars_equal(dataset.preprocessed_data.filtered, before_raw)
    assert_array_equal(obj.uns["peptides"]["raw"], before_evidence["raw"])


@pytest.mark.contract
@pytest.mark.parametrize("phospho", [False, True])
def test_scalar_log_raw_provenance_survives_h5ad_and_workbook(raw_experiment, tmp_path, phospho):
    config, expected = raw_experiment
    if phospho:
        config = phospho_workflow(tmp_path, kinase=False)
        rows = pd.read_csv(config["dataset"]["input_file"], sep="\t")
        rows["sample"] = rows.run.str.removesuffix(".raw")
        rows["feature"] = rows.accession+"|"+rows.locations
        expected = rows.pivot(index="sample", columns="feature", values="intensity")
    config["preprocessing"]["normalization"]["method"] = "log2"
    dataset = Dataset(**config)
    obj = dataset.get_anndata()
    raw = expected.loc[obj.obs_names, obj.var_names].to_numpy()
    assert_allclose(obj.layers["raw"], raw, rtol=3e-7, equal_nan=True)
    if phospho:
        rows = pd.read_csv(config["dataset"]["inject_runs"]["flowthrough"]["input_file"], sep="\t")
        rows["sample"] = rows.condition+"_"+rows.replicate.astype(str)
        background = rows.pivot(index="sample", columns="protein", values="intensity")
        ft = np.column_stack([background.loc[obj.obs_names, feature.split("|")[0]] for feature in obj.var_names])
        assert_allclose(obj.layers["raw_covariate"], ft, rtol=3e-7, equal_nan=True)
        assert_allclose(dataset.preprocessor.intermediate_results.matrices["raw_mat_covariate"], ft.T, rtol=3e-7, equal_nan=True)
    evidence_name = "precursors" if phospho else "peptides"
    evidence = deepcopy(obj.uns[evidence_name])
    exporter = DEExporter(obj, config["exports"]["path_table"], config=config)
    exporter.export_adata(config["exports"]["path_h5ad"])
    exporter.export()
    loaded = ad.read_h5ad(config["exports"]["path_h5ad"])
    assert_array_equal(loaded.obs_names, obj.obs_names)
    assert_array_equal(loaded.var_names, obj.var_names)
    assert_allclose(loaded.layers["raw"], raw, rtol=3e-7, equal_nan=True)
    assert_allclose(obj.layers["raw"], raw, rtol=3e-7, equal_nan=True)
    for key, values in evidence.items():
        assert_array_equal(loaded.uns[evidence_name][key], values)
        assert_array_equal(obj.uns[evidence_name][key], values)
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    for s, sample in enumerate(obj.obs_names):
        assert_allclose(summary.loc[obj.var_names, "Raw_Intensities"+sample], raw[s], rtol=3e-7, equal_nan=True)
        if phospho:
            assert_allclose(summary.loc[obj.var_names, "FT_Raw_"+sample], ft[s], rtol=3e-7, equal_nan=True)
    if phospho:
        assert_allclose(loaded.layers["raw_covariate"], ft, rtol=3e-7, equal_nan=True)


def test_scalar_and_list_log2_preserve_same_raw_scale(protein_config):
    protein_config["preprocessing"]["normalization"]["method"] = ["log2"]
    expected = Dataset(**protein_config).get_anndata()
    protein_config["preprocessing"]["normalization"]["method"] = "log2"
    actual = Dataset(**protein_config).get_anndata()
    assert_allclose(actual.layers["raw"], expected.layers["raw"], equal_nan=True)


@pytest.mark.parametrize("method", ["none", "median_equalization"])
def test_prelogged_background_cannot_overwrite_main_raw_or_exports(tmp_path, method):
    config = phospho_workflow(tmp_path, kinase=False)
    config["preprocessing"]["normalization"]["method"] = method
    config["preprocessing"]["imputation"]["method"] = "mean"
    originals = {}
    for name, path in [("main", config["dataset"]["input_file"]),
                       ("background", config["dataset"]["inject_runs"]["flowthrough"]["input_file"])]:
        frame = pd.read_csv(path, sep="\t")
        originals[name] = frame.copy()
        frame["intensity"] = np.log2(frame.intensity)
        frame.to_csv(path, sep="\t", index=False)
    obj = Dataset(**config).get_anndata()
    main = originals["main"]
    main["sample"] = main.run.str.removesuffix(".raw")
    main["feature"] = main.accession + "|" + main.locations
    expected = main.pivot(index="sample", columns="feature", values="intensity").loc[obj.obs_names, obj.var_names]
    bg = originals["background"]
    bg["sample"] = bg.condition + "_" + bg.replicate.astype(str)
    bg = bg.pivot(index="sample", columns="protein", values="intensity")
    expected_bg = np.column_stack([bg.loc[obj.obs_names, feature.split("|")[0]] for feature in obj.var_names])
    # Prelogged-input convention: raw quantities are reconstructed in linear units
    # separately for each assay; neither assay supplies the other's values/mask.
    assert_allclose(obj.layers["raw"], expected, rtol=3e-7, equal_nan=True)
    assert_allclose(obj.layers["raw_covariate"], expected_bg, rtol=3e-7, equal_nan=True)
    exporter = DEExporter(obj, config["exports"]["path_table"], config=config)
    exporter.export_adata(config["exports"]["path_h5ad"])
    exporter.export()
    loaded = ad.read_h5ad(config["exports"]["path_h5ad"])
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    assert_allclose(loaded.layers["raw"], expected, rtol=3e-7, equal_nan=True)
    assert_allclose(loaded.layers["raw_covariate"], expected_bg, rtol=3e-7, equal_nan=True)
    for j, sample in enumerate(obj.obs_names):
        assert_allclose(summary.loc[obj.var_names, "Raw_Intensities"+sample], expected.iloc[j], rtol=3e-7, equal_nan=True)
        assert_allclose(summary.loc[obj.var_names, "FT_Raw_"+sample], expected_bg[j], rtol=3e-7, equal_nan=True)
