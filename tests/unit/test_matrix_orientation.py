"""Feature/sample identities must survive numeric reconstruction and export."""
import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from proteoflux.export.de_exporter import DEExporter
from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Block, Preprocessor
from tests.fixtures.workflows import base_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n_features", [2, 3, 4])
@pytest.mark.parametrize("methods", [["log2"], ["log2", "median_equalization"]])
@pytest.mark.parametrize("block", [Block.MAIN, Block.COV])
def test_normalization_keeps_feature_rows_and_sample_columns(n_features, methods, block):
    samples = ["sample_z", "sample_a", "sample_m"]
    features = ["feature_z", "feature_a", "feature_m", "feature_b"][:n_features]
    # Log2 values are 11 12 13 / 21 22 23 / 31 32 33, with non-square
    # cases on both sides. Neither labels nor values are interchangeable.
    log_values = 10*np.arange(1, n_features+1)[:, None] + np.arange(1, 4)[None, :]
    raw = pd.DataFrame(np.exp2(log_values), index=pd.Index(features, name="INDEX"), columns=samples)
    frame = pl.from_pandas(raw.reset_index())
    processor = Preprocessor({"normalization": {"method": methods}})
    ir = processor.intermediate_results
    ir.set_columns_and_index(frame)
    suffix = "_covariate" if block is Block.COV else ""
    ir.add_df("raw_df"+suffix, frame)
    processor._normalize_matrix(block)

    expected = log_values.astype(float)
    if "median_equalization" in methods:
        # Sample medians differ by one; equalization shifts columns by +1,0,-1.
        expected += np.array([1., 0., -1.])
    for key, values in (("raw_df", raw.to_numpy()), ("postlog", log_values), ("normalized", expected)):
        actual = ir.dfs[key+suffix].to_pandas().set_index("INDEX")
        wanted = pd.DataFrame(values, index=raw.index, columns=samples, dtype=float)
        assert_frame_equal(actual, wanted, check_exact=True)
    assert_array_equal(ir.matrices["normalized"+suffix], expected)


@pytest.mark.parametrize("repeated_sites", [False, True])
@pytest.mark.parametrize("method", [None, "mean"])
def test_covariate_imputation_and_centering_keep_labelled_axes(repeated_sites, method):
    samples = ["sample_z", "sample_a", "sample_m"]
    parents = ["parent_z", "parent_a", "parent_m"]
    parent_values = np.array([[11., np.nan, 14.], [21., 23., 26.], [31., 34., 39.]])
    row_parents = parents * (2 if repeated_sites else 1)
    features = [f"{parent}|site{i}" for i, parent in enumerate(row_parents)]
    values = np.tile(parent_values, (2 if repeated_sites else 1, 1))
    frame = pl.DataFrame({"INDEX": features, **dict(zip(samples, values.T))})
    processor = Preprocessor({"covariates": {"enabled": True}, "imputation": {"method": method}})
    ir = processor.intermediate_results
    ir.set_columns_and_index(frame)
    ir.add_df("normalized", frame)
    ir.add_df("normalized_covariate", frame)
    ir.dfs["condition_pivot"] = pl.DataFrame({"Sample": samples, "CONDITION": ["A"]*3})
    ir.dfs["cov_index_to_key"] = pl.DataFrame({"INDEX": features, "PARENT_PROTEIN": row_parents})
    ir.metadata["covariate"]["key_col"] = "PARENT_PROTEIN"
    processor._impute_matrix()

    expected = values.copy()
    if method == "mean":
        expected[np.isnan(expected)] = 12.5
    row_medians = np.tile([12.5, 23., 34.], 2 if repeated_sites else 1)
    index = pd.Index(features, name="INDEX")
    for key, wanted in (("imputed", expected), ("imputed_covariate", expected),
                        ("centered_covariate", expected-row_medians[:, None])):
        actual = ir.dfs[key].to_pandas().set_index("INDEX")
        assert_frame_equal(actual, pd.DataFrame(wanted, index=index, columns=samples), check_exact=True)
    assert_array_equal(ir.dfs["normalized"].select(samples).to_numpy(), values)
    assert_array_equal(ir.dfs["normalized_covariate"].select(samples).to_numpy(), values)


@pytest.mark.contract
@pytest.mark.parametrize("n_features", [3, 4])
def test_labelled_normalization_survives_anndata_h5ad_and_workbook(tmp_path, n_features):
    config = base_config(tmp_path)
    config["preprocessing"]["normalization"]["method"] = ["log2"]
    config["preprocessing"]["imputation"]["method"] = None
    samples = ["sample_z", "sample_a", "sample_m"]
    features = ["P_Z", "P_A", "P_M", "P_B"][:n_features]
    log_values = 10*np.arange(1, n_features+1)[:, None] + np.arange(1, 4)[None, :]
    sequences = ["ACDEK", "FGHIR", "LMNPK", "QRSTK"]
    rows = [{"protein": feature, "accession": feature, "sequence": sequences[g],
             "intensity": 2.**log_values[g, s], "run": sample+".raw", "condition": "A",
             "replicate": s+1, "gene": feature, "charge": 2, "qvalue": .001, "pep": .001,
             "fasta": "synthetic", "n_precursors": 1}
            for g, feature in enumerate(features) for s, sample in enumerate(samples)]
    pd.DataFrame(rows).sample(frac=1, random_state=29).to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    obj = Dataset(**config).get_anndata()
    expected = pd.DataFrame(log_values.T.astype(float), index=samples, columns=features)
    assert set(obj.obs_names) == set(samples)
    assert set(obj.var_names) == set(features)
    expected = expected.loc[obj.obs_names, obj.var_names]
    for matrix in (obj.X, obj.layers["lognorm"], obj.layers["normalized"]):
        assert_array_equal(matrix, expected.to_numpy())
    assert_array_equal(obj.layers["raw"], np.exp2(expected.to_numpy()))

    exporter = DEExporter(obj, config["exports"]["path_table"], config=config)
    exporter.export_adata(config["exports"]["path_h5ad"])
    exporter.export()
    loaded = ad.read_h5ad(config["exports"]["path_h5ad"])
    assert_array_equal(loaded.obs_names, obj.obs_names)
    assert_array_equal(loaded.var_names, obj.var_names)
    for matrix in (loaded.X, loaded.layers["lognorm"], loaded.layers["normalized"]):
        assert_array_equal(matrix, expected.to_numpy())
    assert_array_equal(loaded.layers["raw"], np.exp2(expected.to_numpy()))
    workbook = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    for sample in samples:
        assert_allclose(workbook.loc[obj.var_names, "Processed_log2_Intensities"+sample], expected.loc[sample])
        assert_allclose(workbook.loc[obj.var_names, "Raw_Intensities"+sample], np.exp2(expected.loc[sample]))


def test_square_matrices_keep_sample_and_feature_labels(protein_config):
    path = protein_config["dataset"]["input_file"]
    rows = pd.read_csv(path, sep="\t")
    rows = rows[rows.protein.isin([f"P{i:03}" for i in range(9)])]
    rows.to_csv(path, sep="\t", index=False)
    obj = Dataset(**protein_config).get_anndata()
    assert obj.shape == (9, 9)
    rows = rows[(rows.qvalue <= .01) & (rows.pep <= .05)].copy()
    rows.loc[rows.intensity < 2, "intensity"] = np.nan
    rows["run"] = rows.run.str.removesuffix(".raw")
    expected = rows.groupby(["run", "protein"]).intensity.sum(min_count=1).unstack()
    expected = expected.loc[obj.obs_names, obj.var_names]
    assert_allclose(obj.layers["raw"], expected, rtol=3e-7, equal_nan=True)
    log = np.log2(expected.to_numpy())
    normalized = log + np.nanmedian(log) - np.nanmedian(log, axis=1)[:, None]
    assert_allclose(obj.layers["normalized"], normalized, rtol=3e-7, equal_nan=True)
