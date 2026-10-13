"""Protect scale, aggregation, censoring and evidence semantics."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.workflow.rollup.robust_regression import pivot_df_robust_regression
from proteoflux.workflow.normalizers.regression_normalization import regression_normalization
from tests.fixtures.workflows import base_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("regression_type", ["linear", "loess"])
@pytest.mark.parametrize("scale", ["global", "local"])
def test_regression_normalization_recovers_known_sample_offsets(scale, regression_type):
    feature_signal = np.linspace(10., 30., 20)[:, None]
    values = feature_signal + [-2., 1., 7.]
    normalized, _ = regression_normalization(values, scale=scale,
        regression_type=regression_type, condition_labels=["A", "A", "B"])
    # Global reference is the row MEAN (offset 2), not median (offset 1).
    # Local A uses its first sample (-2); singleton B is its own reference (7).
    expected = feature_signal + ([2., 2., 2.] if scale == "global" else [-2., -2., 7.])
    assert_allclose(normalized, expected, rtol=0, atol=1e-12)
    repeated, _ = regression_normalization(normalized, scale=scale,
        regression_type=regression_type, condition_labels=["A", "A", "B"])
    assert_allclose(repeated, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", ["sum", "mean", "median", "min", "max"])
def test_all_null_groups_remain_missing_and_zero_remains_zero(method):
    frame = pl.DataFrame({"INDEX": ["null", "null", "zero", "real", "real"],
                          "FILENAME": ["a"]*5, "SIGNAL": [None, None, 0., 2., 4.]})
    result = Preprocessor()._pivot_df(frame, "FILENAME", "INDEX", "SIGNAL", method).to_pandas().set_index("INDEX")
    assert np.isnan(result.loc["null", "a"])
    assert result.loc["zero", "a"] == 0
    assert result.loc["real", "a"] == {"sum": 6, "mean": 3, "median": 3, "min": 2, "max": 4}[method]


def test_top3_uses_dense_ties_and_sums_precursors_first():
    frame = pl.DataFrame({"INDEX": ["P"]*6, "FILENAME": ["a"]*6,
                          "PEPTIDE_LSEQ": ["AAA", "AAA", "BBB", "CCC", "DDD", "EEE"],
                          "SIGNAL": [6., 4., 8., 6., 6., 1.]})
    # Three ranks (10,8,6), four peptides; peptide_rollup=mean must not change Top3.
    result = Preprocessor({"peptide_rollup_method": "mean"})._pivot_df_top_n(frame, "FILENAME", "INDEX", "SIGNAL")
    assert result["a"].item() == 30


def test_robust_rollup_has_sum_coded_peptide_effects_and_minimum_evidence():
    peptides = pl.DataFrame({"protein": ["P"]*3, "peptide": ["a", "b", "c"],
                             "A": [4., 8., 16.], "B": [16., 32., 64.],
                             "C": [None, 128., None]})
    result = pivot_df_robust_regression(pep_wide=peptides, protein_col="protein", peptide_col="peptide",
                                      sample_cols=["B", "A", "C"], num_cores=1, huber_t=1.345,
                                      max_iter=20, tol=1e-8, min_nonan=2)
    assert_allclose(result.select("B", "A").to_numpy(), [[32, 8]], rtol=1e-9)
    assert result["C"].item() is None


def test_directlfq_preserves_scaling_and_missing_sample_mask():
    rows = [{"INDEX": "P", "PEPTIDE_LSEQ": peptide, "FILENAME": sample, "SIGNAL": value}
            for peptide, values in [("PEPA", [4., 8., None]), ("PEPB", [8., 16., None])]
            for sample, value in zip(["A", "B", "C"], values)]
    frame = pl.DataFrame(rows)
    processor = Preprocessor({"directlfq_cores": 1, "directlfq_min_nonan": 1})
    result = processor._pivot_df_LFQ(frame, "FILENAME", "INDEX", "SIGNAL", "PEPTIDE_LSEQ")
    assert_allclose(result.select("A", "B").to_numpy(), [[12, 24]], rtol=1e-6)
    assert result["C"].item() is None


@pytest.mark.parametrize("method", ["directlfq", "robust_regression"])
def test_connected_rollup_is_equivariant_to_scaling_and_label_permutation(method):
    # Exact multiplicative peptide responses: geometric mean 8, linear sum 28.
    rows = [{"INDEX": "P", "PEPTIDE_LSEQ": p, "FILENAME": s, "SIGNAL": v*m}
            for p, v in [("AAAK", 4.), ("CCCK", 8.), ("DDDK", 16.)]
            for s, m in [("z", 1.), ("a", 4.), ("m", 2.)]]
    frame = pl.DataFrame(rows)

    def quantify(frame, samples):
        if method == "directlfq":
            return Preprocessor({"directlfq_cores": 1})._pivot_df_LFQ(
                frame, "FILENAME", "INDEX", "SIGNAL", "PEPTIDE_LSEQ")
        wide = frame.pivot(index=["INDEX", "PEPTIDE_LSEQ"], on="FILENAME", values="SIGNAL")
        return pivot_df_robust_regression(pep_wide=wide, protein_col="INDEX",
            peptide_col="PEPTIDE_LSEQ", sample_cols=samples, num_cores=1,
            huber_t=1.345, max_iter=20, tol=1e-8, min_nonan=1)

    baseline = quantify(frame, ["z", "a", "m"])
    changed = quantify(frame.reverse().with_columns(pl.col("SIGNAL")*13), ["m", "z", "a"])
    expected = (28. if method == "directlfq" else 8.) * np.array([[1., 4., 2.]])
    assert baseline["INDEX"].to_list() == changed["INDEX"].to_list() == ["P"]
    assert_allclose(baseline.select("z", "a", "m"), expected, rtol=1e-6)
    assert_allclose(changed.select("z", "a", "m"), 13*expected, rtol=1e-6)


def test_dataset_stages_and_identification_are_label_aligned(protein_config):
    data = Dataset(**protein_config).get_anndata()
    rows = pd.read_csv(protein_config["dataset"]["input_file"], sep="\t")
    rows = rows[(rows.qvalue <= .01) & (rows.pep <= .05)].copy()
    rows["run"] = rows.run.str.removesuffix(".raw")
    rows.loc[rows.intensity < 2, "intensity"] = np.nan
    expected = rows.groupby(["run", "protein"]).intensity.sum(min_count=1).unstack()
    expected = expected.reindex(index=data.obs_names, columns=data.var_names)
    assert set(data.var_names) == {f"P{i:03}" for i in range(12)}
    assert data.shape == (9, 12)
    assert_allclose(data.layers["raw"], expected, rtol=2e-7, equal_nan=True)
    log = np.log2(expected.to_numpy())
    assert_allclose(data.layers["lognorm"], log, rtol=2e-7, equal_nan=True)
    norm = log + np.nanmedian(log) - np.nanmedian(log, axis=1)[:, None]
    assert_allclose(data.layers["normalized"], norm, rtol=2e-7, equal_nan=True)
    observed = np.isfinite(norm)
    assert_allclose(data.X[observed], norm[observed], rtol=2e-7)
    assert np.isfinite(data.X).all()
    s, f = data.obs_names.get_loc("A_1"), data.var_names.get_loc("P003")
    assert np.isnan(data.layers["raw"][s, f])
    assert data.layers["qvalue"][s, f] == pytest.approx(.001)
    assert data.layers["spectral_counts"][s, f] == 0
    assert "imputed_only" not in data.layers


@pytest.mark.parametrize("method", ["median_equalization_intersection", "median_equalization_by_tag"])
def test_shared_observed_reference_controls_normalization(tmp_path, method):
    config = base_config(tmp_path)
    config["preprocessing"]["normalization"] = {"method": ["log2", method], "reference_tag": ["REF"]}
    config["preprocessing"]["imputation"]["method"] = None
    rows = []
    for protein, values, tag in [("P", [16., 64., 256.], "REF"),
                                  ("Q", [4., 16., 64.], "ref"),
                                  ("R", [8., 32., 128.], "REF"),
                                  ("sparse", [1024., None, 1024.], "REF")]:
        for r, value in enumerate(values):
            rows.append({"protein": protein, "accession": protein, "sequence": protein.upper()+"K",
                         "intensity": value, "run": f"A_{r+1}", "condition": "A", "replicate": r+1,
                         "fasta": tag, "gene": protein, "charge": 2, "qvalue": .001,
                         "pep": .001, "n_precursors": 1})
    pd.DataFrame(rows).to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    data = Dataset(**config).get_anndata()
    values = pd.DataFrame(data.layers["normalized"], index=data.obs_names, columns=data.var_names)
    assert_allclose(values["P"], 6)
    assert_allclose(values["Q"], 4)
    assert_allclose(values["sparse"], [12, np.nan, 8], equal_nan=True)
    # A large change outside the complete reference intersection cannot move
    # the reference proteins, even after changing input row/sample order.
    changed = pd.DataFrame(rows).iloc[::-1].copy()
    changed.loc[changed.protein == "sparse", "intensity"] *= 1024
    changed.to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    result = Dataset(**config).get_anndata()
    aligned = result[data.obs_names, data.var_names]
    for feature in ("P", "Q", "R"):
        j = data.var_names.get_loc(feature)
        assert_allclose(aligned.layers["normalized"][:, j], data.layers["normalized"][:, j])


def test_peptide_evidence_centering_divides_linear_values(protein_config):
    data = Dataset(**protein_config).get_anndata()
    evidence = data.uns["peptides"]
    raw = np.asarray(evidence["raw"])
    assert_allclose(evidence["centered"], raw/np.nanmean(raw, axis=1)[:, None], rtol=3e-7, equal_nan=True)
    assert set(evidence["cols"]) == set(data.obs_names)
    assert set(evidence["protein_index"]) <= set(data.var_names)
