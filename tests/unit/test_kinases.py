"""Independent KSEA arithmetic and dense references for PhosX optimizations."""
import math
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from proteoflux.analysis.ksea_pipeline import run_ksea_pipeline
from proteoflux.analysis.phosx_adapter import (
    PhosXSession, _FamilyResources, _observed_ks, _null_ks_from_positions,
    _fast_empirical_distribution,
)

pytestmark = pytest.mark.unit


def test_ksea_background_deduplication_support_and_per_contrast_fdr(tmp_path):
    obj = ad.AnnData(np.ones((4, 6)), obs=pd.DataFrame({"CONDITION": ["A", "A", "B", "B"]},
                                                     index=["a1", "a2", "b1", "b2"]),
                     var=pd.DataFrame({"ACCESSION_SITES": [f"P{i}:S1" for i in range(6)],
                                       "GENE_NAMES": [f"G{i}" for i in range(6)]},
                                      index=[f"P{i}|S1" for i in range(6)]))
    obj.layers["raw"] = obj.X.copy()
    obj.uns["contrast_names"] = ["A_vs_B", "B_vs_A"]
    obj.varm["log2fc"] = np.column_stack([np.arange(-2., 4.), -np.arange(-2., 4.)])
    obj.varm["raw_log2fc"] = np.zeros((6, 2))  # must not override standard/adjusted FC
    obj.varm["q_ebayes"] = np.ones((6, 2))  # KSEA background is not DE-significance-filtered
    rows = [("K_A", f"P{i}", "S1") for i in (3, 4, 5, 5)]
    rows += [("K_B", f"P{i}", "S1") for i in (0, 1)]
    rows += [("K_C", f"P{i}", "S1") for i in (0, 1, 2, 3)]
    db = tmp_path/"substrates.tsv"
    database = pd.DataFrame(rows, columns=["kinase", "protein", "site"])
    database["gene"] = ""
    database = pd.concat([database, pd.DataFrame([
        {"kinase": "K_A", "protein": "WRONG_ACCESSION", "site": "S1", "gene": "G0"},
        {"kinase": "K_B", "protein": "", "site": "S1", "gene": "G1"},
    ])], ignore_index=True)
    database.to_csv(db, sep="\t", index=False)
    config = {"analysis": {"kinase_activity": {"method": "ksea", "substrate_database": str(db),
               "min_substrates": 3, "database_columns": {"kinase_gene": "kinase",
               "substrate_uniprot": "protein", "substrate_gene": "gene", "substrate_site": "site"}}}}
    result = run_ksea_pipeline(obj, config).uns["kinase_activity"]
    table = result["results"].set_index(["contrast", "kinase_id"])
    z = 1.5*math.sqrt(3)/math.sqrt(3.5)  # mean(top3)=2; mean(all6)=.5; sample variance=3.5
    p = math.erfc(z/math.sqrt(2))
    z_c = -2/math.sqrt(3.5)  # mean(bottom4)=-.5 versus background mean=.5
    p_c = math.erfc(abs(z_c)/math.sqrt(2))
    for contrast, sign in [("A_vs_B", 1), ("B_vs_A", -1)]:
        tested = table.loc[(contrast, "K_A")]
        assert tested.n_substrates == 3
        assert tested.global_mean_log2fc == pytest.approx(sign*.5)
        assert tested.z_score == pytest.approx(sign*z)
        assert tested.pvalue == pytest.approx(p)
        assert tested.qvalue == pytest.approx(min(2*p, p_c))
        second = table.loc[(contrast, "K_C")]
        assert second.n_substrates == 4
        assert second.z_score == pytest.approx(sign*z_c)
        assert second.pvalue == pytest.approx(p_c)
        assert second.qvalue == pytest.approx(p_c)
        untested = table.loc[(contrast, "K_B")]
        assert not untested.tested and np.isnan(untested.pvalue)
    assignments = result["substrates"]
    assert not assignments.duplicated(["contrast", "kinase_id", "phosphosite_id"]).any()


def dense_ks(positions, weights):
    """Independent O(n) reference, not the optimized hit-position algorithm."""
    hit = np.zeros(len(weights), dtype=bool)
    hit[list(positions)] = True
    steps = np.where(hit, weights/weights[hit].sum(), -1/(~hit).sum())
    walk = np.r_[0., np.cumsum(steps)]
    low, high = walk.min(), walk.max()
    return low if np.isclose(abs(low), abs(high), rtol=1e-12, atol=1e-15) or abs(low) > abs(high) else high


def test_sparse_phosx_ks_matches_dense_walk_for_all_small_hit_sets():
    weights = np.array([5., 2., 1., .5, 3., 4.])
    for count in range(1, len(weights)):
        for positions in combinations(range(len(weights)), count):
            actual = _null_ks_from_positions(np.array(positions), weights, np.arange(count), len(weights)-count)
            assert actual == pytest.approx(dense_ks(positions, weights), abs=2e-15)
    assert _null_ks_from_positions(np.array([1]), np.ones(3), np.arange(1), 2) == -.5


def test_phosx_observed_ranking_uses_absolute_weights():
    assignments = pd.DataFrame({"K1": [1, 0, 1, 0], "K2": [0, 1, 0, 1]})
    scores = pd.Series([3., 1., -2., -4.])
    actual = _observed_ks(assignments, scores)
    expected = [dense_ks([0, 2], abs(scores.to_numpy())), dense_ks([1, 3], abs(scores.to_numpy()))]
    assert_allclose(actual, expected, atol=1e-15)


def test_phosx_pssm_unknown_residues_and_right_inclusive_quantiles():
    family = _FamilyResources(names=("K",), amino_acids=("A", "S"),
                              pssm=np.array([[[2., 3.], [5., 7.]]]),
                              sorted_background=(np.array([1., 5., 7., 7., 14.]),))
    raw = PhosXSession._score_windows(["AS", "_S", "__"], family, take_max=False)
    assert_array_equal(raw[:, 0], [14, 7, 1])
    assert_allclose(PhosXSession._quantile_scale(raw, family)[:, 0], [1, .8, .2])
    with pytest.raises(ValueError, match="non-canonical"):
        PhosXSession._score_windows(["XS"], family, take_max=False)


def test_phosx_seeded_permutations_match_independent_dense_reference():
    assignments = pd.DataFrame({"K1": [1, 0, 1, 0, 0], "K2": [0, 1, 0, 1, 0]})
    scores = pd.Series([4., 3., 1., -2., -5.])
    actual = _fast_empirical_distribution(assignments, scores, n_permutations=23,
                                          n_processes=1, random_seed=47, family_label="test")
    expected = []
    for seed in np.random.SeedSequence(47).generate_state(23, dtype=np.uint32):
        order = np.random.RandomState(int(seed)).choice(5, size=5, replace=False)
        expected.append([dense_ks(np.flatnonzero(assignments[col].to_numpy()[order]), abs(scores.to_numpy()))
                         for col in assignments])
    assert_allclose(actual, expected, atol=2e-15)
