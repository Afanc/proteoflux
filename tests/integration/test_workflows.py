"""Real entrypoint -> preprocessing -> statistics/QC -> H5AD and XLSX."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tests.fixtures.workflows import WORKFLOWS
from tests.scientific_reference import scientific_payload, assert_reference, load_reference

pytestmark = [pytest.mark.integration, pytest.mark.regression]


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_scientific_reference(name, workflow_results):
    obj, config = workflow_results(name)
    expected = load_reference(name)
    assert_reference(scientific_payload(obj), expected["result"])
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    assert set(summary.index.astype(str)) == set(obj.var_names)


def test_robust_rollup_workflow_preserves_connected_quantities_and_component_ratios(tmp_path):
    import anndata as ad
    from proteoflux.main import run_pipeline
    from tests.fixtures.workflows import protein_workflow

    config = protein_workflow(tmp_path)
    config["preprocessing"]["protein_rollup_method"] = "robust_regression"
    rows = pd.read_csv(config["dataset"]["input_file"], sep="\t")
    # P000: one peptide in A/B and a different peptide in C, with no bridge.
    first = sorted(rows.loc[rows.protein == "P000", "sequence"].unique())[0]
    remove = (rows.protein == "P000") & ((rows.sequence == first) == (rows.condition == "C"))
    rows.loc[remove, "intensity"] = np.nan
    rows.to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    run_pipeline(config)
    result = ad.read_h5ad(config["exports"]["path_h5ad"])
    raw = pd.DataFrame(result.layers["raw"], index=result.obs_names, columns=result.var_names)

    observed = rows[(rows.qvalue <= .01) & (rows.pep <= .05)].copy()
    observed.loc[observed.intensity < 2, "intensity"] = np.nan
    observed["run"] = observed.run.str.removesuffix(".raw")
    observed["log_intensity"] = np.log2(observed.intensity)
    expected = np.exp2(observed.groupby(["run", "protein"]).log_intensity.mean().unstack())
    connected = raw.columns[raw.columns != "P000"]
    # Both peptides in every connected protein differ by exactly twofold:
    # the sum-coded model yields their geometric mean, not their linear sum.
    assert_allclose(raw[connected], expected.reindex(index=raw.index, columns=connected), rtol=3e-6, equal_nan=True)
    for sequence, component in observed[observed.protein == "P000"].groupby("sequence"):
        source = component.set_index("run").intensity.reindex(raw.index)
        keep = source.notna() & raw.P000.notna()
        assert keep.sum() >= 2
        assert_allclose(raw.P000[keep]/raw.P000[keep].iloc[0], source[keep]/source[keep].iloc[0], rtol=3e-6)
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    exported = summary.loc[raw.columns, ["Raw_Intensities"+s for s in raw.index]].to_numpy().T
    assert_allclose(exported, raw, rtol=3e-6, equal_nan=True)


@pytest.mark.parametrize("mode", ["free", "nonneg"])
def test_optional_covariate_slope_modes_can_run(tmp_path, mode):
    import anndata as ad
    from scipy.optimize import lsq_linear
    from proteoflux.main import run_pipeline
    from tests.fixtures.workflows import phospho_workflow

    config = phospho_workflow(tmp_path, kinase=False)
    config["ancova"] = {"beta_mode": mode}
    run_pipeline(config)
    result = ad.read_h5ad(config["exports"]["path_h5ad"])
    assert result.varm["log2fc"].shape[1] == 2  # original xfail assertion
    assert result.varm["covariate_beta"].shape == (result.n_vars, 1)
    assert result.varm["covariate_p"].shape == (result.n_vars, 1)
    cov = result.layers["centered_covariate"].astype(float)
    cov -= cov.mean(axis=0)
    feature = np.argmax(cov.std(axis=0))  # independently check a non-flat site
    assert cov[:, feature].std(ddof=1) > .1
    energy = (cov**2).sum(axis=0)
    ridge = 1e-6 * np.median(energy[energy > 0])
    # Ridge is an extra least-squares observation penalizing only the slope.
    design = np.vstack([np.column_stack([np.ones(result.n_obs), cov[:, feature]]), [0., np.sqrt(ridge)]])
    y = result.X[:, feature].astype(float)
    bounds = ([-np.inf, 0. if mode == "nonneg" else -np.inf], [np.inf, np.inf])
    fit = lsq_linear(design, np.r_[y, 0.], bounds=bounds)
    assert fit.success
    assert_allclose(result.varm["covariate_beta"][feature, 0], fit.x[1], rtol=3e-6, atol=2e-6)
    assert_allclose(result.layers["residuals_covariate"][:, feature], y-fit.x[0]-fit.x[1]*cov[:, feature], rtol=3e-6, atol=2e-6)
    if mode == "nonneg":
        assert (result.varm["covariate_beta"] >= 0).all()
    assert_allclose(result.varm["raw_log2fc"], result.varm["log2fc"]+result.varm["cov_part"], rtol=3e-6, atol=2e-6)
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    assert set(summary.index.astype(str)) == set(result.var_names)


def test_separate_and_reused_background_align_same_observed_values(workflow_results):
    separate, _ = workflow_results("phospho")
    reused, _ = workflow_results("phospho_reuse")
    reused = reused[separate.obs_names, separate.var_names]
    for key in ("raw_covariate", "lognorm_covariate", "normalized_covariate"):
        assert_allclose(separate.layers[key], reused.layers[key], rtol=2e-7, equal_nan=True)
    observed = np.isfinite(separate.layers["normalized_covariate"])
    assert_allclose(separate.layers["processed_covariate"][observed], reused.layers["processed_covariate"][observed])
    # The combined input changes first-seen feature order and therefore RNG
    # draw order. Missing draws and resulting statistics have separate goldens;
    # equal seeds do not promise permutation-invariant LC-ConMed imputation.


def test_fixed_one_background_adjustment_and_missing_background_fallback(workflow_results):
    obj, _ = workflow_results("phospho")
    # No batch factors: decomposition is independently an exact mean difference.
    observed = ~np.isnan(obj.layers["raw_covariate"])
    for j, contrast in enumerate(obj.uns["contrast_names"]):
        a, b = contrast.split("_vs_")
        ma, mb = obj.obs.CONDITION == a, obj.obs.CONDITION == b
        supported = observed[ma].any(0) & observed[mb].any(0)
        raw_fc = obj.X[ma].mean(0)-obj.X[mb].mean(0)
        ft_fc = obj.layers["processed_covariate"][ma].mean(0)-obj.layers["processed_covariate"][mb].mean(0)
        assert_allclose(obj.varm["log2fc"][supported, j], (raw_fc-ft_fc)[supported], atol=3e-6)
        assert_allclose(obj.varm["log2fc"][~supported, j], raw_fc[~supported], atol=3e-6)
        assert_array_equal(obj.varm["q_ebayes"][~supported, j], obj.varm["raw_q_ebayes"][~supported, j])
        assert_allclose(obj.varm["cov_part"][~supported, j], 0)


def test_phospho_without_background_matches_unadjusted_fold_changes(workflow_results):
    unadjusted, _ = workflow_results("phospho_unadjusted")
    adjusted, _ = workflow_results("phospho")
    adjusted = adjusted[unadjusted.obs_names, unadjusted.var_names]
    assert not unadjusted.uns["has_covariate"]
    assert "raw_log2fc" not in unadjusted.varm
    assert_allclose(unadjusted.X, adjusted.X, rtol=3e-7)
    assert_allclose(unadjusted.varm["log2fc"], adjusted.varm["raw_log2fc"], rtol=3e-6, atol=2e-6)


def test_pelsa_eligibility_response_direction_and_parent_support(workflow_results):
    obj, _ = workflow_results("pelsa")
    result = obj.uns["pelsa"]["curve_results"].set_index("peptide_id")
    from tests.fixtures.workflows import SEQUENCES
    assert SEQUENCES[15] not in result.index
    assert not result.loc[SEQUENCES[14], "fit_success"]
    for g in range(14):
        curve = result.loc[SEQUENCES[g]]
        assert curve.fit_success
        assert np.sign(curve.curve_fold_change_log2) == (1 if g % 2 else -1)
        # Independent generating EC50; finite pseudo-control and bounded
        # optimization are approximations, so assess scientific recovery separately.
        assert 10**(-curve.pec50) == pytest.approx(.04*1.35**g, rel=.25)
        assert curve.significant_peptides_per_parent == 2
    assert "log2fc" not in obj.varm


def test_phosx_family_fdr_and_feature_assignment_alignment(workflow_results):
    from scipy.stats import false_discovery_control
    from proteoflux.analysis.phosx_adapter import create_phosx_session
    obj, _ = workflow_results("phosx")
    block = obj.uns["kinase_activity"]
    table, assignments = block["results"], block["substrates"]
    session = create_phosx_session()
    for family in ("S/T", "Y"):
        ids = set(session.family(family).names)
        for contrast in obj.uns["contrast_names"]:
            selected = table[(table.contrast == contrast) & table.kinase_id.isin(ids) & table.tested]
            assert len(selected) >= 2, "Fixture must exercise within-family multiplicity"
            assert_allclose(selected.qvalue, np.round(false_discovery_control(selected.pvalue.to_numpy()), 3), atol=1e-12)
            j = list(obj.uns["contrast_names"]).index(contrast)
            for row in selected.itertuples():
                hits = assignments[(assignments.contrast == contrast) & (assignments.kinase_id == row.kinase_id)]
                indices = obj.var_names.get_indexer(hits.phosphosite_id.unique())
                assert (indices >= 0).all()
                assert row.n_substrates == len(indices)
                assert row.kinase_mean_log2fc == pytest.approx(obj.varm["log2fc"][indices, j].mean())


def test_real_cli_uses_cwd_paths_and_legacy_export_block(tmp_path, monkeypatch):
    import anndata as ad
    import yaml
    from typer.testing import CliRunner
    from proteoflux.cli import app
    from tests.fixtures.workflows import protein_workflow
    directory = tmp_path/"inputs"
    directory.mkdir()
    config = protein_workflow(directory)
    config["dataset"]["input_file"] = "input.tsv"
    config["analysis"]["exports"] = config.pop("exports")
    config_dir = tmp_path/"configs"
    config_dir.mkdir()
    config_path = config_dir/"analysis.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(directory)
    run = CliRunner().invoke(app, ["run", "--config", str(config_path)])
    assert run.exit_code == 0, run.output + str(run.exception)
    actual = ad.read_h5ad(config["analysis"]["exports"]["path_h5ad"])
    assert_reference(scientific_payload(actual), load_reference("proteomics")["result"])
