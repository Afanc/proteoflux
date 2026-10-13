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


@pytest.mark.parametrize("disconnected, policy", [(False, None), (True, None), (True, "raise"), (True, "component")])
def test_robust_rollup_workflow_preserves_connected_quantities_and_component_ratios(tmp_path, disconnected, policy):
    import anndata as ad
    from pathlib import Path
    from proteoflux.main import run_pipeline
    from proteoflux.workflow.rollup.robust_regression import DisconnectedObservationError, DisconnectedObservationWarning
    from tests.fixtures.workflows import protein_workflow

    config = protein_workflow(tmp_path)
    config["preprocessing"]["protein_rollup_method"] = "robust_regression"
    if policy is not None:
        config["preprocessing"]["robust_regression_disconnected_policy"] = policy
    rows = pd.read_csv(config["dataset"]["input_file"], sep="\t")
    if disconnected:
        # P000: one peptide in A/B and a different peptide in C, with no bridge.
        first = sorted(rows.loc[rows.protein == "P000", "sequence"].unique())[0]
        remove = (rows.protein == "P000") & ((rows.sequence == first) == (rows.condition == "C"))
        rows.loc[remove, "intensity"] = np.nan
        rows.to_csv(config["dataset"]["input_file"], sep="\t", index=False)
        if policy != "component":
            with pytest.raises(DisconnectedObservationError, match="protein 'P000'.*sample groups"):
                run_pipeline(config)
            assert not Path(config["exports"]["path_h5ad"]).exists()
            assert not Path(config["exports"]["path_table"]).exists()
            return
        # Only A/B share this protein's component. Restrict variance fitting too.
        config["analysis"].update(only_contrasts=["B_vs_A"], separate_contrasts=True)
        with pytest.warns(DisconnectedObservationWarning, match="protein 'P000'"):
            run_pipeline(config)
    else:
        run_pipeline(config)
    result = ad.read_h5ad(config["exports"]["path_h5ad"])
    assert result.uns["preprocessing"]["quantification"]["robust_regression_disconnected_policy"] == (policy or "raise")
    if disconnected:
        assert list(result.uns["contrast_names"]) == ["B_vs_A"]
    raw = pd.DataFrame(result.layers["raw"], index=result.obs_names, columns=result.var_names)

    observed = rows[(rows.qvalue <= .01) & (rows.pep <= .05)].copy()
    observed.loc[observed.intensity < 2, "intensity"] = np.nan
    observed["run"] = observed.run.str.removesuffix(".raw")
    observed["log_intensity"] = np.log2(observed.intensity)
    expected = np.exp2(observed.groupby(["run", "protein"]).log_intensity.mean().unstack())
    # Both peptides in every connected protein differ by exactly twofold:
    # the sum-coded model yields their geometric mean, not their linear sum.
    # Disconnected P000 has one peptide per component, hence zero peptide effect.
    assert_allclose(raw, expected.reindex(index=raw.index, columns=raw.columns), rtol=3e-6, equal_nan=True)
    for sequence, component in observed[observed.protein == "P000"].groupby("sequence"):
        source = component.set_index("run").intensity.reindex(raw.index)
        keep = source.notna() & raw.P000.notna()
        assert keep.sum() >= 2
        assert_allclose(raw.P000[keep]/raw.P000[keep].iloc[0], source[keep]/source[keep].iloc[0], rtol=3e-6)
    summary = pd.read_excel(config["exports"]["path_table"], sheet_name="Summary", index_col=0)
    exported = summary.loc[raw.columns, ["Raw_Intensities"+s for s in raw.index]].to_numpy().T
    assert_allclose(exported, raw, rtol=3e-6, equal_nan=True)


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
        from scipy.stats import false_discovery_control
        assert_array_equal(obj.varm["p_ebayes"][~supported, j], obj.varm["raw_p_ebayes"][~supported, j])
        assert_allclose(obj.varm["q_ebayes"][:, j], false_discovery_control(obj.varm["p_ebayes"][:, j]), rtol=1e-12)
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


def test_phosx_label_order_and_other_contrast_cannot_change_first_contrast(workflow_results):
    from proteoflux.analysis.phosx_pipeline import run_phosx_pipeline
    obj, config = workflow_results("phosx")  # 127 permutations, not production scale
    # This invariant concerns uniquely ranked sites. Tied ranks need a separate
    # scientific tie-breaking specification, not an assumption of invariance.
    for residue in ("S", "Y"):
        scores = obj.varm["log2fc"][obj.var.KINASE_10MER.str[5] == residue, 0]
        assert len(np.unique(scores)) == len(scores)
    changed = obj[::-1, ::-1].copy()
    changed.varm["log2fc"][:, 1] *= 5
    actual = run_phosx_pipeline(changed, config).uns["kinase_activity"]
    contrast = obj.uns["contrast_names"][0]
    for key, identifiers in [("results", ["kinase_id"]),
                             ("substrates", ["kinase_id", "phosphosite_id"])]:
        def selected(payload):
            frame = payload[key]
            return frame[frame.contrast == contrast].sort_values(identifiers).reset_index(drop=True)
        pd.testing.assert_frame_equal(selected(actual), selected(obj.uns["kinase_activity"]),
                                      rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize('phospho', [False, True])
@pytest.mark.parametrize('include_singleton', [False, True])
def test_uniform_pilot_selection_survives_pipeline_and_workbook(tmp_path, phospho, include_singleton):
    import anndata as ad
    from proteoflux.main import run_pipeline
    from tests.fixtures.workflows import protein_workflow, phospho_workflow
    config = phospho_workflow(tmp_path, kinase=False) if phospho else protein_workflow(tmp_path)
    paths = [config['dataset']['input_file']]
    if phospho:
        paths.append(config['dataset']['inject_runs']['flowthrough']['input_file'])
    for path in paths:
        rows = pd.read_csv(path, sep='\t')
        rows = rows[~((rows.condition == 'C') & (rows.replicate > 1))]
        rows.to_csv(path, sep='\t', index=False)
    contrasts = ['A_vs_B', 'A_vs_C'] if include_singleton else ['A_vs_B']
    config['analysis'].update(separate_contrasts=True, only_contrasts=contrasts)
    run_pipeline(config)
    result = ad.read_h5ad(config['exports']['path_h5ad'])
    assert result.uns['pilot_study_mode'] == include_singleton
    assert list(result.uns['contrast_names']) == contrasts
    assert list(result.uns['analysis']['pilot_contrasts']) == (contrasts if include_singleton else [])
    assert result.n_obs == 7  # Selection does not discard raw/processed samples.
    summary = pd.read_excel(config['exports']['path_table'], sheet_name='Summary', index_col=0)
    if include_singleton:
        assert not result.uns['has_covariate']
        assert not any(key in result.varm for key in ('p_raw', 'q_raw', 'p_ebayes', 'q_ebayes',
                                                     'se_raw', 't_raw', 'se_ebayes', 't_ebayes',
                                                     'covariate_adjusted', 'covariate_p', 'ft_p_ebayes'))
        assert not any('PVALUE_' in column or 'QVALUE_' in column for column in summary.columns)
        for j, contrast in enumerate(contrasts):
            a, b = contrast.split('_vs_')
            expected = result.X[result.obs.CONDITION == a].mean(0)-result.X[result.obs.CONDITION == b].mean(0)
            assert_allclose(result.varm['log2fc'][:, j], expected, atol=3e-6)
            prefix = 'RAW_log2FC_' if phospho else 'log2FC_'
            assert_allclose(summary.loc[result.var_names, prefix+contrast], expected, atol=3e-6)
    else:
        assert np.isfinite(result.varm['p_ebayes']).all()
        assert result.uns['has_covariate'] == phospho
        prefix = 'ADJUSTED_PVALUE_' if phospho else 'PVALUE_'
        assert_allclose(summary.loc[result.var_names, prefix+'A_vs_B'], result.varm['p_ebayes'][:, 0])
