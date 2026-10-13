"""Small counterexamples for the public quantitative-input policy."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_array_equal

from proteoflux.workflow.dataset import Dataset
from proteoflux.workflow.preprocessing import Preprocessor
from proteoflux.workflow.imputer_factory import get_imputer
from tests.fixtures.workflows import base_config

pytestmark = pytest.mark.contract


@pytest.mark.parametrize('method', ['log10', ['log2', 'quantile'], ['median_equalization', 'log2'], ['log2', 'log2']])
def test_non_log2_quantitative_paths_are_rejected(method):
    with pytest.raises(ValueError, match='log2'):
        Preprocessor({'normalization': {'method': method}})


def test_removed_imputer_name_has_a_clear_error():
    with pytest.raises(ValueError, match='iterative_hgb'):
        get_imputer(method='randomforest')


@pytest.mark.parametrize('method', ['sum', 'mean', 'median', 'top3', 'directlfq', 'robust_regression', 'count'])
def test_prelogged_input_cannot_be_summed_or_logged_again(method):
    pre = Preprocessor({'protein_rollup_method': method, 'normalization': {'method': 'none'}})
    df = pl.DataFrame({'INDEX': ['P', 'P'], 'FILENAME': ['run', 'run'],
                       'SIGNAL': [2., 3.], 'PEPTIDE_LSEQ': ['ACDK', 'FGHK'], 'CHARGE': [2, 2]})
    # Linear sum would be 4+8=12 (log2=3.585...), not 2+3=5.
    with pytest.raises(ValueError, match='linear.*input|prelogged'):
        pre._pivot_main_block(df)


@pytest.mark.parametrize('method', [None, 'knn', 'iterative_hgb', 'mean', 'median', 'lc_conmed'])
def test_empty_quantitative_sample_is_rejected_before_imputation(method):
    pre = Preprocessor({'imputation': {'method': method}})
    frame = pl.DataFrame({'INDEX': ['Pz', 'Pa', 'Pm'], 's1': [11., 21., 31.],
                         'empty_run': [None]*3, 's3': [13., 23., 33.]})
    ir = pre.intermediate_results
    ir.set_columns_and_index(frame)
    ir.add_df('normalized', frame)
    ir.dfs['condition_pivot'] = pl.DataFrame({'Sample': ['s1', 'empty_run', 's3'], 'CONDITION': ['A']*3})
    with pytest.raises(ValueError, match='empty_run'):
        pre._impute_matrix()
    assert 'imputed' not in ir.matrices


@pytest.mark.parametrize('field,value', [('charge', 0), ('charge', -2), ('charge', 2.5),
                                        ('charge', None), ('sequence', None), ('sequence', ''),
                                        ('sequence', 'ACD123'), ('sequence', 'ACD[broken')])
def test_invalid_precursor_values_fail_before_quantification(tmp_path, field, value):
    config = base_config(tmp_path)
    row = {'protein': 'P1', 'accession': 'P1', 'run': 'A_1', 'condition': 'A',
           'replicate': 1, 'sequence': 'ACDK', 'charge': 2, 'intensity': 100.}
    row[field] = value
    pd.DataFrame([row]).to_csv(config['dataset']['input_file'], sep='\t', index=False)
    with pytest.raises(ValueError, match='PEPTIDE_LSEQ|CHARGE'):
        Dataset(**config)


@pytest.mark.parametrize('removed_by_filter', [False, True])
def test_pipeline_cannot_drop_or_fill_an_empty_sample(protein_config, removed_by_filter):
    path = protein_config['dataset']['input_file']
    frame = pd.read_csv(path, sep='\t')
    sample = 'B_2.raw'
    frame.loc[frame.run == sample, 'qvalue' if removed_by_filter else 'intensity'] = (.9 if removed_by_filter else np.nan)
    frame.to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError, match='B_2'):
        Dataset(**protein_config)


def test_prelogged_single_measurements_preserve_linear_evidence_and_zero_log_signal(tmp_path):
    from tests.fixtures.workflows import protein_rows
    config = base_config(tmp_path)
    frame = protein_rows()
    frame = frame[frame.charge == 2].copy()  # one precursor per protein/sample
    frame['intensity'] = 1.  # linear one is observed log2 zero, not absent evidence
    frame.to_csv(config['dataset']['input_file'], sep='\t', index=False)
    config['preprocessing']['filtering']['min_linear_intensity'] = None
    linear = Dataset(**config).get_anndata()
    frame['intensity'] = 0.
    frame.to_csv(config['dataset']['input_file'], sep='\t', index=False)
    config['preprocessing']['normalization']['method'] = 'none'
    prelogged = Dataset(**config).get_anndata()
    assert_array_equal(prelogged.obs_names, linear.obs_names)
    assert_array_equal(prelogged.var_names, linear.var_names)
    for key in ('raw', 'spectral_counts'):
        assert_array_equal(prelogged.layers[key], linear.layers[key])
    for key in ('raw', 'centered'):
        assert_array_equal(prelogged.uns['peptides'][key], linear.uns['peptides'][key])
    assert_array_equal(prelogged.X, linear.X)


@pytest.mark.parametrize('analysis', ['limma', 'pelsa'])
def test_inference_rejects_declared_non_log2_external_anndata(analysis):
    from proteoflux.analysis.limma_pipeline import run_limma_pipeline
    from proteoflux.analysis.pelsa_pipeline import run_pelsa_pipeline
    from tests.unit.test_statistics import expression
    run = run_limma_pipeline if analysis == 'limma' else run_pelsa_pipeline
    obj = expression()
    with pytest.raises(ValueError, match='log2'):
        run(obj, {'preprocessing': {'normalization': {'method': 'log10'}}})
    obj.uns['preprocessing'] = {'quantitative_scale': 'linear'}
    with pytest.raises(ValueError, match='log2'):
        run(obj, {})


def test_serialized_log2_metadata_can_be_validated_without_retransforming(tmp_path):
    import anndata as ad
    from numpy.testing import assert_allclose
    from proteoflux.analysis.limma_pipeline import run_limma_pipeline
    from tests.unit.test_statistics import expression
    obj = expression()
    obj.uns['preprocessing'] = {'normalization': {'method': ['log2', 'median_equalization']},
                                'quantitative_scale': 'log2'}
    expected = run_limma_pipeline(obj, {})
    path = tmp_path/'log2-input.h5ad'
    obj.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    actual = run_limma_pipeline(loaded, {})
    assert_array_equal(actual.X, obj.X)
    assert_array_equal(actual.layers['raw'], obj.layers['raw'])
    for key in expected.varm:
        assert_allclose(actual.varm[key], expected.varm[key], rtol=1e-12)
