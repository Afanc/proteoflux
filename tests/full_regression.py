"""Label-aware comparison for explicit local runs, without giant JSON payloads."""
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from tests.scientific_reference import portable, assert_reference, probability_field


def compare_full_results(actual, expected, *, rtol=3e-6, atol=2e-6):
    assert actual.obs_names.is_unique and actual.var_names.is_unique
    assert expected.obs_names.is_unique and expected.var_names.is_unique
    assert set(actual.obs_names) == set(expected.obs_names), "sample identifiers"
    assert set(actual.var_names) == set(expected.var_names), "feature identifiers"
    feature_order = actual.var_names.get_indexer(expected.var_names)
    original = actual
    actual = actual[expected.obs_names, expected.var_names]
    for name in ("obs", "var"):
        assert_frame_equal(getattr(actual, name).sort_index(axis=1), getattr(expected, name).sort_index(axis=1),
                           check_dtype=False, check_categorical=False, rtol=rtol, atol=atol)
    assert_reference(portable(actual.uns.get("contrast_names", [])), portable(expected.uns.get("contrast_names", [])))
    assert_reference(portable(actual.uns.get("analysis", {})), portable(expected.uns.get("analysis", {})))
    arrays = {"X": (actual.X, expected.X)}
    for container, exclude in (("layers", {"centered"}), ("varm", {"PCs", "covariate_p"})):
        left, right = getattr(actual, container), getattr(expected, container)
        assert set(left)-exclude == set(right)-exclude, container+" keys"
        for name in set(left)-exclude:
            arrays[f"{container}.{name}"] = (left[name], right[name])
    for name, (left, right) in arrays.items():
        assert left.shape == right.shape, name
        probability = probability_field(name.split(".")[-1])
        for start in range(0, left.shape[0], 2048):
            assert_allclose(left[start:start+2048], right[start:start+2048],
                            rtol=3e-5 if probability else rtol, atol=1e-12 if probability else atol,
                            equal_nan=True, err_msg=name)
    if "residual_variance" in expected.uns:
        assert_allclose(np.asarray(original.uns["residual_variance"])[feature_order],
                        expected.uns["residual_variance"], rtol=rtol, atol=atol, equal_nan=True)
    if "missingness" in expected.uns:
        assert_frame_equal(actual.uns["missingness"].reindex(expected.uns["missingness"].index).sort_index(axis=1),
                           expected.uns["missingness"].sort_index(axis=1))
    for name in ("peptides", "precursors"):
        assert (name in actual.uns) == (name in expected.uns), name
        if name not in expected.uns:
            continue
        left, right = actual.uns[name], expected.uns[name]
        assert set(left) == set(right), name+" fields"
        rows, cols = pd.Index(left["rows"]), pd.Index(left["cols"])
        assert rows.is_unique and cols.is_unique
        assert set(rows) == set(right["rows"]) and set(cols) == set(right["cols"])
        ri, ci = rows.get_indexer(right["rows"]), cols.get_indexer(right["cols"])
        for key in set(left)-{"rows", "cols"}:
            values = np.asarray(left[key])
            if key in {"raw", "centered"}:
                for start in range(0, len(ri), 2048):
                    assert_allclose(values[ri[start:start+2048]][:, ci], np.asarray(right[key])[start:start+2048],
                                    rtol=rtol, atol=atol, equal_nan=True, err_msg=f"{name}.{key}")
            else:
                assert_reference(portable(values[ri]), portable(right[key]), f"{name}.{key}")
    for block, tables in (("kinase_activity", ("results", "substrates", "condition_kinases", "condition_summary", "contrast_summary")),
                          ("pelsa", ("curve_results", "curve_points"))):
        assert (block in actual.uns) == (block in expected.uns), block
        if block not in expected.uns:
            continue
        for table in tables:
            left, right = [pd.DataFrame(obj.uns[block][table]).copy() for obj in (actual, expected)]
            if block == "pelsa":
                keys = ["peptide_id", "sample"] if table == "curve_points" else ["peptide_id"]
                left, right = [frame.set_index(keys) for frame in (left, right)]
            assert_frame_equal(left.sort_index().sort_index(axis=1), right.sort_index().sort_index(axis=1),
                               check_dtype=False, check_categorical=False, rtol=rtol, atol=atol)
            for column in right.columns:
                if probability_field(column):
                    assert_allclose(left.sort_index()[column], right.sort_index()[column],
                                    rtol=3e-5, atol=1e-12, equal_nan=True,
                                    err_msg=f"{block}.{table}.{column}")
