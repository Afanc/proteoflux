"""Compact labelled scientific references; deliberately excludes UI coordinates.

No automatic baseline update is allowed during pytest. Numeric values, NaN/Inf
locations, identifiers, schemas and thresholds are compared separately from
timestamps, paths and non-identifiable embedding orientation.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DIR = Path(__file__).parent / "references"


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items() if v is not None}
    if isinstance(value, (np.ndarray, pd.Index, list, tuple)):
        return [portable(v) for v in list(value)]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is pd.NA:
        return "NaN"
    return value


def frame_payload(frame):
    frame = frame.copy()
    frame.index = frame.index.astype(str)
    assert frame.index.is_unique, "Reference tables require unique labelled rows"
    frame = frame.sort_index().sort_index(axis=1)
    return {"index": frame.index.tolist(), "columns": frame.columns.tolist(),
            "values": portable(frame.to_numpy())}


def scientific_payload(adata):
    assert adata.obs_names.is_unique and adata.var_names.is_unique
    obs_order = np.argsort(adata.obs_names)
    var_order = np.argsort(adata.var_names)
    obj = adata[obs_order, var_order]
    contrasts = list(map(str, obj.uns.get("contrast_names", [])))
    payload = {"samples": obj.obs_names.tolist(), "features": obj.var_names.tolist(),
               "contrasts": contrasts, "X": portable(obj.X),
               "obs": frame_payload(obj.obs), "var": frame_payload(obj.var),
               "layers": {k: portable(v) for k, v in sorted(obj.layers.items()) if k != "centered"},
               # covariate_p has a confirmed ambiguous producer defect, tested
               # separately as desired behavior rather than blessed by a golden.
               "varm": {k: portable(v) for k, v in sorted(obj.varm.items()) if k not in {"PCs", "covariate_p"}},
               "analysis": portable(obj.uns.get("analysis", {})),
               "analysis_type": obj.uns["preprocessing"]["analysis_type"]}
    for key in ("pilot_study_mode", "has_covariate", "missingness_source", "missingness_rule",
                "model_type", "decomposition_rule", "n_fully_imputed_cells"):
        if key in obj.uns:
            payload[key] = portable(obj.uns[key])
    if "residual_variance" in obj.uns:
        payload["residual_variance"] = portable(np.asarray(adata.uns["residual_variance"])[var_order])
    if "missingness" in obj.uns:
        payload["missingness"] = frame_payload(obj.uns["missingness"])
    for name in ("peptides", "precursors"):
        if name not in obj.uns:
            continue
        block = obj.uns[name]
        order = np.argsort(block["rows"])
        sample_order = np.argsort(block["cols"])
        evidence = {"cols": sorted(map(str, block["cols"]))}
        for key, value in block.items():
            if key == "cols":
                continue
            values = np.asarray(value)[order]
            evidence[key] = portable(values[:, sample_order] if key in {"raw", "centered"} else values)
        payload[name] = evidence
    for name in ("consistent_peptides_per_condition", "consistent_precursors_per_condition"):
        if name in obj.uns:
            block = obj.uns[name]
            payload[name] = frame_payload(pd.DataFrame(block["values"], index=block["index"], columns=block["columns"]))
    if "kinase_activity" in obj.uns:
        block = obj.uns["kinase_activity"]
        payload["kinase_activity"] = {"method": block["method"]}
        for key in ("results", "substrates", "condition_summary", "condition_kinases", "contrast_summary"):
            payload["kinase_activity"][key] = frame_payload(block[key])
    if "pelsa" in obj.uns:
        block = obj.uns["pelsa"]
        results = pd.DataFrame(block["curve_results"]).set_index("peptide_id", drop=False)
        points = pd.DataFrame(block["curve_points"])
        points.index = points.peptide_id.astype(str)+"|"+points["sample"].astype(str)
        payload["pelsa"] = {"curve_results": frame_payload(results), "curve_points": frame_payload(points)}
    for key in ("q_ebayes", "ft_q_ebayes", "raw_q_ebayes"):
        if key in obj.varm:
            payload[key+"_below_0.05"] = portable(np.asarray(obj.varm[key]) < .05)
    return portable(payload)


def probability_field(name):
    return name in {"p_raw", "q_raw", "p_ebayes", "q_ebayes", "pvalue", "qvalue",
                    "curve_p_value", "curve_q_value", "covariate_q"} or name.endswith(
                        ("_p_raw", "_q_raw", "_p_ebayes", "_q_ebayes"))


def assert_reference(actual, expected, path="root", *, rtol=3e-6, atol=2e-6):
    """Named failure paths; exact labels/masks and bounded float differences."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), f"{path}: fields differ: {actual.keys() ^ expected.keys()}"
        if set(expected) == {"index", "columns", "values"}:
            assert actual["index"] == expected["index"], path+": row labels"
            assert actual["columns"] == expected["columns"], path+": columns"
            assert len(actual["values"]) == len(expected["values"]), path+": row count"
            for label, left, right in zip(expected["index"], actual["values"], expected["values"]):
                assert len(left) == len(right) == len(expected["columns"]), path+": table shape"
                for column, a, e in zip(expected["columns"], left, right):
                    assert_reference(a, e, f"{path}.{column}[{label}]", rtol=rtol, atol=atol)
            return
        for key in expected:
            assert_reference(actual[key], expected[key], f"{path}.{key}", rtol=rtol, atol=atol)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_reference(a, e, f"{path}[{i}]", rtol=rtol, atol=atol)
    elif isinstance(expected, float):
        if ".pelsa.curve_results." in path:
            rtol, atol = 1e-4, 2e-4
        # p/q arrays need tighter absolute tolerances than float32 intensities.
        if any(probability_field(part.split("[")[0]) for part in path.split(".")):
            rtol, atol = (1e-4 if ".pelsa." in path else 3e-5), 1e-12
        assert isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def load_reference(name):
    return json.loads((REFERENCE_DIR/f"{name}.json").read_text())


def reference_json(value, depth=0):
    """Readable JSON with one row per numeric/table vector, keeping diffs compact."""
    if isinstance(value, dict):
        entries = [json.dumps(k)+": "+reference_json(v, depth+1) for k, v in value.items()]
        opening, closing = "{", "}"
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        entries = [reference_json(v, depth+1) for v in value]
        opening, closing = "[", "]"
    else:
        return json.dumps(value, allow_nan=False)
    if not entries:
        return opening+closing
    return opening+"\n"+",\n".join("  "*(depth+1)+entry for entry in entries)+"\n"+"  "*depth+closing
