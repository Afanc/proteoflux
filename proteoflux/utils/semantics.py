"""
Canonical semantics for ProteoFlux.

This module is intentionally small and declarative:
  - Canonical analysis_type values
  - Canonical configuration keys (and their legacy aliases)

Implementation details live elsewhere (normalizers, pipelines).
"""

ANALYSIS_TYPES_CANONICAL = ("proteomics", "peptidomics", "phospho")

# Canonical rollup keys
CFG_PROTEIN_ROLLUP_METHOD = "protein_rollup_method"
CFG_PEPTIDE_ROLLUP_METHOD = "peptide_rollup_method"

# Legacy aliases (accepted, but deprecated elsewhere)
CFG_LEGACY_PIVOT_SIGNAL_METHOD = "pivot_signal_method"

LOG2_NORMALIZATION_METHODS = (
    "none", "log2", "median_equalization", "median_equalization_intersection",
    "median_equalization_by_tag", "global_linear", "global_loess", "local_linear", "local_loess",
)


def log2_normalization_methods(normalization: dict) -> list[str]:
    """Validate the transformation sequence without inferring scale from values."""
    methods = (normalization or {}).get("method")
    if methods is None:
        methods = ["none"]
    elif isinstance(methods, str):
        methods = [methods]
    else:
        try:
            methods = list(methods)  # H5AD restores lists as numpy arrays.
        except TypeError as exc:
            raise ValueError("normalization.method must be a string or sequence of log2-scale methods") from exc
    methods = [str(method).lower() for method in methods] or ["none"]
    for method in methods:
        if method not in LOG2_NORMALIZATION_METHODS:
            raise ValueError(f"Unsupported normalization {method!r}: quantitative inference "
                             "requires log2 units; log10 and quantile are not supported.")
    if "log2" in methods and (methods[0] != "log2" or methods.count("log2") != 1):
        raise ValueError("Normalization must apply log2 exactly once, before other methods.")
    return methods
