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

