from __future__ import annotations

import warnings
from typing import Optional

from proteoflux.utils.semantics import ANALYSIS_TYPES_CANONICAL

CANONICAL = set(ANALYSIS_TYPES_CANONICAL)


def normalize_analysis_type(raw: Optional[str]) -> str:
    """
    Normalize analysis_type to canonical strings (strict, explicit).

    Canonical:
      - proteomics
      - peptidomics
      - phospho

    Accepted aliases:
      - DIA, DDA, proteo, protein, proteins, proteomics -> proteomics
      - peptido -> peptidomics
      - phosphoproteomics -> phospho
    """
    if raw is None:
        return "proteomics"

    s = str(raw).strip()
    if not s:
        return "proteomics"

    key = s.lower()

    alias_map = {
        "proteo": "proteomics",
        "protein": "proteomics",
        "proteins": "proteomics",
        "proteomics": "proteomics",
        "peptido": "peptidomics",
        "peptidomics": "peptidomics",
        "phosphoproteomics": "phospho",
        "phospho": "phospho",
    }

    if key in alias_map:
        out = alias_map[key]
        if out != key and key not in CANONICAL:
            warnings.warn(
                f"analysis_type={s!r} is deprecated; use {out!r}.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return out

    raise ValueError(
        f"Unsupported analysis_type={s!r}. "
        "Use one of: 'proteomics', 'peptidomics', 'phospho'."
    )

