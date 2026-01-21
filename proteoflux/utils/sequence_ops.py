import re
from typing import Any, Mapping, Callable

import polars as pl

# ----------------------------------------------------------------------
# Shared patterns (single source of truth for peptide-index normalization)
# ----------------------------------------------------------------------

_RE_STRIP_US   = r"^_+|_+$"
_RE_MET_OXI    = r"M\[[^\]]*Oxidation[^\]]*\]"
_RE_NTERM_TAG  = r"^n\[[^\]]*\](?=[A-Z])"
_RE_CTERM_TAG  = r"c\[[^\]]*\]$"
_RE_BRACKETS   = r"\[[^\]]+\]"

def strip_mods(s: str | None) -> str | None:
    """
    Exact copy of DataHarmonizer._strip_mods logic.
    """
    if s is None:
        return None

    out = s.strip("_")

    # 1) n-terminal tags
    out = re.sub(_RE_NTERM_TAG, "", out)

    # 2) C-terminal tags
    out = re.sub(_RE_CTERM_TAG, "", out)

    # Generic: drop any bracketed annotation and parentheses
    out = re.sub(_RE_BRACKETS, "", out)
    out = out.replace("(", "").replace(")", "")

    return out


def convert_numeric_ptms(
    s: str | None,
    *,
    enabled: bool,
    ptm_map: Mapping[str, Any],
) -> str | None:
    """
    Exact copy of DataHarmonizer._convert_numeric_ptms logic, parameterized.
    """
    if s is None:
        return None
    if not enabled:
        return s

    def _replace(match: re.Match) -> str:
        token = match.group(1)  # AA letter or 'n'/'c'
        mass_raw = match.group(2)

        try:
            mass_val = float(mass_raw)
        except ValueError:
            return match.group(0)

        candidates = {
            mass_raw,
            f"{mass_val:.4f}",
            f"{mass_val:.5f}",
            f"{mass_val:.6f}",
        }

        name = None
        meta = None
        for key in candidates:
            if key in ptm_map:
                meta = ptm_map[key]
                if isinstance(meta, dict):
                    name = meta.get("name") or meta.get("label") or key
                else:
                    name = str(meta)
                break

        if name is None:
            return match.group(0)

        return f"{token}[{name}]"

    return re.sub(r"([A-Znc])\[(\-?\d+\.\d+)\]", _replace, s)

def normalize_peptide_index_seq(
    s: str | None,
    *,
    collapse_met_oxidation: bool = True,
    collapse_all_ptms: bool = False,
    convert_numeric_ptms_enabled: bool = False,
    ptm_map: Mapping[str, Any] | None = None,
) -> str | None:
    """Normalize a modified peptide sequence into a canonical *peptide index* sequence.

    Single source of truth for peptide identity normalization across analysis types.

    Rules:
      - Always strip leading/trailing underscores.
      - Optionally collapse Met oxidation tokens: ``M[...Oxidation...]`` -> ``M``.
      - Optionally drop all PTM annotations (brackets and terminal tags).
      - Optionally convert numeric PTMs (requires ``ptm_map``; used in harmonizer).
    """
    if s is None:
        return None

    seq = s

    if convert_numeric_ptms_enabled:
        if ptm_map is None:
            raise ValueError("ptm_map must be provided when convert_numeric_ptms_enabled=True")
        seq = convert_numeric_ptms(seq, enabled=True, ptm_map=ptm_map)

    seq = seq.strip("_").strip()

    if collapse_met_oxidation:
        # Met-only: do not remove '[Oxidation...]' unless it is on M.
        seq = re.sub(_RE_MET_OXI, "M", seq)

    if collapse_all_ptms:
        # Mirror harmonizer semantics: remove terminal tags and any remaining bracket tokens.
        seq = re.sub(_RE_NTERM_TAG, "", seq)
        seq = re.sub(_RE_CTERM_TAG, "", seq)
        seq = re.sub(_RE_BRACKETS, "", seq)

    return seq

def expr_peptide_index_seq(
    col: str,
    *,
    collapse_met_oxidation: bool = True,
    collapse_all_ptms: bool = False,
    convert_numeric_ptms_enabled: bool = False,
    ptm_map: Mapping[str, Any] | None = None,
) -> pl.Expr:
    """Polars expression variant of :func:`normalize_peptide_index_seq`.

    Default is a pure-regex expression (fast). If convert_numeric_ptms_enabled=True,
    falls back to map_elements (slow) and should be used sparingly.
    """
    base = pl.col(col).cast(pl.Utf8)

    if convert_numeric_ptms_enabled:
        if ptm_map is None:
            raise ValueError("ptm_map must be provided when convert_numeric_ptms_enabled=True")
        fn: Callable[[str | None], str | None] = lambda s: normalize_peptide_index_seq(
            s,
            collapse_met_oxidation=collapse_met_oxidation,
            collapse_all_ptms=collapse_all_ptms,
            convert_numeric_ptms_enabled=True,
            ptm_map=ptm_map,
        )
        return base.map_elements(fn, return_dtype=pl.Utf8)

    expr = base.str.replace_all(_RE_STRIP_US, "")

    if collapse_met_oxidation:
        expr = expr.str.replace_all(_RE_MET_OXI, "M")

    if collapse_all_ptms:
        expr = expr.str.replace_all(_RE_NTERM_TAG, "")
        expr = expr.str.replace_all(_RE_CTERM_TAG, "")
        expr = expr.str.replace_all(_RE_BRACKETS, "")

    return expr

def expr_strip_bracket_mods(col: str) -> pl.Expr:
    """
    Exact copy of missed-cleavage SEQ_RAW stripping:
      - remove any [...] tokens
      - strip leading/trailing underscores
    """
    return (
        pl.col(col)
        .str.replace_all(r"\[[^\]]*\]", "")
        .str.replace_all(r"^_+|_+$", "")
    )


def expr_strip_underscores(col: str) -> pl.Expr:
    """Exact copy of missed-cleavage PEP_ID stripping."""
    return pl.col(col).str.replace_all(r"^_+|_+$", "")

