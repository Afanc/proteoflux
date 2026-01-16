import re
from typing import Any, Mapping

import polars as pl

# ----------------------------------------------------------------------
# Python string utilities (used via map_elements in harmonizer)
# ----------------------------------------------------------------------

def strip_mods(s: str | None) -> str | None:
    """
    Exact copy of DataHarmonizer._strip_mods logic.
    """
    if s is None:
        return None

    out = s.strip("_")

    # 1) n-terminal tags
    out = re.sub(r"^n\[[^]]*\](?=[A-Z])", "", out)

    # 2) C-terminal tags
    out = re.sub(r"c\[[^]]*\]$", "", out)

    # Generic: drop any bracketed annotation and parentheses
    out = re.sub(r"\[.*?\]", "", out)
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


def normalize_peptido_index_seq(
    s: str | None,
    *,
    convert_numeric_ptms_enabled: bool,
    collapse_all_ptms: bool,
    ptm_map: Mapping[str, Any],
) -> str | None:
    """
    Exact copy of DataHarmonizer._normalize_peptido_index_seq logic, parameterized.
    """
    if s is None:
        return None

    seq = convert_numeric_ptms(s, enabled=convert_numeric_ptms_enabled, ptm_map=ptm_map)
    seq = seq.strip("_").strip()
    seq = re.sub(r"M\[[^]]*Oxidation[^]]*\]", "M", seq)

    if collapse_all_ptms:
        seq = re.sub(r"^n\[[^]]*\](?=[A-Z])", "", seq)
        seq = re.sub(r"c\[[^]]*\]$", "", seq)
        seq = re.sub(r"\[[^]]+\]", "", seq)

    return seq


# ----------------------------------------------------------------------
# Polars expression utilities (used in preprocessing)
# ----------------------------------------------------------------------

def expr_clean_peptide_seq(col: str) -> pl.Expr:
    """
    Exact copy of the repeated preprocessing expression:
      - strip Oxidation[...] (case-insensitive)
      - strip leading/trailing underscores
    """
    return (
        pl.col(col)
        .str.replace_all(r"(?i)\[Oxidation[^\]]*\]", "")
        .str.replace_all(r"^_+|_+$", "")
    )


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

