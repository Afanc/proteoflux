"""Shared phosphosite identity and sequence-window preparation.

PhosX consumes a ten-residue window (-5..+4), while result exports expose the
more conventional centred 15-mer (-7..+7). Peptide evidence is always used;
an optional local FASTA can complete flanks. No network lookup is performed
and no FASTA contents are stored in AnnData.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import pandas as pd

from proteoflux.utils.utils import log_info, log_time, log_warning


PHOSX_WINDOW_OFFSETS = tuple(range(-5, 5))
SITE_WINDOW_OFFSETS = tuple(range(-7, 8))
ACCESSION_SITES_COLUMN = "ACCESSION_SITES"
TENMER_COLUMN = "KINASE_10MER"
SITE_15MER_COLUMN = "SITE_15MER"

_SITE_RE = re.compile(r"^([STY])(\d+)$", flags=re.IGNORECASE)
_ACCESSION_SITE_RE = re.compile(r"^([^:;]+):([STY])(\d+)$", flags=re.IGNORECASE)
_PEPTIDE_INDEX_RE = re.compile(r"^(.*)\|p(\d+)$", flags=re.IGNORECASE)
_VALID_AA = frozenset("GPAVLIMCFYWHKRQNEDST")
_MISSING_TEXT = frozenset({"", "NA", "N/A", "NAN", "NONE", "NULL", "?", "-"})


def kinase_activity_config(config: dict) -> dict[str, Any]:
    analysis = (config or {}).get("analysis") or {}
    value = analysis.get("kinase_activity") or {}
    if not isinstance(value, dict):
        raise ValueError("analysis.kinase_activity must be a mapping.")
    return value


def configured_kinase_method(config: dict) -> str | None:
    value = kinase_activity_config(config).get("method")
    if value is None or not str(value).strip():
        return None
    method = str(value).strip().lower()
    if method not in {"ksea", "phosx"}:
        raise ValueError(
            "analysis.kinase_activity.method must be 'ksea' or 'phosx'; "
            f"received {method!r}."
        )
    return method


def _configured_fasta_path(config: dict) -> Path | None:
    kinase_cfg = kinase_activity_config(config)
    value = kinase_cfg.get("fasta")
    if value is None or not str(value).strip():
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(
            "analysis.kinase_activity.fasta must be a local FASTA path or null."
        )
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Kinase-activity FASTA does not exist or is not a file: {path}"
        )
    return path


def _fasta_identifiers(header: str) -> list[str]:
    first_token = header.strip().split(maxsplit=1)[0]
    identifiers = [first_token]
    fields = first_token.split("|")
    if len(fields) >= 2 and fields[0].lower() in {"sp", "tr"}:
        identifiers.insert(0, fields[1])
    return [item.strip().upper() for item in identifiers if item.strip()]


@lru_cache(maxsize=4)
def _read_fasta_cached(
    resolved_path: str,
    file_size: int,
    modified_ns: int,
) -> dict[str, str]:
    del file_size, modified_ns  # cache-key inputs; intentionally not persisted
    sequences: dict[str, str] = {}
    current_ids: list[str] = []
    chunks: list[str] = []

    def commit() -> None:
        if not current_ids:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            return
        for identifier in current_ids:
            previous = sequences.get(identifier)
            if previous is not None and previous != sequence:
                raise ValueError(
                    f"FASTA identifier {identifier!r} occurs with conflicting sequences."
                )
            sequences[identifier] = sequence

    with Path(resolved_path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                current_ids = _fasta_identifiers(line[1:])
                chunks = []
                if not current_ids:
                    raise ValueError(
                        f"FASTA header on line {line_number} has no identifier."
                    )
            else:
                if not current_ids:
                    raise ValueError(
                        f"FASTA sequence appears before the first header on line {line_number}."
                    )
                chunks.append(line)
    commit()
    if not sequences:
        raise ValueError(f"FASTA contains no sequence records: {resolved_path}")
    return sequences


def _load_fasta(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    stat = resolved.stat()
    return _read_fasta_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def validate_kinase_sequence_config(config: dict) -> None:
    """Fail fast on the shared method/FASTA configuration."""
    configured_kinase_method(config)
    fasta_path = _configured_fasta_path(config)
    if fasta_path is not None:
        records = _load_fasta(fasta_path)
        log_info(
            "Kinase-activity FASTA validation passed: "
            f"file={fasta_path.name!r}, identifiers={len(records):,}."
        )


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.upper() in _MISSING_TEXT else text


def parse_accession_sites(value: Any) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for item in _text(value).split(";"):
        match = _ACCESSION_SITE_RE.fullmatch(item.strip())
        if match is None:
            continue
        pair = (
            match.group(1).strip().upper(),
            match.group(2).upper(),
            int(match.group(3)),
        )
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _site_only_window(amino_acid: str, offsets: tuple[int, ...]) -> str:
    chars = ["_"] * len(offsets)
    chars[offsets.index(0)] = amino_acid.upper()
    return "".join(chars)


def _normalise_residue(value: str) -> str:
    upper = value.upper()
    return upper if upper in _VALID_AA else "_"


def _window_from_peptide(
    sequence: str,
    focal_position: int,
    phosphoform_variants: str,
    offsets: tuple[int, ...],
) -> tuple[str, set[int]] | None:
    clean_sequence = "".join(char.upper() for char in sequence if char.isalpha())
    if focal_position < 1 or focal_position > len(clean_sequence):
        return None

    phosphorylated: set[int] = set()
    for variant in phosphoform_variants.split("|"):
        for item in variant.split(";"):
            try:
                phosphorylated.add(int(item.strip()))
            except (TypeError, ValueError):
                continue

    chars: list[str] = []
    for offset in offsets:
        peptide_position = focal_position + offset
        if 1 <= peptide_position <= len(clean_sequence):
            residue = _normalise_residue(clean_sequence[peptide_position - 1])
            if (
                offset != 0
                and peptide_position in phosphorylated
                and residue in {"S", "T", "Y"}
            ):
                residue = residue.lower()
        else:
            residue = "_"
        chars.append(residue)

    centre = offsets.index(0)
    chars[centre] = chars[centre].upper()
    if chars[centre] not in {"S", "T", "Y"}:
        return None
    return "".join(chars), {value - focal_position for value in phosphorylated}


def _merge_windows(
    windows: Iterable[str],
    focal_amino_acid: str,
    offsets: tuple[int, ...],
) -> tuple[str, bool]:
    values = [window for window in windows if len(window) == len(offsets)]
    if not values:
        return _site_only_window(focal_amino_acid, offsets), False

    conflict = False
    merged: list[str] = []
    centre = offsets.index(0)
    for column in range(len(offsets)):
        observed = [window[column] for window in values if window[column] != "_"]
        identities = {residue.upper() for residue in observed}
        if not identities:
            residue = "_"
        elif len(identities) > 1:
            residue = "_"
            conflict = True
        else:
            residue = next(iter(identities))
            if column != centre and any(item.islower() for item in observed):
                residue = residue.lower()
        merged.append(residue)

    merged[centre] = focal_amino_acid.upper()
    return "".join(merged), conflict


def _window_from_fasta(
    sequence: str,
    amino_acid: str,
    absolute_position: int,
    auxiliary_offsets: set[int],
    offsets: tuple[int, ...],
) -> str | None:
    if absolute_position < 1 or absolute_position > len(sequence):
        return None
    if sequence[absolute_position - 1].upper() != amino_acid.upper():
        return None

    chars: list[str] = []
    for offset in offsets:
        protein_position = absolute_position + offset
        if 1 <= protein_position <= len(sequence):
            residue = _normalise_residue(sequence[protein_position - 1])
            if (
                offset != 0
                and offset in auxiliary_offsets
                and residue in {"S", "T", "Y"}
            ):
                residue = residue.lower()
        else:
            residue = "_"
        chars.append(residue)
    chars[offsets.index(0)] = chars[offsets.index(0)].upper()
    return "".join(chars)


def _precursor_windows(
    adata: ad.AnnData,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, set[int]]]:
    phosx_windows: dict[str, list[str]] = {}
    export_windows: dict[str, list[str]] = {}
    offsets: dict[str, set[int]] = {}
    payload = adata.uns.get("precursors")
    if not isinstance(payload, dict):
        return phosx_windows, export_windows, offsets

    feature_values = list(payload.get("protein_index", []))
    peptide_values = list(payload.get("peptide_seq", []))
    phosphoform_values = list(payload.get("phosphoform_positions", []))
    if not phosphoform_values:
        phosphoform_values = [""] * len(feature_values)
    if not (len(feature_values) == len(peptide_values) == len(phosphoform_values)):
        raise ValueError(
            "AnnData uns['precursors'] phosphosite sequence metadata is not aligned."
        )

    retained = set(adata.var_names.astype(str))
    for feature_value, peptide_value, phosphoform_value in zip(
        feature_values, peptide_values, phosphoform_values
    ):
        feature = str(feature_value)
        if feature not in retained:
            continue
        match = _PEPTIDE_INDEX_RE.fullmatch(str(peptide_value).strip())
        if match is None:
            continue
        phosx_item = _window_from_peptide(
            match.group(1),
            int(match.group(2)),
            str(phosphoform_value or ""),
            PHOSX_WINDOW_OFFSETS,
        )
        export_item = _window_from_peptide(
            match.group(1),
            int(match.group(2)),
            str(phosphoform_value or ""),
            SITE_WINDOW_OFFSETS,
        )
        if phosx_item is None or export_item is None:
            continue
        phosx_window, item_offsets = phosx_item
        export_window, _ = export_item
        phosx_windows.setdefault(feature, []).append(phosx_window)
        export_windows.setdefault(feature, []).append(export_window)
        offsets.setdefault(feature, set()).update(item_offsets)
    return phosx_windows, export_windows, offsets


@log_time("Preparing kinase sequence windows")
def prepare_kinase_sequences(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """Attach the internal PhosX 10-mer and an export-facing centred 15-mer."""
    analysis_type = str(
        (adata.uns.get("preprocessing", {}) or {}).get("analysis_type", "phospho")
    ).lower()
    if analysis_type != "phospho":
        return adata

    fasta_path = _configured_fasta_path(config)
    fasta = _load_fasta(fasta_path) if fasta_path is not None else None
    precursor_phosx_windows, precursor_export_windows, auxiliary_offsets = (
        _precursor_windows(adata)
    )

    if ACCESSION_SITES_COLUMN not in adata.var.columns:
        raise ValueError(
            "Phospho sequence preparation requires adata.var['ACCESSION_SITES']."
        )
    accession_column = adata.var[ACCESSION_SITES_COLUMN].tolist()

    final_accession_sites: list[str] = []
    tenmers: list[str] = []
    fifteenmers: list[str] = []
    peptide_conflicts = 0
    fasta_conflicts = 0
    fasta_central_mismatches = 0
    fasta_resolved_features = 0

    for feature, compact_value in zip(
        adata.var_names.astype(str), accession_column
    ):
        mappings = parse_accession_sites(compact_value)
        compact = ";".join(
            f"{accession}:{amino_acid}{position}"
            for accession, amino_acid, position in mappings
        )
        final_accession_sites.append(compact)

        # A retained multisite feature has one measured signal but several
        # possible centres. Duplicating that signal into several PhosX rows
        # would inflate evidence, so keep it auditable but ineligible.
        feature_site_text = feature.rsplit("|", 1)[1] if "|" in feature else ""
        site_match = _SITE_RE.fullmatch(feature_site_text.strip())
        if site_match is None:
            tenmers.append("__________")
            fifteenmers.append("_______________")
            continue

        focal_amino_acid = site_match.group(1).upper()

        peptide_window, peptide_conflict = _merge_windows(
            precursor_phosx_windows.get(feature, []),
            focal_amino_acid,
            PHOSX_WINDOW_OFFSETS,
        )
        export_window, export_conflict = _merge_windows(
            precursor_export_windows.get(feature, []),
            focal_amino_acid,
            SITE_WINDOW_OFFSETS,
        )
        peptide_conflicts += int(peptide_conflict or export_conflict)
        final_window = peptide_window
        final_export_window = export_window

        if fasta is not None and mappings:
            resolved_phosx: list[str] = []
            resolved_export: list[str] = []
            for accession, amino_acid, absolute_position in mappings:
                protein_sequence = fasta.get(accession)
                if protein_sequence is None:
                    continue
                phosx_window = _window_from_fasta(
                    protein_sequence,
                    amino_acid,
                    absolute_position,
                    auxiliary_offsets.get(feature, set()),
                    PHOSX_WINDOW_OFFSETS,
                )
                if phosx_window is None:
                    fasta_central_mismatches += 1
                    continue
                export_fasta_window = _window_from_fasta(
                    protein_sequence,
                    amino_acid,
                    absolute_position,
                    auxiliary_offsets.get(feature, set()),
                    SITE_WINDOW_OFFSETS,
                )
                if export_fasta_window is None:
                    # Both windows share the same centre, so this is defensive.
                    fasta_central_mismatches += 1
                    continue
                resolved_phosx.append(phosx_window)
                resolved_export.append(export_fasta_window)

            unique_phosx = list(dict.fromkeys(resolved_phosx))
            unique_export = list(dict.fromkeys(resolved_export))
            if len(unique_phosx) == 1:
                final_window = unique_phosx[0]
                fasta_resolved_features += 1

            if len(unique_export) == 1:
                final_export_window = unique_export[0]

            if len(unique_phosx) > 1 or len(unique_export) > 1:
                fasta_conflicts += 1

        tenmers.append(final_window)
        fifteenmers.append(final_export_window)

    output = adata.copy()
    output.var[ACCESSION_SITES_COLUMN] = pd.Series(
        final_accession_sites, index=output.var_names, dtype="string"
    )
    output.var[TENMER_COLUMN] = pd.Series(
        tenmers, index=output.var_names, dtype="string"
    )
    output.var[SITE_15MER_COLUMN] = pd.Series(
        fifteenmers, index=output.var_names, dtype="string"
    )

    output.uns["kinase_sequences"] = {
        "schema_version": 2,
        "phosx_window": "-5..+4",
        "export_window": "-7..+7",
        "auxiliary_phosphosite_policy": "union across contributing phosphoforms",
        "fasta_used": fasta is not None,
        "fasta_filename": fasta_path.name if fasta_path is not None else "",
        "peptide_consensus_conflicts": int(peptide_conflicts),
        "fasta_accession_conflicts": int(fasta_conflicts),
        "fasta_central_mismatches": int(fasta_central_mismatches),
    }

    log_info(
        "Kinase sequence windows prepared: "
        f"features={output.n_vars:,}, fasta_resolved={fasta_resolved_features:,}, "
        f"fasta_conflicts={fasta_conflicts:,}, "
        f"peptide_conflicts={peptide_conflicts:,}."
    )
    if fasta_central_mismatches:
        log_warning(
            "FASTA mappings with an out-of-range position or central-residue "
            f"mismatch were ignored: {fasta_central_mismatches:,}."
        )
    return output
