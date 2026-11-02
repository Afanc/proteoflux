"""Optional, row-aligned FASTA coordinates for proteomics peptide drill-down."""

from pathlib import Path

import numpy as np

from proteoflux.analysis.kinase_sequences import _load_fasta
from proteoflux.utils.sequence_ops import is_tryptic_cleavage, strip_mods


def load_protein_coverage_fasta(analysis: dict, analysis_type: str) -> dict[str, str] | None:
    settings = analysis.get("protein_coverage")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("analysis.protein_coverage must be a mapping or null.")
    value = settings.get("fasta")
    if value is None or value == "":
        return None
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(item, (str, Path)) or not str(item).strip() for item in paths):
        raise ValueError(
            "analysis.protein_coverage.fasta must be a local FASTA path, "
            "a nonempty list of paths, or null."
        )
    if analysis_type != "proteomics":
        raise ValueError("analysis.protein_coverage.fasta is supported only for proteomics.")
    sequences: dict[str, str] = {}
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Protein-coverage FASTA does not exist or is not a file: {path}")
        for identifier, sequence in _load_fasta(path).items():
            previous = sequences.get(identifier)
            if previous is not None and previous != sequence:
                raise ValueError(
                    f"Protein-coverage FASTA identifier {identifier!r} occurs with "
                    f"conflicting sequences across files (including {path})."
                )
            sequences[identifier] = sequence
    return sequences


def protein_coverage_metadata(
    protein_index: list[str],
    peptide_seq: list[str],
    fasta: dict[str, str],
) -> dict:
    """Map exact accession identities; coordinates are 1-based and inclusive.

    Use the peptide payload's protein index, which is an accession or group in
    the packaged proteomics mappings. Never select one member of a group.
    """
    size = len(protein_index)
    result = {
        "reference_accession": [""] * size,
        "protein_length_aa": np.full(size, np.nan),
        "peptide_start": np.full(size, np.nan),
        "peptide_end": np.full(size, np.nan),
        "peptide_mapping_status": ["unmapped_reference"] * size,
        "tryptic_missed_cleavages": np.zeros(size, dtype=np.int64),
    }
    for i, (protein, modified_sequence) in enumerate(zip(protein_index, peptide_seq)):
        sequence = strip_mods(modified_sequence).upper()
        result["tryptic_missed_cleavages"][i] = sum(
            is_tryptic_cleavage(c, n) for c, n in zip(sequence, sequence[1:])
        )
        if ";" in protein:
            result["peptide_mapping_status"][i] = "ambiguous_protein_group"
            continue
        accession = protein.strip()
        reference = fasta.get(accession)
        if reference is None:
            continue
        result["reference_accession"][i] = accession
        result["protein_length_aa"][i] = len(reference)
        start = reference.find(sequence) if sequence else -1
        if start < 0:
            result["peptide_mapping_status"][i] = "unmapped_peptide"
        elif reference.find(sequence, start + 1) >= 0:
            result["peptide_mapping_status"][i] = "ambiguous_peptide"
        else:
            result["peptide_mapping_status"][i] = "mapped"
            result["peptide_start"][i] = start + 1
            result["peptide_end"][i] = start + len(sequence)
    return result
