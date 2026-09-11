"""Pinned in-memory compatibility layer for PhosX 0.23.0.

PhosX currently exposes a file-oriented CLI and does not accept a random seed.
This module contains all private imports and keeps PhosX's scoring and
enrichment equations while avoiding repeated Pandas objects in the permutation
loop and caching sequence-only work across contrasts.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


REQUIRED_PHOSX_VERSION = "0.23.0"


@dataclass
class PhosXRun:
    activity: pd.DataFrame
    ser_thr_assignments: pd.DataFrame
    tyr_assignments: pd.DataFrame
    version: str


@dataclass
class _FamilyResources:
    names: tuple[str, ...]
    amino_acids: tuple[str, ...]
    pssm: np.ndarray
    sorted_background: tuple[np.ndarray, ...]
    assignment_cache: dict[tuple[int, float], dict[str, np.ndarray]] = field(
        default_factory=dict
    )


def _imports() -> dict[str, Any]:
    try:
        import phosx
        from phosx.utils import (
            benjamini_hochberg,
            concat_keep_na,
            hdf5_to_dict,
            read_pssm_score_quantiles,
            read_pssms,
        )
    except ImportError as exc:
        raise ImportError(
            "PhosX analysis requires the pinned dependency phosx==0.23.0."
        ) from exc

    installed = str(getattr(phosx, "__version__", ""))
    if installed != REQUIRED_PHOSX_VERSION:
        raise RuntimeError(
            "ProteoFlux's PhosX adapter is pinned to phosx==0.23.0; "
            f"found {installed or 'unknown'!r}."
        )
    return {
        "package": phosx,
        "benjamini_hochberg": benjamini_hochberg,
        "concat_keep_na": concat_keep_na,
        "hdf5_to_dict": hdf5_to_dict,
        "read_pssm_score_quantiles": read_pssm_score_quantiles,
        "read_pssms": read_pssms,
    }


def validate_phosx_installation() -> None:
    _imports()


def validate_phosx_resources(overrides: dict[str, Any] | None = None) -> None:
    imported = _imports()
    _resource_paths(imported["package"], overrides)


def _resource_paths(package: Any, overrides: dict[str, Any] | None) -> dict[str, Path]:
    data_dir = Path(package.__file__).resolve().parent / "data"
    defaults = {
        "s_t_pssm": data_dir / "S_T_PSSMs.h5",
        "s_t_quantiles": data_dir / "S_T_PSSM_score_quantiles.h5",
        "y_pssm": data_dir / "Y_PSSMs.h5",
        "y_quantiles": data_dir / "Y_PSSM_score_quantiles.h5",
        # Match the 0.23.0 CLI default (the function-level default is older).
        "kinase_metadata": data_dir / "kinase_metadata_annotated.h5",
    }
    for key, value in (overrides or {}).items():
        if key not in defaults:
            raise ValueError(f"Unknown PhosX resource override: {key!r}.")
        if value is not None and str(value).strip():
            defaults[key] = Path(str(value)).expanduser()
    for key, path in defaults.items():
        if not path.is_file():
            raise FileNotFoundError(f"PhosX resource {key!r} was not found: {path}")
    return defaults


def _family_seed(random_seed: int, family: int) -> int:
    return int(
        np.random.SeedSequence([int(random_seed), int(family)]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


# Empirical KS distribution. Worker input is only a chunk of seeds;
# the read-only arrays are installed once per process.
_NULL_WEIGHTS: np.ndarray | None = None
_NULL_HIT_INDICES: tuple[np.ndarray, ...] = ()
_NULL_HIT_ORDINALS: tuple[np.ndarray, ...] = ()
_NULL_MISS_COUNTS: np.ndarray | None = None
_NULL_RANKS: np.ndarray | None = None


def _init_null_worker(
    weights: np.ndarray, hit_indices: tuple[np.ndarray, ...]
) -> None:
    global _NULL_WEIGHTS
    global _NULL_HIT_INDICES
    global _NULL_HIT_ORDINALS
    global _NULL_MISS_COUNTS
    global _NULL_RANKS

    _NULL_WEIGHTS = np.asarray(weights, dtype=np.float64)
    _NULL_HIT_INDICES = hit_indices
    _NULL_HIT_ORDINALS = tuple(
        np.arange(len(indices), dtype=np.float64) for indices in hit_indices
    )
    _NULL_MISS_COUNTS = np.asarray(
        [len(_NULL_WEIGHTS) - len(indices) for indices in hit_indices],
        dtype=np.float64,
    )
    _NULL_RANKS = np.arange(len(_NULL_WEIGHTS), dtype=np.int64)


def _null_ks_from_positions(
    positions: np.ndarray,
    weights: np.ndarray,
    ordinals: np.ndarray,
    miss_count: float,
) -> float:
    hit_cumsum = np.cumsum(weights[positions], dtype=np.float64)
    hit_cumsum /= hit_cumsum[-1]

    # The running sum can reach a maximum immediately after a hit and a
    # minimum immediately before a hit. Inspecting those positions is
    # algebraically identical to materialising the full dense PhosX vector.
    after_hit = hit_cumsum - ((positions - ordinals) / miss_count)
    before_hit = np.empty_like(after_hit)
    before_hit[0] = -positions[0] / miss_count
    before_hit[1:] = hit_cumsum[:-1] - (
        (positions[1:] - ordinals[1:]) / miss_count
    )

    maximum = max(0.0, float(np.max(after_hit)))
    minimum = min(0.0, float(np.min(before_hit)))
    # PhosX deliberately resolves an exact max/min tie toward the minimum.
    # The sparse closed form reaches the same values through a different
    # floating-point summation path, so normalize numerical near-ties first.
    if np.isclose(abs(maximum), abs(minimum), rtol=1e-12, atol=1e-15):
        return minimum
    return maximum if abs(maximum) > abs(minimum) else minimum


def _null_ks_chunk(seeds: np.ndarray) -> np.ndarray:
    if _NULL_WEIGHTS is None or _NULL_MISS_COUNTS is None or _NULL_RANKS is None:
        raise RuntimeError("PhosX null-distribution worker was not initialized.")

    n_sites = len(_NULL_WEIGHTS)
    result = np.empty((len(seeds), len(_NULL_HIT_INDICES)), dtype=np.float64)
    inverse_order = np.empty(n_sites, dtype=np.int64)
    for row, seed in enumerate(seeds):
        # RandomState.choice matches the seeded PhosX 0.23.0 row shuffle.
        order = np.random.RandomState(int(seed)).choice(
            n_sites, size=n_sites, replace=False
        )
        inverse_order[order] = _NULL_RANKS
        for column, original_hits in enumerate(_NULL_HIT_INDICES):
            positions = np.sort(inverse_order[original_hits])
            result[row, column] = _null_ks_from_positions(
                positions,
                _NULL_WEIGHTS,
                _NULL_HIT_ORDINALS[column],
                float(_NULL_MISS_COUNTS[column]),
            )
    return result


def _fast_empirical_distribution(
    assignments: pd.DataFrame,
    scores: pd.Series,
    *,
    n_permutations: int,
    n_processes: int,
    random_seed: int,
    family_label: str,
) -> np.ndarray:
    matrix = assignments.to_numpy(dtype=np.uint8, copy=False)
    hit_indices = tuple(
        np.flatnonzero(matrix[:, column]).astype(np.int64, copy=False)
        for column in range(matrix.shape[1])
    )
    weights = np.abs(scores.to_numpy(dtype=np.float64, copy=False))
    seeds = np.random.SeedSequence(random_seed).generate_state(
        n_permutations, dtype=np.uint32
    )

    n_workers = min(max(1, int(n_processes)), n_permutations)
    chunk_size = max(1, min(128, int(np.ceil(n_permutations / (n_workers * 20)))))
    chunks = tuple(
        seeds[start : start + chunk_size]
        for start in range(0, n_permutations, chunk_size)
    )
    description = f" Performing permutations ({family_label})"
    outputs: list[np.ndarray] = []

    if n_workers == 1:
        _init_null_worker(weights, hit_indices)
        progress = tqdm(
            total=n_permutations, desc=description, ncols=80, leave=False
        )
        try:
            for chunk in chunks:
                output = _null_ks_chunk(chunk)
                outputs.append(output)
                progress.update(len(chunk))
        finally:
            progress.close()
    else:
        with mp.Pool(
            processes=n_workers,
            initializer=_init_null_worker,
            initargs=(weights, hit_indices),
        ) as pool:
            progress = tqdm(
                total=n_permutations, desc=description, ncols=80, leave=False
            )
            try:
                for chunk, output in zip(
                    chunks, pool.imap(_null_ks_chunk, chunks, chunksize=1)
                ):
                    outputs.append(output)
                    progress.update(len(chunk))
            finally:
                progress.close()

    return np.concatenate(outputs, axis=0)


def _observed_ks(assignments: pd.DataFrame, scores: pd.Series) -> np.ndarray:
    matrix = assignments.to_numpy(dtype=np.uint8, copy=False)
    weights = np.abs(scores.to_numpy(dtype=np.float64, copy=False))
    n_sites = len(weights)
    result = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        positions = np.flatnonzero(matrix[:, column]).astype(np.int64, copy=False)
        result[column] = _null_ks_from_positions(
            positions,
            weights,
            np.arange(len(positions), dtype=np.float64),
            float(n_sites - len(positions)),
        )
    return result


class PhosXSession:
    """Read-only resources and sequence caches shared across contrasts."""

    def __init__(
        self,
        imported: dict[str, Any],
        resources: dict[str, Path],
    ) -> None:
        self.imported = imported
        self.resources = resources
        self._families: dict[str, _FamilyResources] = {}
        self._kinase_metadata: dict[str, Any] | None = None
        self._activation_adjacency: dict[
            tuple[tuple[str, ...], float, float], np.ndarray
        ] = {}

    def family(self, family: str) -> _FamilyResources:
        if family in self._families:
            return self._families[family]
        if family == "S/T":
            pssm_path = self.resources["s_t_pssm"]
            quantile_path = self.resources["s_t_quantiles"]
        elif family == "Y":
            pssm_path = self.resources["y_pssm"]
            quantile_path = self.resources["y_quantiles"]
        else:
            raise ValueError(f"Unknown PhosX kinase family: {family!r}.")

        pssm_dict = self.imported["read_pssms"](str(pssm_path))
        background = self.imported["read_pssm_score_quantiles"](
            str(quantile_path)
        )
        names = tuple(pssm_dict.keys())
        if not names:
            raise ValueError(f"PhosX {family} PSSM resource contains no kinases.")
        amino_acids = tuple(str(value) for value in pssm_dict[names[0]].columns)
        pssm = np.stack(
            [pssm_dict[name].to_numpy(dtype=np.float64) for name in names], axis=0
        )
        sorted_background = tuple(
            np.sort(background[name].to_numpy(dtype=np.float64)) for name in names
        )
        loaded = _FamilyResources(
            names=names,
            amino_acids=amino_acids,
            pssm=pssm,
            sorted_background=sorted_background,
        )
        self._families[family] = loaded
        return loaded

    @staticmethod
    def _score_windows(
        sequences: list[str],
        family: _FamilyResources,
        *,
        take_max: bool,
    ) -> np.ndarray:
        amino_index = {
            amino_acid: index
            for index, amino_acid in enumerate(family.amino_acids)
        }
        n_kinases, pssm_length, _ = family.pssm.shape

        def score_window(window: str) -> np.ndarray:
            if len(window) != pssm_length:
                raise ValueError(
                    f"PhosX expected a {pssm_length}-residue window; received "
                    f"{window!r}."
                )
            score = np.ones(n_kinases, dtype=np.float64)
            for position, residue in enumerate(window):
                if residue == "_":
                    continue
                residue_index = amino_index.get(residue)
                if residue_index is None:
                    raise ValueError(
                        f"PhosX cannot score non-canonical residue {residue!r} "
                        f"in sequence {window!r}."
                    )
                score *= family.pssm[:, position, residue_index]
            return score

        output = np.empty((len(sequences), n_kinases), dtype=np.float64)
        for row, sequence in enumerate(sequences):
            if take_max:
                if len(sequence) < pssm_length:
                    raise ValueError(
                        "PhosX activation-loop sequence cannot be shorter than "
                        f"{pssm_length} residues: {sequence!r}."
                    )
                best = np.full(n_kinases, -np.inf, dtype=np.float64)
                for start in range(len(sequence) - pssm_length + 1):
                    best = np.maximum(
                        best, score_window(sequence[start : start + pssm_length])
                    )
                output[row] = best
            else:
                output[row] = score_window(sequence)
        return output

    @staticmethod
    def _quantile_scale(
        raw_scores: np.ndarray, family: _FamilyResources
    ) -> np.ndarray:
        scaled = np.empty_like(raw_scores, dtype=np.float64)
        for column, background in enumerate(family.sorted_background):
            scaled[:, column] = (
                np.searchsorted(background, raw_scores[:, column], side="right")
                / len(background)
            )
        return scaled

    def assignments(
        self,
        sequences: list[str],
        *,
        family_name: str,
        n_top_kinases: int,
        min_quantile: float,
    ) -> tuple[pd.DataFrame, int]:
        family = self.family(family_name)
        if not sequences:
            return pd.DataFrame(columns=family.names, dtype=np.uint8), 0
        cache_key = (int(n_top_kinases), float(min_quantile))
        cache = family.assignment_cache.setdefault(cache_key, {})
        unique_sequences = list(dict.fromkeys(sequences))
        missing = [sequence for sequence in unique_sequences if sequence not in cache]

        if missing:
            raw = self._score_windows(missing, family, take_max=False)
            scaled = self._quantile_scale(raw, family)
            if scaled.shape[1] <= n_top_kinases:
                raise ValueError(
                    f"PhosX {family_name} has only {scaled.shape[1]} PSSMs but "
                    f"n_top_kinases={n_top_kinases}."
                )
            kth = scaled.shape[1] - n_top_kinases - 1
            thresholds = np.partition(scaled, kth, axis=1)[:, kth]
            thresholds = np.maximum(thresholds, min_quantile)
            binary = scaled > thresholds[:, None]
            for sequence, values in zip(missing, binary):
                cache[sequence] = values.astype(np.uint8, copy=True)

        matrix = np.vstack([cache[sequence] for sequence in sequences])
        return pd.DataFrame(matrix, columns=family.names), len(missing)

    def activation_adjacency(
        self,
        kinase_names: list[str],
        *,
        s_t_threshold: float,
        y_threshold: float,
    ) -> tuple[np.ndarray, bool]:
        key = (tuple(kinase_names), float(s_t_threshold), float(y_threshold))
        cached = self._activation_adjacency.get(key)
        if cached is not None:
            return cached, True

        if self._kinase_metadata is None:
            self._kinase_metadata = self.imported["hdf5_to_dict"](
                str(self.resources["kinase_metadata"])
            )
        aloop_sequences = self._kinase_metadata["aloop_seq"]
        kinase_position = {name: index for index, name in enumerate(kinase_names)}
        target_names = [
            name
            for name in kinase_names
            if name in aloop_sequences and pd.notna(aloop_sequences[name])
        ]
        target_sequences = [str(aloop_sequences[name]) for name in target_names]
        adjacency = np.zeros(
            (len(kinase_names), len(kinase_names)), dtype=np.uint8
        )

        for family_name, threshold in (
            ("S/T", s_t_threshold),
            ("Y", y_threshold),
        ):
            family = self.family(family_name)
            source_names = [name for name in family.names if name in kinase_position]
            if not source_names or not target_names:
                continue
            source_indices = [family.names.index(name) for name in source_names]
            raw = self._score_windows(target_sequences, family, take_max=True)
            scaled = self._quantile_scale(raw, family)
            binary = scaled[:, source_indices] > threshold
            target_positions = [kinase_position[name] for name in target_names]
            source_positions = [kinase_position[name] for name in source_names]
            adjacency[np.ix_(target_positions, source_positions)] |= binary.astype(
                np.uint8
            )

        self._activation_adjacency[key] = adjacency
        return adjacency, False


def create_phosx_session(
    resource_overrides: dict[str, Any] | None = None,
) -> PhosXSession:
    imported = _imports()
    resources = _resource_paths(imported["package"], resource_overrides)
    return PhosXSession(imported, resources)


def _empty_family_result(names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "KS": np.nan,
            "p value": np.nan,
            "FDR q value": np.nan,
            "Activity Score": 0.0,
        },
        index=list(names),
    )


def _compute_family(
    seqrnk: pd.DataFrame,
    *,
    session: PhosXSession,
    family_name: str,
    n_permutations: int,
    random_seed: int,
    n_processes: int,
    n_top_kinases: int,
    min_n_hits: int,
    min_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = seqrnk.loc[
        seqrnk["Score"].ne(0) & seqrnk[["Sequence", "Score"]].notna().all(axis=1)
    ].copy()
    if family_name == "S/T":
        ranked = ranked.loc[ranked["Sequence"].str[5].isin(["S", "T"])]
    else:
        ranked = ranked.loc[ranked["Sequence"].str[5].eq("Y")]
    ranked.sort_values(by="Score", ascending=False, inplace=True, ignore_index=True)
    family = session.family(family_name)

    if ranked.empty:
        return (
            _empty_family_result(family.names),
            pd.DataFrame(columns=family.names),
        )

    all_assignments, _ = session.assignments(
        ranked["Sequence"].astype(str).tolist(),
        family_name=family_name,
        n_top_kinases=n_top_kinases,
        min_quantile=min_quantile,
    )
    tested_assignments = all_assignments.loc[
        :, all_assignments.sum(axis=0) >= min_n_hits
    ]
    if tested_assignments.shape[1] == 0:
        return _empty_family_result(family.names), all_assignments

    empirical = _fast_empirical_distribution(
        tested_assignments,
        ranked["Score"],
        n_permutations=n_permutations,
        n_processes=n_processes,
        random_seed=random_seed,
        family_label=family_name,
    )
    observed = np.round(
        _observed_ks(tested_assignments, ranked["Score"]), decimals=5
    )
    more_extreme = np.where(
        observed[None, :] > 0,
        empirical > observed[None, :],
        empirical < observed[None, :],
    )
    p_value_decimals = int(np.ceil(np.log10(n_permutations + 1)))
    pvalues = np.round(
        (more_extreme.sum(axis=0) + 1) / (n_permutations + 1),
        decimals=p_value_decimals,
    )
    results = pd.DataFrame(
        {"KS": observed, "p value": pvalues}, index=tested_assignments.columns
    )
    results["FDR q value"] = session.imported["benjamini_hochberg"](
        results["p value"]
    ).round(decimals=p_value_decimals)
    activity_score = np.minimum(
        -np.log10(results["p value"]), np.log10(n_permutations + 1)
    )
    results["Activity Score"] = (
        activity_score * np.sign(results["KS"])
    ).round(decimals=5)
    missing = [name for name in family.names if name not in results.index]
    if missing:
        results = pd.concat([results, _empty_family_result(tuple(missing))])
    # A stable family order also makes the activation-loop adjacency cache
    # reusable when the set of tested kinases differs between contrasts.
    results = results.reindex(family.names)
    return results, all_assignments


def _aligned_assignment_matrix(
    assignments: pd.DataFrame, kinase_names: list[str]
) -> np.ndarray:
    matrix = np.zeros((len(assignments), len(kinase_names)), dtype=np.float32)
    if assignments.empty:
        return matrix
    position = {name: index for index, name in enumerate(kinase_names)}
    common = [name for name in assignments.columns if name in position]
    if common:
        columns = [position[name] for name in common]
        matrix[:, columns] = assignments.loc[:, common].to_numpy(
            dtype=np.float32, copy=False
        )
    return matrix


def _substrate_redundancy(
    s_t_assignments: pd.DataFrame,
    y_assignments: pd.DataFrame,
    kinase_names: list[str],
) -> np.ndarray:
    assigned = np.vstack(
        [
            _aligned_assignment_matrix(s_t_assignments, kinase_names),
            _aligned_assignment_matrix(y_assignments, kinase_names),
        ]
    )
    intersections = assigned.T @ assigned
    counts = np.diag(intersections)
    unions = counts[:, None] + counts[None, :] - intersections
    redundancy = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions != 0,
    )
    np.fill_diagonal(redundancy, 0.0)
    return np.round(redundancy, decimals=5)


def _activation_direction(
    activity: pd.DataFrame,
    adjacency: np.ndarray,
    redundancy: np.ndarray,
    *,
    redundancy_threshold: float,
    decay_factor: float,
    upregulation: bool,
) -> pd.Series:
    kinase_names = activity.index.astype(str).tolist()
    original = activity["Activity Score"].to_numpy(dtype=np.float64, copy=True)
    target = original > 0 if upregulation else original < 0
    values = original if upregulation else -original
    values = values.copy()
    values[~target] = 0.01

    directed = adjacency.T.astype(np.float64, copy=True)
    np.fill_diagonal(directed, 0.0)
    upstream = np.max(directed * values[None, :], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        activity_ratio = np.round(values[:, None] / values[None, :], decimals=5)
        upstream_ratio = np.round(
            upstream[:, None] / upstream[None, :], decimals=5
        )
    upstream_ratio[~np.isfinite(upstream_ratio)] = 1.0
    upstream_ratio[upstream_ratio > 1] = 1.0

    competing = (redundancy >= redundancy_threshold) & (redundancy != 0)
    decay = np.round(
        np.exp(-(decay_factor * ((activity_ratio - 1) ** 2))), decimals=5
    )
    modifier_matrix = np.round(
        1 - (competing * decay * (1 - upstream_ratio)), decimals=5
    )
    modifiers = np.min(modifier_matrix, axis=1)
    modifiers[~target] = 1.0
    modified = values * modifiers
    if not upregulation:
        modified = -modified
    return pd.Series(
        modified[target],
        index=np.asarray(kinase_names, dtype=object)[target],
        name="Activity Score",
    )


def _apply_activation_evidence(
    activity: pd.DataFrame,
    s_t_assignments: pd.DataFrame,
    y_assignments: pd.DataFrame,
    *,
    session: PhosXSession,
    s_t_threshold: float,
    y_threshold: float,
    upreg_redundancy_threshold: float,
    downreg_redundancy_threshold: float,
    decay_factor: float,
) -> pd.DataFrame:
    kinase_names = activity.index.astype(str).tolist()

    adjacency, _ = session.activation_adjacency(
        kinase_names,
        s_t_threshold=s_t_threshold,
        y_threshold=y_threshold,
    )
    redundancy = _substrate_redundancy(
        s_t_assignments, y_assignments, kinase_names
    )
    upreg = _activation_direction(
        activity,
        adjacency,
        redundancy,
        redundancy_threshold=upreg_redundancy_threshold,
        decay_factor=decay_factor,
        upregulation=True,
    )
    downreg = _activation_direction(
        activity,
        adjacency,
        redundancy,
        redundancy_threshold=downreg_redundancy_threshold,
        decay_factor=decay_factor,
        upregulation=False,
    )
    output = activity.rename(columns={"Activity Score": "Legacy Activity Score"})
    output["Activity Score"] = output["Legacy Activity Score"]
    output.loc[upreg.index, "Activity Score"] = upreg
    output.loc[downreg.index, "Activity Score"] = downreg
    return output


def _run_phosx(
    seqrnk: pd.DataFrame,
    *,
    session: PhosXSession,
    n_permutations: int,
    random_seed: int,
    n_processes: int,
    ser_thr_only: bool,
    tyr_only: bool,
    s_t_n_top_kinases: int,
    y_n_top_kinases: int,
    min_n_hits: int,
    s_t_min_quantile: float,
    y_min_quantile: float,
    no_upstream_activation_evidence: bool,
    a_loop_s_t_quantile_threshold: float,
    a_loop_y_quantile_threshold: float,
    upreg_redundancy_threshold: float,
    downreg_redundancy_threshold: float,
    decay_factor: float,
) -> PhosXRun:
    if tyr_only:
        st_activity = pd.DataFrame()
        st_assignments = pd.DataFrame()
    else:
        st_activity, st_assignments = _compute_family(
            seqrnk,
            session=session,
            family_name="S/T",
            n_permutations=n_permutations,
            random_seed=_family_seed(random_seed, 0),
            n_processes=n_processes,
            n_top_kinases=s_t_n_top_kinases,
            min_n_hits=min_n_hits,
            min_quantile=s_t_min_quantile,
        )

    if ser_thr_only:
        y_activity = pd.DataFrame()
        y_assignments = pd.DataFrame()
    else:
        y_activity, y_assignments = _compute_family(
            seqrnk,
            session=session,
            family_name="Y",
            n_permutations=n_permutations,
            random_seed=_family_seed(random_seed, 1),
            n_processes=n_processes,
            n_top_kinases=y_n_top_kinases,
            min_n_hits=min_n_hits,
            min_quantile=y_min_quantile,
        )

    activity = session.imported["concat_keep_na"](st_activity, y_activity)
    if not no_upstream_activation_evidence:
        activity = _apply_activation_evidence(
            activity,
            st_assignments,
            y_assignments,
            session=session,
            s_t_threshold=a_loop_s_t_quantile_threshold,
            y_threshold=a_loop_y_quantile_threshold,
            upreg_redundancy_threshold=upreg_redundancy_threshold,
            downreg_redundancy_threshold=downreg_redundancy_threshold,
            decay_factor=decay_factor,
        )

    return PhosXRun(
        activity=activity,
        ser_thr_assignments=st_assignments,
        tyr_assignments=y_assignments,
        version=REQUIRED_PHOSX_VERSION,
    )


def run_phosx_in_memory(
    seqrnk: pd.DataFrame,
    *,
    n_permutations: int = 20000,
    random_seed: int = 1,
    n_processes: int = 1,
    ser_thr_only: bool = False,
    tyr_only: bool = False,
    s_t_n_top_kinases: int = 5,
    y_n_top_kinases: int = 5,
    min_n_hits: int = 4,
    s_t_min_quantile: float = 0.95,
    y_min_quantile: float = 0.90,
    no_upstream_activation_evidence: bool = False,
    a_loop_s_t_quantile_threshold: float = 0.95,
    a_loop_y_quantile_threshold: float = 0.95,
    upreg_redundancy_threshold: float = 0.5,
    downreg_redundancy_threshold: float = 0.5,
    decay_factor: float = 256.0,
    resource_overrides: dict[str, Any] | None = None,
    session: PhosXSession | None = None,
) -> PhosXRun:
    """Run the pinned PhosX equations without temporary files."""
    if list(seqrnk.columns) != ["Sequence", "Score"]:
        raise ValueError("PhosX input must contain exactly ['Sequence', 'Score'].")
    active_session = session or create_phosx_session(resource_overrides)
    return _run_phosx(
        seqrnk,
        session=active_session,
        n_permutations=n_permutations,
        random_seed=random_seed,
        n_processes=n_processes,
        ser_thr_only=ser_thr_only,
        tyr_only=tyr_only,
        s_t_n_top_kinases=s_t_n_top_kinases,
        y_n_top_kinases=y_n_top_kinases,
        min_n_hits=min_n_hits,
        s_t_min_quantile=s_t_min_quantile,
        y_min_quantile=y_min_quantile,
        no_upstream_activation_evidence=no_upstream_activation_evidence,
        a_loop_s_t_quantile_threshold=a_loop_s_t_quantile_threshold,
        a_loop_y_quantile_threshold=a_loop_y_quantile_threshold,
        upreg_redundancy_threshold=upreg_redundancy_threshold,
        downreg_redundancy_threshold=downreg_redundancy_threshold,
        decay_factor=decay_factor,
    )
