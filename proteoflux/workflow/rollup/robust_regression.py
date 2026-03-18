from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
import scipy.linalg as la
from tqdm import tqdm

def _find_nameswitch_indices(arr: np.ndarray) -> np.ndarray:
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    start_indices = np.insert(change_indices, 0, 0)
    start_indices = np.append(start_indices, len(arr))
    return start_indices


def _get_configured_pool(num_cores: int | None) -> mp.pool.Pool:
    mp.freeze_support()
    if num_cores is None:
        num_cores = min(mp.cpu_count(), 60)
    return mp.Pool(num_cores)


def _sample_coef_col(sample_name: str) -> str:
    return f"C(sample)[{sample_name}]"


def _fit_one_protein_star(args):
    return _fit_one_protein(*args)


def _prune_design_full_rank(
    x: np.ndarray,
    colnames: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    MSqRobSum-like singular-column removal using pivoted QR.
    """
    n, p = x.shape
    if n == 0 or p == 0:
        return x[:, :0], []

    _, r, piv = la.qr(x, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return x[:, :0], []

    tol = np.finfo(float).eps * max(n, p) * diag.max()
    rank = int(np.sum(diag > tol))
    keep = sorted(piv[:rank].tolist())
    return x[:, keep], [colnames[i] for i in keep]


def _extract_observed_long_arrays(
    values: np.ndarray,
    sample_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    valid_mask = np.isfinite(values) & (values > 0)

    if not valid_mask.any():
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            [],
            {},
        )

    peptide_row_idx, sample_col_idx = np.nonzero(valid_mask)
    y_linear = values[peptide_row_idx, sample_col_idx].astype(float, copy=False)

    observed_sample_idx = np.flatnonzero(valid_mask.any(axis=0))
    observed_sample_names = [str(sample_names[j]) for j in observed_sample_idx]
    sample_codes = np.searchsorted(observed_sample_idx, sample_col_idx).astype(
        np.int32,
        copy=False,
    )

    observed_peptide_idx = np.flatnonzero(valid_mask.any(axis=1))
    peptide_codes = np.searchsorted(observed_peptide_idx, peptide_row_idx).astype(
        np.int32,
        copy=False,
    )

    sample_count_arr = np.bincount(sample_codes, minlength=len(observed_sample_names))
    sample_counts = {
        observed_sample_names[i]: int(sample_count_arr[i])
        for i in range(len(observed_sample_names))
    }

    return y_linear, sample_codes, peptide_codes, observed_sample_names, sample_counts


def _build_sum_coded_design(
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    observed_sample_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Same model space as:
        0 + C(sample) + C(peptide, Sum)
    """
    n_obs = int(sample_codes.size)
    n_samples = int(len(observed_sample_names))
    n_peptides = int(peptide_codes.max()) + 1 if peptide_codes.size else 0

    sample_block = np.zeros((n_obs, n_samples), dtype=float)
    sample_block[np.arange(n_obs), sample_codes] = 1.0
    sample_colnames = [_sample_coef_col(s) for s in observed_sample_names]

    if n_peptides <= 1:
        return sample_block, sample_colnames

    pep_block = np.zeros((n_obs, n_peptides - 1), dtype=float)
    last_code = n_peptides - 1

    non_last = peptide_codes < last_code
    if np.any(non_last):
        pep_block[np.where(non_last)[0], peptide_codes[non_last]] = 1.0

    last_mask = peptide_codes == last_code
    if np.any(last_mask):
        pep_block[last_mask, :] = -1.0

    pep_colnames = [f"C(peptide, Sum)[S.{i}]" for i in range(n_peptides - 1)]
    x = np.concatenate([sample_block, pep_block], axis=1)
    return x, sample_colnames + pep_colnames


def _mad_scale(resid: np.ndarray) -> float:
    if resid.size == 0:
        return 1.0
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.sqrt(np.mean(resid * resid)) if resid.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(scale)


def _huber_weights(resid: np.ndarray, scale: float, t: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        return np.ones_like(resid, dtype=float)

    u = resid / scale
    w = np.ones_like(u, dtype=float)
    mask = np.abs(u) > t
    w[mask] = t / np.abs(u[mask])
    return w


def _solve_weighted_ls_normal_eq(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    wx = x * w[:, None]
    xtwx = x.T @ wx
    xtwy = x.T @ (w * y)

    try:
        return np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        sqrt_w = np.sqrt(w)
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w
        beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
        return beta


def _is_connected_observation_graph(
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    n_samples: int,
    n_peptides: int,
) -> bool:
    """
    Connectivity of the bipartite graph:
      sample nodes 0..n_samples-1
      peptide nodes n_samples..n_samples+n_peptides-1

    For the additive sample+peptide model, disconnected components are the
    main source of non-identifiability beyond the usual constraint.
    """
    if n_samples == 0 or n_peptides == 0:
        return False

    n_nodes = n_samples + n_peptides
    parent = np.arange(n_nodes, dtype=np.int32)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for s, p in zip(sample_codes.tolist(), peptide_codes.tolist(), strict=False):
        union(int(s), n_samples + int(p))

    roots = set()
    active_samples = np.unique(sample_codes)
    active_peptides = np.unique(peptide_codes)

    for s in active_samples.tolist():
        roots.add(find(int(s)))
    for p in active_peptides.tolist():
        roots.add(find(n_samples + int(p)))

    return len(roots) == 1


def _structured_predict(
    beta: np.ndarray,
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    n_samples: int,
    n_peptides: int,
) -> np.ndarray:
    pred = beta[sample_codes].copy()

    if n_peptides <= 1:
        return pred

    pep_beta = beta[n_samples:]
    last_code = n_peptides - 1

    non_last = peptide_codes < last_code
    if np.any(non_last):
        pred[non_last] += pep_beta[peptide_codes[non_last]]

    last_mask = peptide_codes == last_code
    if np.any(last_mask):
        pred[last_mask] -= np.sum(pep_beta)

    return pred


def _structured_xtwx_xtwy(
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_samples: int,
    n_peptides: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X^T W X and X^T W y directly from codes for the model:
        0 + sample + peptide(sum)
    without materializing X.

    Parameter order:
      [sample effects | peptide sum-contrast effects]
    where peptide block has n_peptides - 1 columns.
    """
    p_pep = max(0, n_peptides - 1)
    p = n_samples + p_pep

    xtwx = np.zeros((p, p), dtype=float)
    xtwy = np.zeros(p, dtype=float)

    wy = w * y

    # Sample block
    sample_diag = np.bincount(sample_codes, weights=w, minlength=n_samples)
    xtwx[np.arange(n_samples), np.arange(n_samples)] = sample_diag
    xtwy[:n_samples] = np.bincount(sample_codes, weights=wy, minlength=n_samples)

    if p_pep == 0:
        return xtwx, xtwy

    last_code = n_peptides - 1
    reg_mask = peptide_codes < last_code
    last_mask = peptide_codes == last_code

    reg_s = sample_codes[reg_mask]
    reg_p = peptide_codes[reg_mask]
    reg_w = w[reg_mask]
    reg_wy = wy[reg_mask]

    # peptide RHS
    pep_rhs = np.bincount(reg_p, weights=reg_wy, minlength=p_pep)
    if np.any(last_mask):
        pep_rhs -= np.sum(wy[last_mask])
    xtwy[n_samples:] = pep_rhs

    # sample-peptide block
    sp = np.zeros((n_samples, p_pep), dtype=float)
    if reg_p.size:
        np.add.at(sp, (reg_s, reg_p), reg_w)

    if np.any(last_mask):
        last_s = sample_codes[last_mask]
        last_w_by_sample = np.bincount(last_s, weights=w[last_mask], minlength=n_samples)
        sp -= last_w_by_sample[:, None]

    xtwx[:n_samples, n_samples:] = sp
    xtwx[n_samples:, :n_samples] = sp.T

    # peptide-peptide block
    pep_block = np.diag(np.bincount(reg_p, weights=reg_w, minlength=p_pep))
    if np.any(last_mask):
        pep_block += np.sum(w[last_mask]) * np.ones((p_pep, p_pep), dtype=float)

    xtwx[n_samples:, n_samples:] = pep_block
    return xtwx, xtwy


def _solve_structured_weighted_ls(
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_samples: int,
    n_peptides: int,
) -> np.ndarray:
    xtwx, xtwy = _structured_xtwx_xtwy(
        sample_codes=sample_codes,
        peptide_codes=peptide_codes,
        y=y,
        w=w,
        n_samples=n_samples,
        n_peptides=n_peptides,
    )
    try:
        return np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        # Fallback to explicit path only if needed
        x, _ = _build_sum_coded_design(
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            observed_sample_names=[str(i) for i in range(n_samples)],
        )
        sqrt_w = np.sqrt(w)
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w
        beta, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
        return beta


def _fit_rlm_numpy_irls_explicit(
    x: np.ndarray,
    y: np.ndarray,
    huber_t: float,
    max_iter: int,
    tol: float,
) -> tuple[int, str, list[float]]:

    for it in range(1, max_iter + 1):
        resid = y - x @ beta
        scale = _mad_scale(resid)
        w = _huber_weights(resid, scale=scale, t=huber_t)

        beta_new = _solve_weighted_ls_normal_eq(x=x, y=y, w=w)

        denom = max(1.0, np.linalg.norm(beta))
        rel_change = np.linalg.norm(beta_new - beta) / denom
        beta = beta_new

        if rel_change < tol:
            return beta, it

    return beta, max_iter


def _fit_rlm_numpy_irls_structured(
    sample_codes: np.ndarray,
    peptide_codes: np.ndarray,
    y: np.ndarray,
    huber_t: float,
    max_iter: int,
    tol: float,
    n_samples: int,
    n_peptides: int,
) -> tuple[np.ndarray, int]:
    beta = _solve_structured_weighted_ls(
        sample_codes=sample_codes,
        peptide_codes=peptide_codes,
        y=y,
        w=np.ones_like(y, dtype=float),
        n_samples=n_samples,
        n_peptides=n_peptides,
    )

    for it in range(1, max_iter + 1):
        resid = y - _structured_predict(
            beta=beta,
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            n_samples=n_samples,
            n_peptides=n_peptides,
        )
        scale = _mad_scale(resid)
        w = _huber_weights(resid, scale=scale, t=huber_t)

        beta_new = _solve_structured_weighted_ls(
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            y=y,
            w=w,
            n_samples=n_samples,
            n_peptides=n_peptides,
        )

        denom = max(1.0, np.linalg.norm(beta))
        rel_change = np.linalg.norm(beta_new - beta) / denom
        beta = beta_new

        if rel_change < tol:
            return beta, it

    return beta, max_iter


def _fit_one_protein(
    idx: int,
    protein_id: str,
    values: np.ndarray,
    sample_cols: Sequence[str],
    huber_t: float,
    max_iter: int,
    tol: float,
    min_nonan: int,
) -> tuple[int, str, list[float]]:

    y_linear, sample_codes, peptide_codes, observed_sample_names, sample_counts = (
        _extract_observed_long_arrays(
            values=values,
            sample_names=sample_cols,
        )
    )

    if y_linear.size == 0:
        return idx, protein_id, [np.nan] * len(sample_cols)

    y = np.log2(y_linear)

    n_obs = float(y.size)
    n_samples = len(observed_sample_names)
    n_peptides = int(peptide_codes.max()) + 1 if peptide_codes.size else 0

    fast_path = False

    if (
        n_peptides > 0
        and _is_connected_observation_graph(
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            n_samples=n_samples,
            n_peptides=n_peptides,
        )
    ):
        fast_path = True
        colnames = (
            [_sample_coef_col(s) for s in observed_sample_names]
            + [f"C(peptide, Sum)[S.{i}]" for i in range(max(0, n_peptides - 1))]
        )
        x = None
    else:
        x, colnames = _build_sum_coded_design(
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            observed_sample_names=observed_sample_names,
        )
        x, colnames = _prune_design_full_rank(x, colnames)

        if x.shape[1] == 0:
            return idx, protein_id, [np.nan] * len(sample_cols)

    if fast_path:
        beta, _ = _fit_rlm_numpy_irls_structured(
            sample_codes=sample_codes,
            peptide_codes=peptide_codes,
            y=y,
            huber_t=huber_t,
            max_iter=max_iter,
            tol=tol,
            n_samples=n_samples,
            n_peptides=n_peptides,
        )
    else:
        assert x is not None
        beta, _ = _fit_rlm_numpy_irls_explicit(
            x=x,
            y=y,
            huber_t=huber_t,
            max_iter=max_iter,
            tol=tol,
        )

    beta_map = {name: float(beta[i]) for i, name in enumerate(colnames)}

    out_vals: list[float] = []
    for sample in sample_cols:
        if sample_counts.get(sample, 0) < min_nonan:
            out_vals.append(np.nan)
            continue

        col = _sample_coef_col(sample)
        if col not in beta_map:
            out_vals.append(np.nan)
            continue

        out_vals.append(float(2 ** beta_map[col]))
    return idx, protein_id, out_vals


def _run_sequential(
    items: list[tuple],
) -> list[tuple[int, str, list[float]]]:
    out: list[tuple[int, str, list[float]]] = []

    for args in tqdm(items, desc="Robust regression", file=sys.stdout, leave=False):
        out.append(_fit_one_protein(*args))

    return out


def _run_multiprocessing(
    items: list[tuple],
    num_cores: int | None,
) -> list[tuple[int, str, list[float]]]:
    if not items:
        return []

    if num_cores is None:
        n_workers = min(mp.cpu_count(), 60)
    else:
        n_workers = max(int(num_cores), 1)

    chunksize = max(1, min(64, len(items) // (n_workers * 8) if len(items) > n_workers else 1))

    with _get_configured_pool(num_cores) as pool:
        iterator = pool.imap_unordered(
            _fit_one_protein_star,
            items,
            chunksize=chunksize,
        )
        return list(
            tqdm(
                iterator,
                total=len(items),
                desc="Robust regression",
                file=sys.stdout,
                leave=False,
            )
        )

def pivot_df_robust_regression(
    *,
    pep_wide: pl.DataFrame,
    protein_col: str,
    peptide_col: str,
    sample_cols: Sequence[str],
    num_cores: int | None,
    huber_t: float,
    max_iter: int,
    tol: float,
    min_nonan: int,
) -> pl.DataFrame:
    required = {protein_col, peptide_col, *sample_cols}
    missing = sorted(required - set(pep_wide.columns))
    if missing:
        raise ValueError(
            f"robust_regression: missing required columns: {missing!r}"
        )

    if pep_wide.height == 0:
        return pl.DataFrame(
            {
                protein_col: [],
                **{c: [] for c in sample_cols},
            }
        )

    pep_wide_pd = (
        pep_wide.select([protein_col, peptide_col, *sample_cols])
        .to_pandas()
        .sort_values([protein_col, peptide_col], kind="stable")
        .reset_index(drop=True)
    )

    protein_ids = pep_wide_pd[protein_col].astype(str).to_numpy()
    value_matrix = pep_wide_pd.loc[:, list(sample_cols)].to_numpy(dtype=float, copy=False)

    switches = _find_nameswitch_indices(protein_ids)
    n_proteins = len(switches) - 1

    items = []
    sample_cols_list = list(sample_cols)
    for idx in range(n_proteins):
        start = int(switches[idx])
        stop = int(switches[idx + 1])
        protein_id = str(protein_ids[start])
        block_values = value_matrix[start:stop, :]
        items.append(
            (
                idx,
                protein_id,
                block_values,
                sample_cols_list,
                huber_t,
                max_iter,
                tol,
                min_nonan,
            )
        )

    if num_cores is not None and num_cores <= 1:
        results = _run_sequential(items)
    else:
        results = _run_multiprocessing(items, num_cores=num_cores)

    results.sort(key=lambda x: x[0])

    rows = []
    for _, protein_id, values in results:
        row = {protein_col: protein_id}
        row.update(dict(zip(sample_cols, values, strict=True)))
        rows.append(row)

    out = pl.DataFrame(rows)
    out = out.with_columns(
        [
            pl.when(pl.col(c).is_nan())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in sample_cols
        ]
    )

    return out
