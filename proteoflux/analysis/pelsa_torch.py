"""Batched torch backend for PELSA 4PL fitting."""

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from proteoflux.utils.utils import log_info


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PELSA torch backend requires torch. Install torch or use optimizer.backend='scipy'."
        ) from e
    return torch


def _logit01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _to_unconstrained(p: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    z = (p - lower) / (upper - lower)
    return _logit01(z)


def _torch_4pl(torch, x, pec50, slope, front, back):
    z = torch.pow(10.0, slope * (x + pec50))
    return back + (front - back) / (1.0 + z)


def _build_dense_torch_input(ratio_df: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    nonzero = ratio_df["concentration"].to_numpy(dtype=float) > 0
    finite = (
        nonzero
        & np.isfinite(ratio_df["log10_concentration"].to_numpy(dtype=float))
        & np.isfinite(ratio_df["ratio"].to_numpy(dtype=float))
        & (ratio_df["ratio"].to_numpy(dtype=float) > 0)
    )

    df = ratio_df.loc[finite, ["peptide_id", "sample", "log10_concentration", "ratio"]].copy()
    if df.empty:
        raise ValueError("PELSA torch backend received no valid nonzero ratio points.")

    peptides = pd.Index(ratio_df["peptide_id"].drop_duplicates().astype(str))
    samples = pd.Index(df["sample"].drop_duplicates().astype(str))

    x_by_sample = (
        df.drop_duplicates("sample")
        .set_index("sample")
        .loc[samples, "log10_concentration"]
        .to_numpy(dtype=float)
    )

    y = np.full((len(peptides), len(samples)), np.nan, dtype=np.float32)

    p_codes = pd.Categorical(df["peptide_id"].astype(str), categories=peptides).codes
    s_codes = pd.Categorical(df["sample"].astype(str), categories=samples).codes
    y[p_codes, s_codes] = df["ratio"].to_numpy(dtype=np.float32)

    valid = np.isfinite(y) & (y > 0)
    x_levels = np.unique(x_by_sample[np.isfinite(x_by_sample)])

    n_conc = np.zeros(y.shape[0], dtype=np.int16)
    for xv in x_levels:
        n_conc += valid[:, x_by_sample == xv].any(axis=1)

    return peptides, x_by_sample.astype(np.float32), y, n_conc

def _sanitize_curve_results_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    string_cols = ["peptide_id", "failure_reason", "optimizer_backend"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    bool_cols = ["fit_success", "pEC50_inside_range"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = pd.Series(df[col], dtype="boolean").fillna(False).astype(bool)

    int_cols = ["n_points", "n_concentrations", "optimizer_steps", "optimizer_n_starts", "optimizer_best_start"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

    return df

def fit_4pl_torch_from_ratio_df(ratio_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    torch = _require_torch()

    pelsa_cfg = ((config or {}).get("analysis", {}) or {}).get("pelsa", {}) or {}
    bounds_cfg = pelsa_cfg.get("bounds", {}) or {}

    device = "cpu"
    dtype_name = "float32"
    dtype = torch.float32

    steps = int(pelsa_cfg.get("torch_steps", 300))
    lr = float(pelsa_cfg.get("torch_lr", 0.03))
    batch_size = int(pelsa_cfg.get("torch_batch_size", 4096))
    n_starts = int(pelsa_cfg.get("torch_n_starts", 1))
    if n_starts < 1:
        raise ValueError("PELSA torch_n_starts must be >= 1.")

    pec50_margin = float(bounds_cfg.get("pec50_margin_log10", 2.0))
    slope_lo, slope_hi = bounds_cfg.get("slope", [0.01, 10.0])
    front_lo, front_hi = bounds_cfg.get("front", [1e-4, 1e6])
    back_lo, back_hi = bounds_cfg.get("back", [1e-4, 1e6])

    t0 = perf_counter()
    peptides, x_np, y_np, n_conc = _build_dense_torch_input(ratio_df)
    valid_np = np.isfinite(y_np) & (y_np > 0)
    n_points = valid_np.sum(axis=1).astype(int)
    eligible = n_conc >= 4

    log_info(
        "PELSA torch input: "
        f"features={len(peptides)}, eligible={int(eligible.sum())}, "
        f"samples_nonzero={len(x_np)}, build_time={perf_counter() - t0:.2f}s"
    )

    result = pd.DataFrame(
        {
            "peptide_id": peptides.to_numpy(dtype=str),
            "fit_success": False,
            "failure_reason": np.where(eligible, "not_fitted", "insufficient_nonzero_points"),
            "n_points": n_points,
            "n_concentrations": n_conc.astype(int),
        }
    )

    x_min = float(np.nanmin(x_np))
    x_max = float(np.nanmax(x_np))
    pec50_lo = float(np.min(-x_np) - pec50_margin)
    pec50_hi = float(np.max(-x_np) + pec50_margin)

    lower_np = np.array([pec50_lo, slope_lo, front_lo, back_lo], dtype=np.float32)
    upper_np = np.array([pec50_hi, slope_hi, front_hi, back_hi], dtype=np.float32)

    eligible_idx = np.flatnonzero(eligible)
    x = torch.tensor(x_np, device=device, dtype=dtype)[None, :]

    lower = torch.tensor(lower_np, device=device, dtype=dtype)[None, :]
    upper = torch.tensor(upper_np, device=device, dtype=dtype)[None, :]

    out_rows = []

    t_fit = perf_counter()
    for start in tqdm(
        range(0, len(eligible_idx), batch_size),
        desc="PELSA torch 4PL fitting",
    ):
        idx = eligible_idx[start : start + batch_size]
        y_batch_np = y_np[idx, :]
        mask_np = np.isfinite(y_batch_np) & (y_batch_np > 0)

        y = torch.tensor(np.nan_to_num(y_batch_np, nan=0.0), device=device, dtype=dtype)
        mask = torch.tensor(mask_np.astype(np.float32), device=device, dtype=dtype)

        y_mean = np.nanmean(np.where(mask_np, y_batch_np, np.nan), axis=1)
        y_mean = np.where(np.isfinite(y_mean), y_mean, 1.0)

        low_mask = x_np == x_min
        high_mask = x_np == x_max

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            front0 = np.nanmean(
                    np.where(mask_np[:, low_mask], y_batch_np[:, low_mask], np.nan),
                    axis=1,
            )
            back0 = np.nanmean(
                    np.where(mask_np[:, high_mask], y_batch_np[:, high_mask], np.nan),
                    axis=1,
            )

        front0 = np.where(np.isfinite(front0), front0, y_mean)
        back0 = np.where(np.isfinite(back0), back0, y_mean)

        base_p0 = np.column_stack(
            [
                np.full(len(idx), float(np.nanmedian(-x_np)), dtype=np.float32),
                np.full(len(idx), 1.0, dtype=np.float32),
                front0.astype(np.float32),
                back0.astype(np.float32),
            ]
        )

        p0_list = [base_p0]
        if n_starts >= 2:
            p0_list.append(
                np.column_stack(
                    [
                        np.full(len(idx), float(np.nanmin(-x_np)), dtype=np.float32),
                        np.full(len(idx), 1.0, dtype=np.float32),
                        front0.astype(np.float32),
                        back0.astype(np.float32),
                    ]
                )
            )
        if n_starts >= 3:
            p0_list.append(
                np.column_stack(
                    [
                        np.full(len(idx), float(np.nanmax(-x_np)), dtype=np.float32),
                        np.full(len(idx), 1.0, dtype=np.float32),
                        front0.astype(np.float32),
                        back0.astype(np.float32),
                    ]
                )
            )
        if n_starts >= 4:
            p0_list.append(
                np.column_stack(
                    [
                        np.full(len(idx), float(np.nanmedian(-x_np)), dtype=np.float32),
                        np.full(len(idx), 3.0, dtype=np.float32),
                        front0.astype(np.float32),
                        back0.astype(np.float32),
                    ]
                )
            )
        if n_starts >= 5:
            p0_list.append(
                np.column_stack(
                    [
                        np.full(len(idx), float(np.nanmedian(-x_np)), dtype=np.float32),
                        np.full(len(idx), 1.0, dtype=np.float32),
                        back0.astype(np.float32),
                        front0.astype(np.float32),
                    ]
                )
            )

        p0 = np.stack(p0_list[:n_starts], axis=1)  # curves × starts × params
        p0 = np.clip(p0, lower_np + 1e-8, upper_np - 1e-8)

        u0 = _to_unconstrained(p0, lower_np, upper_np)
        u = torch.tensor(u0, device=device, dtype=dtype, requires_grad=True)

        opt = torch.optim.Adam([u], lr=lr)

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            p = lower[:, None, :] + (upper[:, None, :] - lower[:, None, :]) * torch.sigmoid(u)

            pred = _torch_4pl(
                torch,
                x[:, None, :],
                p[:, :, 0:1],
                p[:, :, 1:2],
                p[:, :, 2:3],
                p[:, :, 3:4],
            )

            resid = (pred - y[:, None, :]) * mask[:, None, :]
            loss = torch.sum(resid * resid) / torch.clamp(mask.sum(), min=1.0)
            loss.backward()
            opt.step()

        with torch.no_grad():
            p_all = lower[:, None, :] + (upper[:, None, :] - lower[:, None, :]) * torch.sigmoid(u)
            pred = _torch_4pl(
                torch,
                x[:, None, :],
                p_all[:, :, 0:1],
                p_all[:, :, 1:2],
                p_all[:, :, 2:3],
                p_all[:, :, 3:4],
            )
            resid = (pred - y[:, None, :]) * mask[:, None, :]
            sse_all = torch.sum(resid * resid, dim=2)
            best_start = torch.argmin(sse_all, dim=1)

            row_ix = torch.arange(p_all.shape[0], device=device)
            p = p_all[row_ix, best_start, :]
            sse_curve = sse_all[row_ix, best_start]

            pred_best = pred[row_ix, best_start, :]
            resid_best = (pred_best - y) * mask

            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            y_bar = torch.sum(y * mask, dim=1, keepdim=True) / denom[:, None]
            sse_null = torch.sum(((y - y_bar) * mask) ** 2, dim=1)

            rmse = torch.sqrt(sse_curve / denom)
            r2 = 1.0 - sse_curve / torch.clamp(sse_null, min=1e-20) #clamp min to 0 ? hmmm

            y_low = _torch_4pl(torch, torch.tensor([[x_min]], device=device, dtype=dtype), p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4])[:, 0]
            y_high = _torch_4pl(torch, torch.tensor([[x_max]], device=device, dtype=dtype), p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4])[:, 0]
            curve_fc = torch.log2(y_high / y_low)

            p_np = p.detach().cpu().numpy()
            rows = pd.DataFrame(
                {
                    "row_idx": idx,
                    "fit_success": True,
                    "failure_reason": None,
                    "pec50": p_np[:, 0],
                    "slope": p_np[:, 1],
                    "front": p_np[:, 2],
                    "back": p_np[:, 3],
                    "sse_curve": sse_curve.cpu().numpy(),
                    "sse_null": sse_null.cpu().numpy(),
                    "rmse": rmse.cpu().numpy(),
                    "r2": r2.cpu().numpy(),
                    "curve_fold_change_log2": curve_fc.cpu().numpy(),
                    "pEC50_inside_range": (
                        (p_np[:, 0] >= np.min(-x_np)) & (p_np[:, 0] <= np.max(-x_np))
                    ),
                    "optimizer_backend": "torch_adam",
                    "optimizer_steps": steps,
                    "optimizer_n_starts": n_starts,
                    "optimizer_best_start": best_start.cpu().numpy(),
                }
            )
            out_rows.append(rows)

    if out_rows:
        fitted = pd.concat(out_rows, ignore_index=True).set_index("row_idx")
        for col in fitted.columns:
            result.loc[fitted.index, col] = fitted[col].to_numpy()

    # CurveCurator-style recalibrated F statistic.
    ok = result["fit_success"].to_numpy(dtype=bool)
    n = result["n_points"].to_numpy(dtype=float)
    k = 4.0
    sse0 = result["sse_null"].to_numpy(dtype=float)
    sse1 = result["sse_curve"].to_numpy(dtype=float)

    f_value = np.full(len(result), np.nan, dtype=float)
    p_value = np.full(len(result), np.nan, dtype=float)

    valid = ok & np.isfinite(sse0) & np.isfinite(sse1) & (sse1 > 0) & (n > k)
    f_value[valid] = ((sse0[valid] - sse1[valid]) / sse1[valid]) * (n[valid] / k)
    f_value[valid] = np.maximum(f_value[valid], 0.0)

    # Effective dof from CurveCurator-style 4PL calibration.
    def _low_n_slope_adjustment(nn):
        return 1.0 / (((nn - 4.0) ** 4) / nn + 4.0)

    dfn = 5.0
    dfd = (0.8 - _low_n_slope_adjustment(n[valid])) * (n[valid] - 2.5)
    p_value[valid] = stats.f.sf(f_value[valid], dfn=dfn, dfd=dfd, scale=1.0, loc=0.12)

    q_value = np.full(len(result), np.nan, dtype=float)
    p_ok = np.isfinite(p_value)
    if np.any(p_ok):
        q_value[p_ok] = multipletests(p_value[p_ok], method="fdr_bh")[1]

    result["curve_f_value"] = f_value
    result["curve_p_value"] = p_value
    result["curve_q_value"] = q_value
    result["curve_neglog10_q"] = -np.log10(np.clip(q_value, 1e-300, 1.0))

    log_info(
        f"PELSA torch 4PL fitting wall-time: {perf_counter() - t_fit:.2f}s "
        f"(batch_size={batch_size}, steps={steps}, lr={lr}, n_starts={n_starts})"
    )

    return _sanitize_curve_results_for_h5ad(result)
