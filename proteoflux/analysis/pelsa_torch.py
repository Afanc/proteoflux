"""Batched CPU torch backend for PELSA 4PL fitting in log2FC space."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from proteoflux.utils.utils import log_info

PARAM_COLUMNS = ("pec50", "slope", "front", "back")
N_4PL_PARAMS = 4


@dataclass(frozen=True)
class TorchFitConfig:
    """Runtime options for the torch 4PL optimizer."""

    steps: int = 300
    lr: float = 0.03
    batch_size: int = 4096
    n_starts: int = 1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    loss: str = "mse"
    dtype_name: str = "float32"
    huber_delta: float = 1.0
    transition_leverage_lambda: float = 0.05
    transition_leverage_eps: float = 1e-6
    num_threads: int | None = None


@dataclass(frozen=True)
class PelsaBounds:
    """Global and feature-wise bounds used for constrained fitting."""

    pec50_margin_log10: float = 0.0
    slope: tuple[float, float] = (0.01, 10.0)
    front: tuple[float, float] = (-15.0, 15.0)
    back: tuple[float, float] = (-15.0, 15.0)
    response_margin_fraction: float = 0.5


@dataclass(frozen=True)
class DenseTorchInput:
    """Dense peptide x sample matrix representation of valid log2FC responses."""

    peptides: pd.Index
    x: np.ndarray
    y: np.ndarray
    n_concentrations: np.ndarray


@dataclass(frozen=True)
class BatchBounds:
    lower: np.ndarray
    upper: np.ndarray
    response_low: np.ndarray
    response_high: np.ndarray


@dataclass(frozen=True)
class BatchInitialValues:
    front0: np.ndarray
    back0: np.ndarray


class FourPLTorchModel:
    """Constrained 4PL model using an unconstrained torch parameter tensor."""

    def __init__(self, torch, lower, upper, initial_values):
        self.torch = torch
        self.lower = lower
        self.upper = upper
        self.u = torch.tensor(
            initial_values,
            device=lower.device,
            dtype=lower.dtype,
            requires_grad=True,
        )

    def parameters(self):
        return [self.u]

    def constrained_parameters(self):
        return self.lower[:, None, :] + (
            self.upper[:, None, :] - self.lower[:, None, :]
        ) * self.torch.sigmoid(self.u)

    def predict(self, x):
        p = self.constrained_parameters()
        return four_pl_torch(
            self.torch,
            x[:, None, :],
            p[:, :, 0:1],
            p[:, :, 1:2],
            p[:, :, 2:3],
            p[:, :, 3:4],
        )


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PELSA torch backend requires torch. Install torch or use "
            "optimizer.backend='scipy'."
        ) from exc
    return torch


def four_pl_numpy(x, pec50, slope, front, back):
    x = np.asarray(x, dtype=float)
    return back + (front - back) / (1.0 + np.power(10.0, slope * (x + pec50)))


def four_pl_jacobian_numpy(x, pec50, slope, front, back):
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = np.power(10.0, slope * (x + pec50))
        denom = 1.0 + z
        denom2 = denom * denom

        d_pec50 = np.log(10.0) * (back - front) * slope * z / denom2
        d_slope = np.log(10.0) * (back - front) * (x + pec50) * z / denom2
        d_front = 1.0 / denom
        d_back = 1.0 - d_front

        jac = np.vstack([d_pec50, d_slope, d_front, d_back]).T
        jac[~np.isfinite(jac)] = np.nan
        return jac


def four_pl_torch(torch, x, pec50, slope, front, back):
    z = torch.pow(10.0, slope * (x + pec50))
    return back + (front - back) / (1.0 + z)


def estimate_pec50_ci_batch(
    *,
    x_np: np.ndarray,
    y_batch_np: np.ndarray,
    mask_np: np.ndarray,
    params_np: np.ndarray,
    n_params: int = N_4PL_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate local 95% CI for pEC50 from nonlinear LS covariance."""
    n = params_np.shape[0]
    se = np.full(n, np.nan, dtype=float)
    ci_low = np.full(n, np.nan, dtype=float)
    ci_high = np.full(n, np.nan, dtype=float)
    ci_width_norm = np.full(n, np.nan, dtype=float)

    x_span = float(np.nanmax(x_np) - np.nanmin(x_np))
    if not np.isfinite(x_span) or x_span <= 0:
        return se, ci_low, ci_high, ci_width_norm

    for i in range(n):
        keep = mask_np[i]
        n_obs = int(np.sum(keep))
        dof = n_obs - n_params
        if dof <= 0:
            continue

        p = params_np[i].astype(float)
        if not np.isfinite(p).all():
            continue

        x_i = x_np[keep]
        y_i = y_batch_np[i, keep].astype(float)
        y_hat = four_pl_numpy(x_i, *p)
        resid = y_i - y_hat
        sigma2 = float(np.nansum(resid * resid)) / dof

        jac = four_pl_jacobian_numpy(x_i, *p)
        if not np.isfinite(jac).all():
            continue

        try:
            cov = sigma2 * np.linalg.pinv(jac.T @ jac)
        except np.linalg.LinAlgError:
            continue

        var_pec50 = float(cov[0, 0])
        if not np.isfinite(var_pec50) or var_pec50 < 0:
            continue

        se_i = float(np.sqrt(var_pec50))
        half_width = float(stats.t.ppf(0.975, dof)) * se_i

        se[i] = se_i
        ci_low[i] = p[0] - half_width
        ci_high[i] = p[0] + half_width
        ci_width_norm[i] = (2.0 * half_width) / x_span

    return se, ci_low, ci_high, ci_width_norm


def parse_torch_fit_config(pelsa_cfg: dict) -> TorchFitConfig:
    betas = tuple(pelsa_cfg.get("torch_betas", (0.9, 0.999)))
    if len(betas) != 2:
        raise ValueError("PELSA torch_betas must contain exactly two values.")

    cfg = TorchFitConfig(
        steps=int(pelsa_cfg.get("torch_steps", TorchFitConfig.steps)),
        lr=float(pelsa_cfg.get("torch_lr", TorchFitConfig.lr)),
        batch_size=int(pelsa_cfg.get("torch_batch_size", TorchFitConfig.batch_size)),
        n_starts=int(pelsa_cfg.get("torch_n_starts", TorchFitConfig.n_starts)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(pelsa_cfg.get("torch_eps", TorchFitConfig.eps)),
        weight_decay=float(
            pelsa_cfg.get("torch_weight_decay", TorchFitConfig.weight_decay)
        ),
        loss=str(pelsa_cfg.get("torch_loss", TorchFitConfig.loss)).lower(),
        dtype_name=str(pelsa_cfg.get("torch_dtype", TorchFitConfig.dtype_name)),
        huber_delta=float(
            pelsa_cfg.get("torch_huber_delta", TorchFitConfig.huber_delta)
        ),
        transition_leverage_lambda=float(
            pelsa_cfg.get(
                "torch_transition_leverage_lambda",
                TorchFitConfig.transition_leverage_lambda,
            )
        ),
        transition_leverage_eps=float(
            pelsa_cfg.get("torch_transition_leverage_eps", TorchFitConfig.transition_leverage_eps)
        ),
        num_threads=pelsa_cfg.get("torch_num_threads"),
    )

    if cfg.steps < 1:
        raise ValueError("PELSA torch_steps must be >= 1.")
    if cfg.lr <= 0:
        raise ValueError("PELSA torch_lr must be > 0.")
    if cfg.batch_size < 1:
        raise ValueError("PELSA torch_batch_size must be >= 1.")
    if cfg.n_starts < 1:
        raise ValueError("PELSA torch_n_starts must be >= 1.")
    valid_losses = {
        "mse",
        "huber",
        "mse_transition_support",
        "huber_transition_support",
    }
    if cfg.loss not in valid_losses:
        raise ValueError(
            f"Unsupported PELSA torch_loss={cfg.loss!r}. "
            f"Expected one of {sorted(valid_losses)}."
        )
    if cfg.huber_delta <= 0:
        raise ValueError("PELSA torch_huber_delta must be > 0.")
    if cfg.transition_leverage_lambda < 0:
        raise ValueError("PELSA torch_transition_leverage_lambda must be >= 0.")
    if cfg.transition_leverage_eps <= 0:
        raise ValueError("PELSA torch_transition_leverage_eps must be > 0.")
    if cfg.num_threads is not None and int(cfg.num_threads) < 1:
        raise ValueError("PELSA torch_num_threads must be >= 1 when provided.")
    if not 0 <= cfg.betas[0] < 1 or not 0 <= cfg.betas[1] < 1:
        raise ValueError("PELSA torch_betas values must be in [0, 1).")

    return cfg


def parse_pelsa_bounds(bounds_cfg: dict) -> PelsaBounds:
    bounds = PelsaBounds(
        pec50_margin_log10=float(
            bounds_cfg.get("pec50_margin_log10", PelsaBounds.pec50_margin_log10)
        ),
        slope=tuple(bounds_cfg.get("slope", PelsaBounds.slope)),
        front=tuple(bounds_cfg.get("front", PelsaBounds.front)),
        back=tuple(bounds_cfg.get("back", PelsaBounds.back)),
        response_margin_fraction=float(
            bounds_cfg.get(
                "response_margin_fraction",
                PelsaBounds.response_margin_fraction,
            )
        ),
    )

    if len(bounds.slope) != 2 or len(bounds.front) != 2 or len(bounds.back) != 2:
        raise ValueError("PELSA slope/front/back bounds must be length-2 sequences.")
    if bounds.response_margin_fraction < 0:
        raise ValueError("PELSA response_margin_fraction must be >= 0.")
    if any(hi <= lo for lo, hi in (bounds.slope, bounds.front, bounds.back)):
        raise ValueError("PELSA lower bounds must be smaller than upper bounds.")
    if not (bounds.front[0] < 0 < bounds.front[1]):
        raise ValueError("PELSA front bounds must span 0 in log2FC response space.")
    if not (bounds.back[0] < 0 < bounds.back[1]):
        raise ValueError("PELSA back bounds must span 0 in log2FC response space.")

    return bounds


def torch_dtype(torch, dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError("PELSA torch_dtype must be 'float32' or 'float64'.")


def logit01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def to_unconstrained(p: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return logit01((p - lower) / (upper - lower))


def build_dense_torch_input(ratio_df: pd.DataFrame) -> DenseTorchInput:
    log_conc = ratio_df["log10_concentration"].to_numpy(dtype=float)
    response = ratio_df["log2_ratio"].to_numpy(dtype=float)
    finite = np.isfinite(log_conc) & np.isfinite(response)

    df = ratio_df.loc[
        finite,
        ["peptide_id", "sample", "log10_concentration", "log2_ratio"],
    ].copy()
    if df.empty:
        raise ValueError("PELSA torch backend received no valid log2FC response points.")

    peptides = pd.Index(ratio_df["peptide_id"].drop_duplicates().astype(str))
    samples = pd.Index(df["sample"].drop_duplicates().astype(str))
    x_by_sample = (
        df.drop_duplicates("sample")
        .set_index("sample")
        .loc[samples, "log10_concentration"]
        .to_numpy(dtype=np.float32)
    )

    y = np.full((len(peptides), len(samples)), np.nan, dtype=np.float32)
    p_codes = pd.Categorical(df["peptide_id"].astype(str), categories=peptides).codes
    s_codes = pd.Categorical(df["sample"].astype(str), categories=samples).codes
    y[p_codes, s_codes] = df["log2_ratio"].to_numpy(dtype=np.float32)

    valid = np.isfinite(y)
    x_levels = np.unique(x_by_sample[np.isfinite(x_by_sample)])
    n_concentrations = np.zeros(y.shape[0], dtype=np.int16)
    for x_level in x_levels:
        n_concentrations += valid[:, x_by_sample == x_level].any(axis=1)

    return DenseTorchInput(peptides, x_by_sample, y, n_concentrations)


def sanitize_curve_results_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("peptide_id", "failure_reason", "optimizer_backend"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in ("fit_success", "pEC50_inside_range"):
        if col in df.columns:
            df[col] = pd.Series(df[col], dtype="boolean").fillna(False).astype(bool)

    int_cols = (
        "n_points",
        "n_concentrations",
        "optimizer_steps",
        "optimizer_n_starts",
        "optimizer_best_start",
    )
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

    return df


def build_empty_result(
    peptides: pd.Index,
    n_points: np.ndarray,
    n_concentrations: np.ndarray,
    eligible: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peptide_id": peptides.to_numpy(dtype=str),
            "fit_success": False,
            "failure_reason": np.where(
                eligible,
                "not_fitted",
                "insufficient_nonzero_points",
            ),
            "n_points": n_points,
            "n_concentrations": n_concentrations.astype(int),
        }
    )


def masked_nanminmax(y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_masked = np.where(mask, y, np.nan)
        return np.nanmin(y_masked, axis=1), np.nanmax(y_masked, axis=1)


def masked_nanmean(y: np.ndarray, mask: np.ndarray, axis: int = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.where(mask, y, np.nan), axis=axis)


def compute_batch_bounds(
    y_batch: np.ndarray,
    mask: np.ndarray,
    bounds: PelsaBounds,
    pec50_limits: tuple[float, float],
) -> BatchBounds:
    y_min, y_max = masked_nanminmax(y_batch, mask)
    y_center = masked_nanmean(y_batch, mask)
    y_center = np.where(np.isfinite(y_center), y_center, 0.0)

    min_span = np.maximum(0.05 * np.abs(y_center), 1e-3)
    y_span = y_max - y_min
    y_span = np.where(np.isfinite(y_span) & (y_span > 0), y_span, min_span)

    response_low = y_min - bounds.response_margin_fraction * y_span
    response_high = y_max + bounds.response_margin_fraction * y_span
    response_low = np.where(np.isfinite(response_low), response_low, y_center - min_span)
    response_high = np.where(np.isfinite(response_high), response_high, y_center + min_span)

    front_lower = np.maximum(float(bounds.front[0]), response_low)
    front_upper = np.minimum(float(bounds.front[1]), response_high)
    back_lower = np.maximum(float(bounds.back[0]), response_low)
    back_upper = np.minimum(float(bounds.back[1]), response_high)

    bad_bounds = (
        ~np.isfinite(front_lower)
        | ~np.isfinite(front_upper)
        | ~np.isfinite(back_lower)
        | ~np.isfinite(back_upper)
        | (front_upper <= front_lower)
        | (back_upper <= back_lower)
    )
    if np.any(bad_bounds):
        bad_n = int(np.sum(bad_bounds))
        raise ValueError(f"Invalid PELSA response bounds for {bad_n} features.")

    n_rows = y_batch.shape[0]
    lower = np.column_stack(
        [
            np.full(n_rows, pec50_limits[0], dtype=np.float32),
            np.full(n_rows, bounds.slope[0], dtype=np.float32),
            front_lower.astype(np.float32),
            back_lower.astype(np.float32),
        ]
    )
    upper = np.column_stack(
        [
            np.full(n_rows, pec50_limits[1], dtype=np.float32),
            np.full(n_rows, bounds.slope[1], dtype=np.float32),
            front_upper.astype(np.float32),
            back_upper.astype(np.float32),
        ]
    )

    return BatchBounds(lower, upper, response_low, response_high)


def estimate_initial_front_back(
    x: np.ndarray,
    y_batch: np.ndarray,
    mask: np.ndarray,
) -> BatchInitialValues:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_mean = masked_nanmean(y_batch, mask)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)

    low_mask = x == x_min
    high_mask = x == x_max
    front0 = masked_nanmean(y_batch[:, low_mask], mask[:, low_mask])
    back0 = masked_nanmean(y_batch[:, high_mask], mask[:, high_mask])

    front0 = np.where(np.isfinite(front0), front0, y_mean)
    back0 = np.where(np.isfinite(back0), back0, y_mean)
    return BatchInitialValues(front0.astype(np.float32), back0.astype(np.float32))


def default_start_specs(x: np.ndarray, n_starts: int) -> list[tuple[float, float, bool]]:
    """Return start specifications as (pec50, slope, swap_plateaus).

    The first five entries reproduce the legacy hard-coded starts exactly.
    Additional starts are deterministic grid-like combinations, so n_starts can be
    increased without adding more conditional branches.
    """
    x_neg = -x
    median_pec50 = float(np.nanmedian(x_neg))
    specs = [
        (median_pec50, 1.0, False),
        (float(np.nanmin(x_neg)), 1.0, False),
        (float(np.nanmax(x_neg)), 1.0, False),
        (median_pec50, 3.0, False),
        (median_pec50, 1.0, True),
    ]

    if n_starts <= len(specs):
        return specs[:n_starts]

    pec50_grid = np.linspace(float(np.nanmin(x_neg)), float(np.nanmax(x_neg)), n_starts)
    slope_grid = np.geomspace(0.5, 5.0, n_starts)
    for i in range(n_starts - len(specs)):
        specs.append((float(pec50_grid[i % n_starts]), float(slope_grid[i]), bool(i % 2)))

    return specs


def build_initial_values(
    *,
    x: np.ndarray,
    starts: BatchInitialValues,
    lower: np.ndarray,
    upper: np.ndarray,
    n_starts: int,
) -> np.ndarray:
    p0_list = []
    for pec50, slope, swap_plateaus in default_start_specs(x, n_starts):
        front0, back0 = starts.front0, starts.back0
        if swap_plateaus:
            front0, back0 = back0, front0
        p0_list.append(
            np.column_stack(
                [
                    np.full(lower.shape[0], pec50, dtype=np.float32),
                    np.full(lower.shape[0], slope, dtype=np.float32),
                    front0,
                    back0,
                ]
            )
        )

    p0 = np.stack(p0_list, axis=1)
    p0 = np.minimum(np.maximum(p0, lower[:, None, :] + 1e-8), upper[:, None, :] - 1e-8)
    return to_unconstrained(p0, lower[:, None, :], upper[:, None, :])


def make_optimizer(torch, params, cfg: TorchFitConfig):
    return torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def masked_mse_loss(torch, pred, y, mask, cfg: TorchFitConfig):
    resid = (pred - y[:, None, :]) * mask[:, None, :]
    return torch.sum(resid * resid) / torch.clamp(mask.sum(), min=1.0)


def masked_huber_loss(torch, pred, y, mask, cfg: TorchFitConfig):
    resid = (pred - y[:, None, :]) * mask[:, None, :]
    abs_resid = torch.abs(resid)
    delta = cfg.huber_delta
    loss = torch.where(
        abs_resid <= delta,
        0.5 * resid * resid,
        delta * (abs_resid - 0.5 * delta),
    )
    return torch.sum(loss) / torch.clamp(mask.sum(), min=1.0)


def transition_leverage_penalty(torch, x, mask, pred, params, cfg: TorchFitConfig):
    """Penalize fitted transitions with weak observed leverage.

    This optimizer-only regularizer is direction-agnostic. It penalizes
    transitions that are weakly supported or supported only on one side of
    the fitted EC50.

    The score combines:
    - transition support: observed points lie in the fitted non-plateau region
    - transition symmetry: support exists on both sides of the fitted EC50

    Final SSE/F-test metrics remain ordinary residual metrics computed after
    fitting.
    """
    pec50 = params[:, :, 0:1]
    slope = params[:, :, 1:2]
    front = params[:, :, 2:3]
    back = params[:, :, 3:4]

    amplitude = torch.clamp(torch.abs(front - back), min=cfg.transition_leverage_eps)
    low_plateau = torch.minimum(front, back)

    # Position along the fitted response range, independent of curve direction:
    #   z ~= 0 or 1 -> plateau
    #   z ~= 0.5    -> transition / EC50 region
    y_position = (pred - low_plateau) / amplitude
    y_position = torch.clamp(
        y_position,
        min=cfg.transition_leverage_eps,
        max=1.0 - cfg.transition_leverage_eps,
    )
    y_weight = 4.0 * y_position * (1.0 - y_position)

    # Soft left/right assignment around the fitted EC50.  The sharpness is
    # tied to the fitted slope, so no extra width hyperparameter is needed:
    # steep curves define a sharper left/right split, shallow curves a softer one.
    center = -pec50
    side_sharpness = torch.clamp(slope, min=1e-6) * np.log(10.0)
    right_weight = torch.sigmoid(side_sharpness * (x[:, None, :] - center))
    left_weight = 1.0 - right_weight

    left_support = torch.sum(mask[:, None, :] * y_weight * left_weight, dim=2)
    right_support = torch.sum(mask[:, None, :] * y_weight * right_weight, dim=2)
    transition_support = left_support + right_support

    possible = torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=1.0)
    support_fraction = transition_support / possible

    # Symmetry is 1 when left/right support is balanced and approaches 0 when
    # support is one-sided.  This specifically targets edge/single-side fits.
    balance = (
        4.0 * left_support * right_support
        / torch.clamp(transition_support * transition_support, min=cfg.transition_leverage_eps)
    )
    leverage_fraction = support_fraction * balance

    penalty = -torch.log(leverage_fraction + cfg.transition_leverage_eps)
    return torch.mean(penalty)


def mse_transition_support_loss(torch, model, x, y, mask, cfg: TorchFitConfig):
    pred = model.predict(x)
    data_loss = masked_mse_loss(torch, pred, y, mask, cfg)
    params = model.constrained_parameters()
    penalty = transition_leverage_penalty(torch, x, mask, pred, params, cfg)
    return data_loss + cfg.transition_leverage_lambda * penalty


def huber_transition_support_loss(torch, model, x, y, mask, cfg: TorchFitConfig):
    pred = model.predict(x)
    data_loss = masked_huber_loss(torch, pred, y, mask, cfg)
    params = model.constrained_parameters()
    penalty = transition_leverage_penalty(torch, x, mask, pred, params, cfg)
    return data_loss + cfg.transition_leverage_lambda * penalty


def mse_loss(torch, model, x, y, mask, cfg: TorchFitConfig):
    return masked_mse_loss(torch, model.predict(x), y, mask, cfg)


def huber_loss(torch, model, x, y, mask, cfg: TorchFitConfig):
    return masked_huber_loss(torch, model.predict(x), y, mask, cfg)



def get_loss_function(cfg: TorchFitConfig) -> Callable:
    if cfg.loss == "mse":
        return mse_loss
    if cfg.loss == "huber":
        return huber_loss
    if cfg.loss == "mse_transition_support":
        return mse_transition_support_loss
    if cfg.loss == "huber_transition_support":
        return huber_transition_support_loss
    raise ValueError(f"Unsupported PELSA torch loss: {cfg.loss!r}.")


def train_model(torch, model, x, y, mask, cfg: TorchFitConfig) -> None:
    optimizer = make_optimizer(torch, model.parameters(), cfg)
    loss_fn = get_loss_function(cfg)

    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(torch, model, x, y, mask, cfg)
        loss.backward()
        optimizer.step()


def tensor_from_numpy(torch, array: np.ndarray, dtype):
    return torch.from_numpy(array).to(dtype=dtype, device="cpu")


def evaluate_model(torch, model, x, y, mask, x_min: float, x_max: float):
    with torch.no_grad():
        params_all = model.constrained_parameters()
        pred_all = model.predict(x)
        resid_all = (pred_all - y[:, None, :]) * mask[:, None, :]
        sse_all = torch.sum(resid_all * resid_all, dim=2)
        best_start = torch.argmin(sse_all, dim=1)

        row_ix = torch.arange(params_all.shape[0], device="cpu")
        params = params_all[row_ix, best_start, :]
        sse_curve = sse_all[row_ix, best_start]

        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        y_bar = torch.sum(y * mask, dim=1, keepdim=True) / denom[:, None]
        sse_null = torch.sum(((y - y_bar) * mask) ** 2, dim=1)

        rmse = torch.sqrt(sse_curve / denom)
        r2 = 1.0 - sse_curve / torch.clamp(sse_null, min=1e-20)

        x_low = torch.tensor([[x_min]], device="cpu", dtype=y.dtype)
        x_high = torch.tensor([[x_max]], device="cpu", dtype=y.dtype)
        y_low = four_pl_torch(
            torch,
            x_low,
            params[:, 0:1],
            params[:, 1:2],
            params[:, 2:3],
            params[:, 3:4],
        )[:, 0]
        y_high = four_pl_torch(
            torch,
            x_high,
            params[:, 0:1],
            params[:, 1:2],
            params[:, 2:3],
            params[:, 3:4],
        )[:, 0]
        curve_fc = y_high - y_low

    return {
        "params": params.cpu().numpy(),
        "sse_curve": sse_curve.cpu().numpy(),
        "sse_null": sse_null.cpu().numpy(),
        "rmse": rmse.cpu().numpy(),
        "r2": r2.cpu().numpy(),
        "curve_fold_change_log2": curve_fc.cpu().numpy(),
        "best_start": best_start.cpu().numpy(),
    }


def fit_batch(
    *,
    torch,
    x_np: np.ndarray,
    y_batch_np: np.ndarray,
    idx: np.ndarray,
    bounds: PelsaBounds,
    fit_cfg: TorchFitConfig,
    dtype,
    pec50_limits: tuple[float, float],
    observed_pec50_limits: tuple[float, float],
) -> pd.DataFrame:
    mask_np = np.isfinite(y_batch_np)
    batch_bounds = compute_batch_bounds(y_batch_np, mask_np, bounds, pec50_limits)
    starts = estimate_initial_front_back(x_np, y_batch_np, mask_np)
    initial_values = build_initial_values(
        x=x_np,
        starts=starts,
        lower=batch_bounds.lower,
        upper=batch_bounds.upper,
        n_starts=fit_cfg.n_starts,
    )

    x = tensor_from_numpy(torch, x_np[None, :], dtype)
    y = tensor_from_numpy(torch, np.nan_to_num(y_batch_np, nan=0.0), dtype)
    mask = tensor_from_numpy(torch, mask_np.astype(np.float32), dtype)
    lower = tensor_from_numpy(torch, batch_bounds.lower, dtype)
    upper = tensor_from_numpy(torch, batch_bounds.upper, dtype)

    model = FourPLTorchModel(torch, lower, upper, initial_values)
    train_model(torch, model, x, y, mask, fit_cfg)
    metrics = evaluate_model(
        torch,
        model,
        x,
        y,
        mask,
        x_min=float(np.nanmin(x_np)),
        x_max=float(np.nanmax(x_np)),
    )

    params = metrics["params"]
    amplitude = np.abs(params[:, 2] - params[:, 3])
    normalized_rmse = metrics["rmse"] / np.clip(amplitude, 1e-12, None)

    pec50_se, pec50_ci_low, pec50_ci_high, pec50_ci_width_norm = (
        estimate_pec50_ci_batch(
            x_np=x_np,
            y_batch_np=y_batch_np,
            mask_np=mask_np,
            params_np=params,
        )
    )

    return pd.DataFrame(
        {
            "row_idx": idx,
            "fit_success": True,
            "failure_reason": None,
            "pec50": params[:, 0],
            "slope": params[:, 1],
            "front": params[:, 2],
            "back": params[:, 3],
            "response_bound_low": batch_bounds.response_low,
            "response_bound_high": batch_bounds.response_high,
            "sse_curve": metrics["sse_curve"],
            "sse_null": metrics["sse_null"],
            "rmse": metrics["rmse"],
            "normalized_rmse": normalized_rmse,
            "r2": metrics["r2"],
            "curve_fold_change_log2": metrics["curve_fold_change_log2"],
            "pEC50_inside_range": (
                (params[:, 0] >= observed_pec50_limits[0])
                & (params[:, 0] <= observed_pec50_limits[1])
            ),
            "pEC50_se": pec50_se,
            "pEC50_ci_low": pec50_ci_low,
            "pEC50_ci_high": pec50_ci_high,
            "pEC50_ci_width_norm": pec50_ci_width_norm,
            "optimizer_backend": "torch_adam",
            "optimizer_steps": fit_cfg.steps,
            "optimizer_n_starts": fit_cfg.n_starts,
            "optimizer_best_start": metrics["best_start"],
        }
    )


def add_curve_statistics(result: pd.DataFrame) -> pd.DataFrame:
    """Add CurveCurator-style recalibrated F statistic and BH q-value."""
    result = result.copy()
    ok = result["fit_success"].to_numpy(dtype=bool)
    n = result["n_points"].to_numpy(dtype=float)
    sse0 = result["sse_null"].to_numpy(dtype=float)
    sse1 = result["sse_curve"].to_numpy(dtype=float)

    f_value = np.full(len(result), np.nan, dtype=float)
    p_value = np.full(len(result), np.nan, dtype=float)

    valid = ok & np.isfinite(sse0) & np.isfinite(sse1) & (sse1 > 0) & (n > N_4PL_PARAMS)
    f_value[valid] = ((sse0[valid] - sse1[valid]) / sse1[valid]) * (
        n[valid] / N_4PL_PARAMS
    )
    f_value[valid] = np.maximum(f_value[valid], 0.0)

    def low_n_slope_adjustment(nn):
        return 1.0 / (((nn - 4.0) ** 4) / nn + 4.0)

    dfn = 5.0
    dfd = (0.8 - low_n_slope_adjustment(n[valid])) * (n[valid] - 2.5)
    p_value[valid] = stats.f.sf(f_value[valid], dfn=dfn, dfd=dfd, scale=1.0, loc=0.12)

    q_value = np.full(len(result), np.nan, dtype=float)
    p_ok = np.isfinite(p_value)
    if np.any(p_ok):
        q_value[p_ok] = multipletests(p_value[p_ok], method="fdr_bh")[1]

    result["curve_f_value"] = f_value
    result["curve_p_value"] = p_value
    result["curve_q_value"] = q_value
    result["curve_neglog10_q"] = -np.log10(np.clip(q_value, 1e-300, 1.0))
    return result


def fit_4pl_torch_from_ratio_df(ratio_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit peptide-level 4PL curves from a PELSA log2FC table using CPU torch."""
    torch = require_torch()

    pelsa_cfg = ((config or {}).get("analysis", {}) or {}).get("pelsa", {}) or {}
    bounds = parse_pelsa_bounds(pelsa_cfg.get("bounds", {}) or {})
    fit_cfg = parse_torch_fit_config(pelsa_cfg)
    dtype = torch_dtype(torch, fit_cfg.dtype_name)

    if fit_cfg.num_threads is not None:
        torch.set_num_threads(int(fit_cfg.num_threads))

    t0 = perf_counter()
    dense = build_dense_torch_input(ratio_df)
    valid_np = np.isfinite(dense.y)
    n_points = valid_np.sum(axis=1).astype(int)
    eligible = dense.n_concentrations >= N_4PL_PARAMS

    log_info(
        "PELSA torch log2FC input: "
        f"features={len(dense.peptides)}, eligible={int(eligible.sum())}, "
        f"samples_nonzero={len(dense.x)}, build_time={perf_counter() - t0:.2f}s"
    )

    result = build_empty_result(
        dense.peptides,
        n_points,
        dense.n_concentrations,
        eligible,
    )

    x_neg = -dense.x
    observed_pec50_limits = (float(np.nanmin(x_neg)), float(np.nanmax(x_neg)))
    pec50_limits = (
        observed_pec50_limits[0] - bounds.pec50_margin_log10,
        observed_pec50_limits[1] + bounds.pec50_margin_log10,
    )

    eligible_idx = np.flatnonzero(eligible)
    out_rows = []
    t_fit = perf_counter()

    for start in tqdm(
        range(0, len(eligible_idx), fit_cfg.batch_size),
        desc="PELSA torch 4PL fitting",
    ):
        idx = eligible_idx[start : start + fit_cfg.batch_size]
        out_rows.append(
            fit_batch(
                torch=torch,
                x_np=dense.x,
                y_batch_np=dense.y[idx, :],
                idx=idx,
                bounds=bounds,
                fit_cfg=fit_cfg,
                dtype=dtype,
                pec50_limits=pec50_limits,
                observed_pec50_limits=observed_pec50_limits,
            )
        )

    if out_rows:
        fitted = pd.concat(out_rows, ignore_index=True).set_index("row_idx")
        for col in fitted.columns:
            result.loc[fitted.index, col] = fitted[col].to_numpy()

    result = add_curve_statistics(result)

    log_info(
        f"PELSA torch 4PL fitting wall-time: {perf_counter() - t_fit:.2f}s "
        f"(batch_size={fit_cfg.batch_size}, steps={fit_cfg.steps}, "
        f"lr={fit_cfg.lr}, n_starts={fit_cfg.n_starts}, "
        f"betas={fit_cfg.betas}, loss={fit_cfg.loss}, "
        f"huber_delta={fit_cfg.huber_delta}, "
        f"transition_leverage_lambda={fit_cfg.transition_leverage_lambda}, "
        f"response_margin_fraction={bounds.response_margin_fraction})"
    )

    return sanitize_curve_results_for_h5ad(result)


# Backward-compatible aliases for existing internal imports/tests.
_four_pl_numpy = four_pl_numpy
_four_pl_jacobian_numpy = four_pl_jacobian_numpy
_estimate_pec50_ci_batch = estimate_pec50_ci_batch
_require_torch = require_torch
_logit01 = logit01
_to_unconstrained = to_unconstrained
_torch_4pl = four_pl_torch
_sanitize_curve_results_for_h5ad = sanitize_curve_results_for_h5ad


def _build_dense_torch_input(
    ratio_df: pd.DataFrame,
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    dense = build_dense_torch_input(ratio_df)
    return dense.peptides, dense.x, dense.y, dense.n_concentrations
