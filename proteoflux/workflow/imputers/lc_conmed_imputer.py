import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-8

def _pooled_sd(groups):
    """
    Compute pooled within-group standard deviation.

    Formula:
        sqrt( sum_k (n_k - 1) * s_k^2 / sum_k (n_k - 1) )

    Parameters
    ----------
    groups : list of 1D np.ndarray
        Each array contains observed (non-NaN) values for a condition.

    Returns
    -------
    float
        Pooled standard deviation (>= _EPS).
    """
    num = 0.0
    den = 0
    for g in groups:
        if g.size >= 2:
            s = np.std(g, ddof=1)
            num += (g.size - 1) * (s ** 2)
            den += (g.size - 1)
    if den == 0:
        non_empty = [g for g in groups if g.size > 0]
        if not non_empty:
            return _EPS
        allv = np.concatenate(non_empty, axis=0)
        if allv.size >= 2:
            s = np.std(allv, ddof=1)
        else:
            s = 0.0
        return max(float(s), _EPS)
    return max(float(np.sqrt(num / den)), _EPS)

class LC_ConMedImputer(BaseEstimator, TransformerMixin):
    """
    Hybrid condition-aware imputer for log-scale proteomics data.
    Optimized version with per-row/per-condition caching.

    Missing value rule for cell X[i, j]:
      • If the same condition has >= in_min_obs (now hardcoded to 1) observed values:
            center = median of that condition
            sd_pool = pooled SD across all conditions with data
            draw ~ Normal(center, jitter_frac * sd_pool)
            clipped to [Q_lower, Q_upper] of that condition
      • Else (no data in this condition):
            LoD fallback = (global LoD - lod_shift) + Normal(0, lod_sd_width)
            clipped to <= global LoD
    """

    def __init__(
        self,
        condition_map,
        sample_index,
        group_column="CONDITION",
        jitter_frac=0.20,
        q_lower=0.25,
        q_upper=0.75,
        lod_k=None,
        lod_shift=0.20,
        lod_sd_width=0.05,
        random_state=42,
    ):
        self.condition_map = condition_map
        self.sample_index = sample_index
        self.group_column = group_column

        self.in_min_obs = 1
        self.lod_k = lod_k
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.jitter_frac = float(jitter_frac)
        self.lod_shift = float(lod_shift)
        self.lod_sd_width = float(lod_sd_width)
        self.random_state = random_state

    def fit(self, X, y=None):
        """Precompute condition indices and global LoD threshold."""
        X = np.asarray(X, dtype=np.float64)

        # sample -> condition in column order
        cmap = self.condition_map.set_index("Sample")[self.group_column]
        self._cond = np.array([cmap[s] for s in self.sample_index], dtype=object)

        # Cache condition levels and column indices
        cond_levels = np.unique(self._cond)
        self._cond_levels = cond_levels
        self._cond_idx = {c: np.where(self._cond == c)[0] for c in cond_levels}

        # Global LoD (median of lowest k intensities)
        obs = X[~np.isnan(X)]
        if obs.size == 0:
            self._lod_global = 0.0
        else:
            if self.lod_k is None:
                K_adapt = int(np.ceil(0.01 * obs.size))
                K = max(10, min(50, K_adapt))
            else:
                K = int(self.lod_k)
            K = min(K, obs.size)
            smallest = np.partition(obs, K - 1)[:K]
            self._lod_global = float(np.median(smallest))

        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X):
        """Perform condition-aware imputation."""
        X = np.asarray(X, dtype=np.float64)
        X_imp = X.copy()

        # Hoist NaN mask once for the whole matrix
        nan_mask = np.isnan(X_imp)

        # Use the precomputed order from the global mask (row-major)
        rows, cols = np.where(nan_mask)
        if rows.size == 0:
            return X_imp

        # Group missing (i, j) by row while preserving original order
        per_row_missing = {}
        for i, j in zip(rows, cols):
            per_row_missing.setdefault(i, []).append(j)

        # Local bindings
        cond = self._cond
        cond_idx = self._cond_idx
        in_min_obs = self.in_min_obs
        q_lower = self.q_lower
        q_upper = self.q_upper
        jitter_frac = self.jitter_frac
        rng = self._rng
        lod_global = self._lod_global
        lod_shift = self.lod_shift
        lod_sd = self.lod_sd_width
        _EPS_local = _EPS

        for i, js in per_row_missing.items():
            X_row = X_imp[i]
            row_nan = nan_mask[i]

            obs_by_cond = {}
            # collect observed values per cond

            for c, idx_c in cond_idx.items():
                not_nan_c = ~row_nan[idx_c]
                if np.any(not_nan_c):
                    vals = X_row[idx_c][not_nan_c]
                else:
                    vals = np.array([], dtype=np.float64)
                obs_by_cond[c] = vals

            # Pooled SD across all conditions with any data
            groups = [v for v in obs_by_cond.values() if v.size >= in_min_obs]
            sd_pool = _pooled_sd(groups)
            sd_jitter = max(jitter_frac * sd_pool, _EPS_local)

            # cache cond stats
            cond_stats = {}
            for c, vals in obs_by_cond.items():
                if vals.size >= in_min_obs:
                    center = float(np.median(vals))
                    if vals.size > 1:
                        q_lo = float(np.quantile(vals, q_lower, method="linear"))
                        q_hi = float(np.quantile(vals, q_upper, method="linear"))
                        if q_lo > q_hi:  # safety
                            q_lo, q_hi = q_hi, q_lo
                    else:
                        q_lo = q_hi = center
                    cond_stats[c] = (center, q_lo, q_hi)

            # Fill missing
            for j in js:
                cj = cond[j]
                in_group_vals = obs_by_cond.get(cj, np.array([], dtype=np.float64))

                if in_group_vals.size >= in_min_obs:
                    center, q_lo, q_hi = cond_stats[cj]
                    draw = rng.normal(center, sd_jitter)
                    if draw < q_lo:
                        draw = q_lo
                    elif draw > q_hi:
                        draw = q_hi
                    X_imp[i, j] = draw
                else:
                    # MinDet fallback
                    base = lod_global - lod_shift
                    jitter = rng.normal(0.0, max(lod_sd, _EPS_local))
                    val = base + jitter
                    if val > lod_global:
                        val = lod_global
                    X_imp[i, j] = val

        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
