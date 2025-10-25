import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-8

def _pooled_sd(groups):
    """
    Pooled within-group SD across multiple groups:
      sqrt( sum_k (n_k - 1) * s_k^2 / sum_k (n_k - 1) )
    groups: list of 1D arrays (each group's observed values)
    Returns float (>= _EPS).
    """
    num = 0.0
    den = 0
    for g in groups:
        if g.size >= 2:
            s = np.std(g, ddof=1)
            num += (g.size - 1) * (s ** 2)
            den += (g.size - 1)
    if den == 0:
        # fallback: overall SD across all values (ddof=1 if possible)
        allv = np.concatenate([g for g in groups if g.size > 0], axis=0)
        if allv.size >= 2:
            s = np.std(allv, ddof=1)
        elif allv.size == 1:
            s = 0.0
        else:
            s = 0.0
        return max(float(s), _EPS)
    return max(float(np.sqrt(num / den)), _EPS)


class LC_ConMedImputer(BaseEstimator, TransformerMixin):
    """
    Hybrid condition-aware imputer for proteomics (log scale).
    Shape: X is (n_features, n_samples). Rows = proteins/features, Cols = samples.

    Rule for a missing cell X[i, j]:
      • If protein i has ≥ in_min_obs observed values IN THE SAME CONDITION as sample j:
            center := median of that condition for protein i
            sd_pool := pooled within-condition SD of protein i across ALL conditions
                       that have ≥ in_min_obs observations
            draw ~ Normal(center, jitter_frac * sd_pool)
            if clip_upper: draw = min(draw, condition_median)
      • Else (no signal in that condition):
            column LoD := np.nanquantile(column j, lod_q)
            value := (column LoD - lod_shift) + Normal(0, lod_sd_width)   # tiny jitter
            if clip_lod: value = min(value, column LoD)
    """

    def __init__(
        self,
        condition_map,           # pandas DataFrame with columns: ["Sample", group_column]
        sample_index,            # list/array of sample names in the SAME order as X's columns
        group_column="CONDITION",
        jitter_frac=0.20,        # jitter = jitter_frac × pooled_sd
        q_lower=0.25,
        q_upper=0.75,
        # MinDet fallback params (used only when condition has no signal):
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
        X = np.asarray(X, dtype=np.float64)
        n_feat, n_samp = X.shape

        # sample -> condition in column order
        cmap = self.condition_map.set_index("Sample")[self.group_column]
        self._cond = np.array([cmap[s] for s in self.sample_index], dtype=object)

        # LoD(s) for fallback
        obs = X[~np.isnan(X)]
        if obs.size == 0:
            # degenerate case: nothing observed; pick zero floor
            self._lod_global = 0.0
        else:
            if self.lod_k is None:
                # Adaptive K with bounds keeps LoD in the true low tail
                K_adapt = int(np.ceil(0.01 * obs.size))
                K = max(10, min(50, K_adapt))   # 10 ≤ K ≤ 50, if no k is given
            else:
                K = int(self.lod_k)

            K = min(K, obs.size)  # safe bound
            # take median of the K smallest values
            smallest = np.partition(obs, K-1)[:K]
            self._lod_global = float(np.median(smallest))

        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_imp = X.copy()
        n_feat, n_samp = X.shape
        rows, cols = np.where(np.isnan(X_imp))
        if not len(rows):
            return X_imp

        for i, j in zip(rows, cols):
            cond_j = self._cond[j]

            # observed values of protein i within each condition
            # (build once per row if speed becomes an issue; fine as-is for clarity)
            obs_by_cond = {}
            for c in np.unique(self._cond):
                idx_c = (self._cond == c)
                vals = X[i, idx_c]
                vals = vals[~np.isnan(vals)]
                obs_by_cond[c] = vals

            in_group_vals = obs_by_cond.get(cond_j, np.array([]))
            if in_group_vals.size >= self.in_min_obs:
                # center at in-condition median
                center = float(np.median(in_group_vals))
                # pooled SD across conditions with data
                groups = [v for v in obs_by_cond.values() if v.size >= self.in_min_obs]
                sd_pool = _pooled_sd(groups)
                sd_jitter = max(self.jitter_frac * sd_pool, _EPS)
                draw = self._rng.normal(center, sd_jitter)
                # clip to in-condition [Q_lower, Q_upper]
                q_lo = float(np.quantile(in_group_vals, self.q_lower, method="linear")) if in_group_vals.size > 1 else center
                q_hi = float(np.quantile(in_group_vals, self.q_upper, method="linear")) if in_group_vals.size > 1 else center
                if q_lo > q_hi:  # safety (shouldn't happen)
                    q_lo, q_hi = min(q_lo, q_hi), max(q_lo, q_hi)
                draw = min(max(draw, q_lo), q_hi)
                X_imp[i, j] = draw
            else:
                # MinDet fallback for this column
                lod = self._lod_global
                base = lod - self.lod_shift
                jitter = self._rng.normal(0.0, max(self.lod_sd_width, _EPS))
                val = base + jitter
                val = min(val, lod)
                X_imp[i, j] = val

        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

