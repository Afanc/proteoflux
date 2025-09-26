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


class HybridImputer(BaseEstimator, TransformerMixin):
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

    Notes:
      - Keeps matrix orientation consistent with the rest of ProteoFlux.
      - Randomness is controlled via random_state for reproducibility.
    """

    def __init__(
        self,
        condition_map,           # pandas DataFrame with columns: ["Sample", group_column]
        sample_index,            # list/array of sample names in the SAME order as X's columns
        group_column="CONDITION",
        in_min_obs=1,            # "ANY real measurement" ⇒ 1 (you can set 2 if you prefer)
        jitter_frac=0.20,        # jitter = jitter_frac × pooled_sd
        clip_upper=True,         # cap draws at in-condition median
        # MinDet fallback params (used only when condition has no signal):
        lod_q=0.01,
        lod_shift=0.20,
        lod_sd_width=0.05,
        clip_lod=True,
        random_state=42,
    ):
        self.condition_map = condition_map
        self.sample_index = sample_index
        self.group_column = group_column

        self.in_min_obs = int(in_min_obs)
        self.jitter_frac = float(jitter_frac)
        self.clip_upper = bool(clip_upper)

        self.lod_q = float(lod_q)
        self.lod_shift = float(lod_shift)
        self.lod_sd_width = float(lod_sd_width)
        self.clip_lod = bool(clip_lod)

        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_feat, n_samp = X.shape

        # sample -> condition in column order
        cmap = self.condition_map.set_index("Sample")[self.group_column]
        self._cond = np.array([cmap[s] for s in self.sample_index], dtype=object)

        # per-column LoD (for fallback)
        col_has_val = np.any(~np.isnan(X), axis=0)
        col_lod = np.full(n_samp, np.nan, dtype=np.float64)
        if np.any(col_has_val):
            col_lod[col_has_val] = np.nanquantile(X[:, col_has_val], self.lod_q, axis=0)
        if np.isnan(col_lod).any():
            gmin = np.nanmin(X) if np.isfinite(np.nanmin(X)) else 0.0
            col_lod[np.isnan(col_lod)] = gmin
        self._col_lod = col_lod

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
                if self.clip_upper:
                    # cap at in-condition median to avoid inflating that condition
                    draw = min(draw, center)
                X_imp[i, j] = draw
            else:
                # MinDet fallback for this column
                base = self._col_lod[j] - self.lod_shift
                jitter = self._rng.normal(0.0, max(self.lod_sd_width, _EPS))
                val = base + jitter
                if self.clip_lod:
                    val = min(val, self._col_lod[j])
                X_imp[i, j] = val

        return X_imp

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

