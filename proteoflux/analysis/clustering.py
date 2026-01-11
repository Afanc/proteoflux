"""Clustering utilities for PCA/UMAP and missingness patterns.

Provides:
  - run_clustering: PCA -> neighbors -> UMAP; hierarchical clustering on samples/features.
                   Supports optional feature capping via variance- or random-based selection,
                   while writing back sample-level results to the original AnnData.
  - run_clustering_missingness: hierarchical clustering on binary missingness masks.
"""

import warnings
import numpy as np
import scipy.cluster.hierarchy as sch
from anndata import AnnData
import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from typing import Optional, Dict, Any, Sequence
from proteoflux.utils.utils import log_time

def _lc_mindet_impute(
    M: np.ndarray,
    *,
    lod_k: Optional[int] = None,
    lod_shift: float = 0.20,
    lod_sd_width: float = 0.05,
    random_seed: int = 0,
) -> np.ndarray:
    """Left-censor (MinDet-like) imputation for clustering/QC only.

    - Global LoD is median of the lowest K observed intensities.
    - Missing values are filled with (LoD - lod_shift) + Normal(0, lod_sd_width),
      clipped to <= LoD.
    - No condition-aware centering. No mean imputation.
    """
    M = np.asarray(M, dtype=np.float64)
    if not np.isnan(M).any():
        return M.astype(np.float32, copy=False)

    obs = M[~np.isnan(M)]
    if obs.size == 0:
        # Degenerate: keep finite zeros so downstream does not crash.
        return np.zeros_like(M, dtype=np.float32)

    if lod_k is None:
        K_adapt = int(np.ceil(0.01 * obs.size))
        K = max(10, min(50, K_adapt))
    else:
        K = int(lod_k)
    K = min(K, obs.size)
    smallest = np.partition(obs, K - 1)[:K]
    lod = float(np.median(smallest))

    rng = np.random.default_rng(int(random_seed))
    base = lod - float(lod_shift)
    sd = max(float(lod_sd_width), 1e-8)

    out = M.copy()
    nan_mask = np.isnan(out)
    draws = base + rng.normal(0.0, sd, size=int(nan_mask.sum()))
    draws = np.minimum(draws, lod)
    out[nan_mask] = draws
    return out.astype(np.float32, copy=False)

def _colmean_impute(M: np.ndarray) -> np.ndarray:
    """Return a copy where NaNs are replaced by per-column means (finite if any finite exists)."""
    M = np.asarray(M)
    if not np.isnan(M).any():
        return M
    with np.errstate(invalid="ignore", divide="ignore"):
        col_mean = np.nanmean(M, axis=0)
    # if a column is all-NaN, nanmean gives NaN -> replace with 0 so linkage/PCA stay finite
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(M.dtype, copy=False)
    return np.where(np.isnan(M), col_mean, M)

def _pick_feature_indices(adata: AnnData, layer: Optional[str], max_features: Optional[int],
                          strategy: str = "variance", random_seed: int = 0) -> np.ndarray:
    """
    Return feature indices (≤ max_features) by variance or deterministic random sampling.

    strategy:
      - "variance": keep features with largest variance (ignoring NaNs)
      - "random"  : deterministic random subset (seeded)
    """
    n_vars = adata.n_vars
    if (max_features is None) or (n_vars <= max_features):
        return np.arange(n_vars)

    # choose matrix to compute stats from
    M = adata.layers[layer] if layer is not None else adata.X

    if strategy == "variance":
        # nanvar over samples axis, higher variance -> more informative
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Degrees of freedom <= 0 for slice\.",
                category=RuntimeWarning,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                var = np.nanvar(M, axis=0)
        var = np.where(np.isfinite(var), var, -np.inf)
        idx = np.argsort(var)[::-1][:max_features]
        return np.sort(idx)  # keep ascending index order for stable slicing
    elif strategy == "random":
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_vars, size=max_features, replace=False)
        return np.sort(idx)
    else:
        raise ValueError(f"Unknown feature selection strategy: {strategy}")


def _maybe_subset(adata: AnnData, feat_idx: Sequence[int]) -> AnnData:
    """Return a lightweight copy with selected variables for computations."""
    # Using copy() so downstream modifications don't touch the original.
    return adata[:, feat_idx].copy() if len(feat_idx) < adata.n_vars else adata


@log_time("Running clustering pipeline")
def run_clustering(
    adata: AnnData,
    layer: Optional[str] = None,
    n_pcs: int = 50,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    hierarchical_method: str = "ward",
    hierarchical_metric: str = "euclidean",
    max_features: Optional[int] = None,
    feature_selection: str = "variance", # or "random"
    random_seed: int = 0,
) -> AnnData:
    """PCA/UMAP + hierarchical clustering (samples & features), with optional feature capping.

    If `max_features` < `adata.n_vars`, computations run on a temporary subset chosen by
    `feature_selection`. Sample-level results are written back to the original `adata`.
    """

    # Small-n guard: UMAP's spectral init fails when n_obs <= 3 (eigsh requires k < N).
    n_obs = adata.n_obs
    small_n = n_obs <= 3

    # choose feature subset (indices on original adata)
    feat_idx = _pick_feature_indices(adata, layer, max_features, feature_selection, random_seed)

    # work on a temporary view/copy if we subselect
    A = _maybe_subset(adata, feat_idx)

    # Keep references to names for downstream order storage
    samples = A.obs_names
    features = A.var_names

    # Select data matrix (samples × features)
    data = A.layers[layer] if layer is not None else A.X

    # If 'layer' is provided, we use LC-only MinDet imputation
    if layer is not None:
        X_for_embed = _lc_mindet_impute(
            data,
            random_seed=random_seed,
        )
    else:
        # Caller explicitly wants embeddings on adata.X (already fully imputed upstream)
        X_for_embed = np.asarray(A.X, dtype=np.float32)

    # Embeddings are computed on a scratch AnnData to guarantee we never overwrite A.X.
    E = AnnData(X=X_for_embed, obs=A.obs.copy())

    # PCA (cap PCs for tiny sample counts)
    n_comps = max(1, min(int(n_pcs), E.n_obs - 1))
    sc.tl.pca(E, n_comps=n_comps)
    adata.uns["pca"] = E.uns.get("pca", {})

    # Neighbors on PCA space
    if "neighbors" not in E.uns:
        nn = int(round(E.n_obs ** 0.5))
        nn = max(10, min(30, nn))
        nn = min(nn, E.n_obs - 1)
        sc.pp.neighbors(E, n_neighbors=nn, use_rep="X_pca")

    # Keep UMAP codepath in the module, in case I change my mind
    # but we compute MDS instead 
    if not small_n:
        Xp = np.asarray(E.obsm["X_pca"][:, :n_comps], dtype=np.float64)

        # Correlation distance between samples
        # corr_ij = corr(Xp[i], Xp[j]); distance = 1 - corr_ij
        C = np.corrcoef(Xp)  # shape (n_obs, n_obs)

        # numerical safety
        C = np.clip(C, -1.0, 1.0)
        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=int(random_seed),
            normalized_stress="auto",
        )
        E.obsm["X_mds"] = mds.fit_transform(D).astype(np.float32, copy=False)
        E.uns["mds"] = {"stress": float(getattr(mds, "stress_", np.nan)), "metric": "correlation_pca"}
    else:
        E.uns["mds"] = {"skipped_reason": "n_obs<=3"}

    # Hierarchical clustering on samples (using PCA space)
    pca_matrix = E.obsm["X_pca"][:, :n_comps]  # shape: n_samples × n_pcs
    sample_linkage = sch.linkage(pca_matrix, method=hierarchical_method, metric=hierarchical_metric)
    sample_leaves = sch.leaves_list(sample_linkage)

    # Prepare data for hierarchical clustering on intensity & centered
    # Linkage must be finite: impute for linkage only.
    data_link = X_for_embed.astype(np.float32, copy=False)
    mats = {"intensity": data_link}
    Xc = data_link - np.mean(data_link, axis=0, keepdims=True)
    A.layers["centered"] = Xc
    mats["centered"] = Xc

    # We NaN-pad non-selected features so downstream code can always read adata.layers["centered"].
    centered_full = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=Xc.dtype)

    # 'feat_idx' indexes columns on the original adata; Xc matches A = adata[:, feat_idx]
    centered_full[:, feat_idx] = Xc
    adata.layers["centered"] = centered_full

    # Write results into A.uns
    for tag, M in mats.items():
        s_link = sch.linkage(M,   method=hierarchical_method, metric=hierarchical_metric)
        f_link = sch.linkage(M.T, method=hierarchical_method, metric=hierarchical_metric)
        s_leaves = sch.leaves_list(s_link)
        f_leaves = sch.leaves_list(f_link)

        A.uns[f"{tag}_sample_linkage"]  = s_link
        A.uns[f"{tag}_feature_linkage"] = f_link
        A.uns[f"{tag}_sample_order"]    = samples[s_leaves].tolist()
        A.uns[f"{tag}_feature_order"]   = features[f_leaves].tolist()

    # Expose the subset info
    if max_features is not None and A.n_vars < adata.n_vars:
        A.uns["clustering_feature_subset_indices"] = np.asarray(feat_idx)
        A.uns["clustering_feature_subset_names"]   = features.tolist()
        A.uns["clustering_feature_selection"]      = feature_selection
        A.uns["clustering_max_features"]           = max_features

    # IMPORTANT: write back *sample-level* results to the original adata, and mirror uns keys.
    # This keeps downstream code (that expects full-length feature arrays) unchanged. Headaches.
    adata.obsm["X_pca"] = E.obsm["X_pca"]
    if "X_mds" in E.obsm:
        adata.obsm["X_mds"] = E.obsm["X_mds"]

    if "neighbors" in E.uns:
        adata.uns["neighbors"] = E.uns["neighbors"]
        for k in ("distances", "connectivities"):
            if k in E.obsp:
                adata.obsp[k] = E.obsp[k]

    if "mds" in E.uns:
        adata.uns["mds"] = E.uns["mds"]

    # Mirror clustering results into adata.uns under the same keys
    for key in list(A.uns.keys()):
        if key.endswith(('_sample_linkage', '_feature_linkage', '_sample_order', '_feature_order')) or \
           key.startswith('clustering_'):
            adata.uns[key] = A.uns[key]

    return adata

@log_time("Running missingness clustering pipeline")
def run_clustering_missingness(
    adata: AnnData,
    layer: str = "normalized",
    hierarchical_method: str = "ward",
    hierarchical_metric: str = "euclidean",
    max_features: Optional[int] = None,
    feature_selection: str = "variance",
    random_seed: int = 0,
) -> AnnData:
    """Cluster the binary missingness pattern (1=NaN, 0=observed) of `adata[layer]`.

    Missingness is computed across the full matrix; for linkage, we may cap the
    number of features considered (≤ `max_features`) using variance- or random-based
    selection. Orders saved in `adata.uns['missing_*_order']` reflect the subset if capped.
    """
    # Build binary missingness: samples × features (uint8 is compact)
    mat = adata.layers[layer]
    missing = np.isnan(mat).astype(np.uint8)

    # Pick a feature subset for linkage
    n_vars = adata.n_vars
    if max_features is not None and n_vars > max_features:
        if feature_selection == "variance":
            # variance on 0/1 == p*(1-p)
            var = missing.var(axis=0)
            feat_idx = np.argsort(var)[::-1][:max_features]
            feat_idx = np.sort(feat_idx)
        elif feature_selection == "random":
            rng = np.random.default_rng(random_seed)
            feat_idx = np.sort(rng.choice(n_vars, size=max_features, replace=False))
        else:
            raise ValueError(f"Unknown feature selection strategy: {feature_selection}")
    else:
        feat_idx = np.arange(n_vars)

    # work view for clustering computations
    Miss = missing[:, feat_idx].astype(np.float32, copy=False)

    # Cluster samples (rows in Miss)
    s_link = sch.linkage(Miss, method=hierarchical_method, metric=hierarchical_metric)
    leaves_s = sch.leaves_list(s_link)
    adata.uns['missing_sample_linkage'] = s_link
    adata.uns['missing_sample_order']   = adata.obs_names[leaves_s].tolist()

    # Cluster features (cols in Miss)
    f_link = sch.linkage(Miss.T, method=hierarchical_method, metric=hierarchical_metric)
    leaves_f = sch.leaves_list(f_link)
    adata.uns['missing_feature_linkage'] = f_link
    adata.uns['missing_feature_order']   = adata.var_names[feat_idx][leaves_f].tolist()

    # provenance
    if len(feat_idx) < n_vars:
        adata.uns['missing_feature_subset_indices'] = feat_idx
        adata.uns['missing_feature_selection']      = feature_selection
        adata.uns['missing_max_features']           = int(max_features)

    return adata
