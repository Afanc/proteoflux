"""Clustering utilities for PCA/UMAP and missingness patterns.

Provides:
  - run_clustering: PCA -> neighbors -> UMAP; hierarchical clustering on samples/features.
                   Supports optional feature capping via variance- or random-based selection,
                   while writing back sample-level results to the original AnnData.
  - run_clustering_missingness: hierarchical clustering on binary missingness masks.
"""

import numpy as np
import scipy.cluster.hierarchy as sch
from anndata import AnnData
import scanpy as sc
from typing import Optional, Dict, Any, Sequence
from proteoflux.utils.utils import log_time

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
        var = np.nanvar(M, axis=0)
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

    # choose feature subset (indices on original adata)
    feat_idx = _pick_feature_indices(adata, layer, max_features, feature_selection, random_seed)

    # work on a temporary view/copy if we subselect
    A = _maybe_subset(adata, feat_idx)

    # Keep references to names for downstream order storage
    samples = A.obs_names
    features = A.var_names

    # Select data matrix for centering step later (samples × features)
    data = A.layers[layer] if layer is not None else A.X  # samples × features

    # PCA
    sc.tl.pca(A, n_comps=n_pcs)
    adata.uns['pca'] = A.uns.get('pca', {})

    # UMAP
    if 'neighbors' not in A.uns:
        sc.pp.neighbors(A, n_pcs=n_pcs)
    sc.tl.umap(A, **(umap_kwargs or {}))

    # Hierarchical clustering on samples (using PCA space)
    pca_matrix = A.obsm['X_pca'][:, :n_pcs]  # shape: n_samples × n_pcs
    sample_linkage = sch.linkage(pca_matrix, method=hierarchical_method, metric=hierarchical_metric)
    sample_leaves = sch.leaves_list(sample_linkage)

    # Prepare data for hierarchical clustering on intensity & centered
    mats = {"intensity": data}
    Xc = data - np.nanmean(data, axis=0, keepdims=True)
    A.layers["centered"] = Xc
    mats["centered"] = Xc

    # Keep AnnData invariant: all layers are (n_obs × n_vars).
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
    adata.obsm['X_pca']  = A.obsm['X_pca']
    if 'X_umap' in A.obsm:
        adata.obsm['X_umap'] = A.obsm['X_umap']
    # neighbors graph lives in .uns/.obsp; copy if present
    if 'neighbors' in A.uns:
        adata.uns['neighbors'] = A.uns['neighbors']
        for k in ('distances', 'connectivities'):
            if k in A.obsp:
                adata.obsp[k] = A.obsp[k]

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
