import numpy as np
import scipy.cluster.hierarchy as sch
from anndata import AnnData
import scanpy as sc
from typing import Optional, Dict, Any
from proteoflux.utils.utils import log_time

@log_time("Running clustering pipeline")
def run_clustering(
    adata: AnnData,
    layer: Optional[str] = None,
    n_pcs: int = 50,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    hierarchical_method: str = "ward",
    hierarchical_metric: str = "euclidean",
) -> AnnData:
    """
    Compute PCA, UMAP, and hierarchical clustering on an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Key of adata.layers to use; if None, uses adata.X.
    n_pcs
        Number of principal components to compute.
    umap_kwargs
        Additional keyword arguments for sc.tl.umap.
    hierarchical_method
        Linkage method for hierarchical clustering.
    hierarchical_metric
        Distance metric for hierarchical clustering.

    Returns
    -------
    adata
        The same AnnData object with new entries:
        - adata.obsm['X_pca']  : PCA coordinates
        - adata.obsm['X_umap'] : UMAP embedding
        - adata.uns['sample_linkage'] : linkage matrix (samples)
        - adata.uns['feature_linkage']: linkage matrix (features)
        - adata.uns['sample_order']   : ordered sample names
        - adata.uns['feature_order']  : ordered feature names
    """
    # Select data matrix
    data = adata.layers[layer] if layer is not None else adata.X  # samples × features
    samples, features = adata.obs_names, adata.var_names

    # 1) PCA
    sc.tl.pca(adata, n_comps=n_pcs)

    # 2) UMAP (requires neighbors)
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_pcs=n_pcs)
    sc.tl.umap(adata, **(umap_kwargs or {}))

    # 3) Hierarchical clustering on samples (using PCA space)
    pca_matrix = adata.obsm['X_pca'][:, :n_pcs]  # shape: n_samples × n_pcs
    sample_linkage = sch.linkage(pca_matrix, method=hierarchical_method, metric=hierarchical_metric)
    sample_leaves = sch.leaves_list(sample_linkage)
    sample_order = adata.obs_names[sample_leaves].tolist()

    # 3) Prepare data for hierarchical clustering
    mats = {"intensity": adata.X}
    Xc = adata.X - np.nanmean(adata.X, axis=0, keepdims=True)
    adata.layers["centered"] = Xc
    mats["centered"] = Xc

    for tag, M in mats.items():
        s_link = sch.linkage(M,   method=hierarchical_method, metric=hierarchical_metric)
        f_link = sch.linkage(M.T, method=hierarchical_method, metric=hierarchical_metric)
        s_leaves = sch.leaves_list(s_link)
        f_leaves = sch.leaves_list(f_link)

        adata.uns[f"{tag}_sample_linkage"]  = s_link
        adata.uns[f"{tag}_feature_linkage"] = f_link
        adata.uns[f"{tag}_sample_order"]    = samples[s_leaves].tolist()
        adata.uns[f"{tag}_feature_order"]   = features[f_leaves].tolist()

    return adata

@log_time("Running missingness clustering pipeline")
def run_clustering_missingness(
    adata: AnnData,
    layer: str = "normalized",
    hierarchical_method: str = "ward",
    hierarchical_metric: str = "euclidean",
) -> AnnData:
    """
    Compute hierarchical clustering on the missingness pattern of adata[layer].

    - adata.uns['missing_sample_linkage']: linkage matrix for samples
    - adata.uns['missing_sample_order']  : list of sample names in clustered order
    - adata.uns['missing_feature_linkage']: linkage matrix for features
    - adata.uns['missing_feature_order']  : list of feature names in clustered order
    """
    # 1) Build binary missingness: samples × features
    mat = adata.layers[layer]
    missing = np.isnan(mat).astype(int)

    # 2) Cluster samples (columns in your heatmap)
    sample_linkage = sch.linkage(missing, method=hierarchical_method, metric=hierarchical_metric)
    leaves_s       = sch.leaves_list(sample_linkage)
    adata.uns['missing_sample_linkage'] = sample_linkage
    adata.uns['missing_sample_order']   = adata.obs_names[leaves_s].tolist()

    # 3) Cluster features (rows in your heatmap)
    feature_linkage = sch.linkage(missing.T, method=hierarchical_method, metric=hierarchical_metric)
    leaves_f        = sch.leaves_list(feature_linkage)
    adata.uns['missing_feature_linkage'] = feature_linkage
    #adata.uns['missing_feature_order']   = adata.var_names[leaves_f].tolist()
    adata.uns['missing_feature_order'] = adata.var_names.tolist()

    return adata
