import anndata as ad
import numpy as np
from typing import Optional

def compute_nrsc(
    adata: ad.AnnData,
    contrast_names: list[str],
) -> Optional[np.ndarray]:
    """
    Compute normalized relative spectral counts per feature and contrast:

        nrSC = (a - b) / (a + b)

    where a and b are the summed per-sample spectral counts on each side of
    the contrast. Returns a (n_vars x n_contrasts) float32 array, or None
    when the spectral_counts layer is unavailable.
    """
    if "spectral_counts" not in adata.layers:
        log_warning("Skipping nrSC: missing adata.layers['spectral_counts']")
        return None

    sc = np.asarray(adata.layers["spectral_counts"], dtype=np.float32)
    if sc.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"Unexpected spectral_counts layer shape {sc.shape}, "
            f"expected {(adata.n_obs, adata.n_vars)}"
        )

    cond = adata.obs["CONDITION"].astype(str).to_numpy()
    out = np.full((adata.n_vars, len(contrast_names)), np.nan, dtype=np.float32)

    for j, cname in enumerate(contrast_names):
        if "_vs_" not in cname:
            raise ValueError(f"Contrast name {cname!r} does not match expected '<A>_vs_<B>'")
        A, B = cname.split("_vs_", 1)
        a = np.nansum(sc[cond == A, :], axis=0, dtype=np.float64)
        b = np.nansum(sc[cond == B, :], axis=0, dtype=np.float64)
        denom = a + b
        ok = denom > 0
        out[ok, j] = ((a[ok] - b[ok]) / denom[ok]).astype(np.float32, copy=False)
    return out
