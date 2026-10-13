import anndata as ad

def validate_analysis_input(adata: ad.AnnData) -> None:
    if adata.n_obs < 2:
        raise ValueError(
            f"Invalid study design: ProteoFlux requires at least 2 samples for analysis; found n_obs={adata.n_obs}."
        )


def validate_log2_input(adata: ad.AnnData, config: dict) -> None:
    """Reject declared incompatible scales; external AnnData callers must supply log2 values."""
    from proteoflux.utils.semantics import log2_normalization_methods
    for preprocessing in ((config or {}).get("preprocessing") or {},
                          adata.uns.get("preprocessing") or {}):
        log2_normalization_methods(preprocessing.get("normalization") or {})
        scale = preprocessing.get("quantitative_scale", "log2")
        if scale != "log2":
            raise ValueError(f"Quantitative inference requires log2 values, got declared scale {scale!r}.")
