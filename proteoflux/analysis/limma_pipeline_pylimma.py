import anndata as ad
from pylimma import fit
from proteoflux.utils.utils import log_time

@log_time("pyLimma pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    use_r = config.get("analysis", {}).get("use_r_limma", False)
    if use_r:
        from proteoflux.analysis.r_limma_pipeline import run_r_limma_pipeline
        return run_r_limma_pipeline(adata, config)

    # Extract design-related config values
    group = config["design"]["group_column"]
    remove_batch = config["design"].get("remove_batch", None)
    contrast_name = config["analysis"].get("contrast_name", None)
    use_voom = config["analysis"].get("use_voom", False)

    # Run full pyLimma pipeline
    adata_out, _, _ = fit(
        adata=adata,
        group=group,
        formula=None,
        design_matrix=None,
        contrast_name=contrast_name,
        remove_batch=remove_batch,
        use_voom=use_voom,
    )

    return adata_out
