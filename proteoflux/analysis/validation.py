import anndata as ad

def validate_analysis_input(adata: ad.AnnData) -> None:
    if adata.n_obs < 2:
        raise ValueError(
            f"Invalid study design: ProteoFlux requires at least 2 samples for analysis; found n_obs={adata.n_obs}."
        )
