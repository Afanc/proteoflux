"""PELSA curve-fitting workflow.

Initial scaffold: preprocessing + AnnData export only.
Curve fitting/statistics will be added here.
"""

import anndata as ad

from proteoflux.utils.utils import log_time, log_info


@log_time("PELSA pipeline")
def run_pelsa_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    if "Concentration" not in adata.obs.columns:
        raise ValueError(
            "PELSA pipeline requires adata.obs['Concentration']. "
            "Add a numeric 'Concentration' column to the annotation file."
        )

    adata = adata.copy()
    adata.obs["Concentration"] = adata.obs["Concentration"].astype(float)

    n_conc = int(adata.obs["Concentration"].nunique())
    log_info(
        f"PELSA scaffold: detected {n_conc} distinct concentration level(s). "
        "Curve fitting not yet implemented."
    )

    adata.uns.setdefault("analysis", {})
    adata.uns["analysis"]["analysis_type"] = "pelsa"
    adata.uns["analysis"]["analysis_method"] = "pelsa_curve_fit"
    adata.uns["pelsa"] = {
        "status": "preprocessing_only",
        "concentration_column": "Concentration",
        "n_concentrations": n_conc,
    }

    return adata
