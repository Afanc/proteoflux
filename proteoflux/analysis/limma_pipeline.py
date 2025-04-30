import anndata as ad
from proteoflux.design.designmatrixbuilder import DesignMatrixBuilder
from proteoflux.design.contrast import apply_contrasts
from proteoflux.design.contrastbuilder import ContrastBuilder
from proteoflux.analysis.linearmodelfitter import LinearModelFitter
from proteoflux.analysis.statisticaltester import StatisticalTester
from proteoflux.analysis.ebayes_moderator import EbayesModerator
from proteoflux.utils.utils import logger, log_time

@log_time("pyLimma pipeline")
def run_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    use_r = config.get("analysis", {}).get("use_r_limma", False)
    if use_r:
        from proteoflux.analysis.r_limma_pipeline import run_r_limma_pipeline
        return run_r_limma_pipeline(adata, config)

    builder = DesignMatrixBuilder(adata.obs, config["design"])
    X, design_info = builder.build()

    contrast_builder = ContrastBuilder(design_info)
    C, contrast_names = contrast_builder.make_all_pairwise_contrasts()
    adata.uns["contrast_names"] = contrast_names

    fitter = LinearModelFitter(adata.X, X)
    results = fitter.fit().get_results()

    logfc, se, v_j = apply_contrasts(results, C, contrast_names)

    # eBayes
    ebayes_method = config.get("analysis", {}).get("ebayes_method", "moments")
    moderator = EbayesModerator(
        results["residual_variance"],
        results["df_residual"],
        method=ebayes_method)
    moderator.fit()

    # Classical stats
    tester = StatisticalTester(
        logfc,
        se,
        results["df_residual"],
        contrast_names,
        adata.var.index,
        df_prior = moderator.d0)

    stats = tester.compute()

    # Save all the data
    adata.uns['residuals'] = results['residuals']
    adata.uns['residual_variance'] = results['residual_variance']
    adata.varm["log2fc"] = logfc
    adata.varm["t"] = stats["t"]
    adata.varm["p"] = stats["p"]
    adata.varm["q"] = stats["q"]

    bayes_stats = moderator.apply_to_contrasts(logfc, v_j)

    adata.varm["t_ebayes"] = bayes_stats["t_ebayes"]
    adata.varm["p_ebayes"] = bayes_stats["p_ebayes"]
    adata.varm["q_ebayes"] = bayes_stats["q_ebayes"]

    # missingness per contrast
    raw_matrix = adata.layers["raw"].T
    conditions = adata.obs[config["design"]["group_column"]].values
    miss = tester.compute_missingness(raw_matrix, conditions)
    adata.uns["missingness"] = miss

    return adata
