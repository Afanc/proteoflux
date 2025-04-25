import anndata as ad
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr, PackageNotInstalledError
from proteoflux.design.designmatrixbuilder import DesignMatrixBuilder
from proteoflux.design.contrastbuilder import ContrastBuilder
from proteoflux.analysis.statisticaltester import StatisticalTester

pandas2ri.activate()

try:
    limma = importr("limma")
except PackageNotInstalledError:
    raise RuntimeError(
        "The R package 'limma' is not installed. Please run:\n"
        "    R -e \"if (!requireNamespace('BiocManager')) install.packages('BiocManager'); "
        "BiocManager::install('limma')\"\n"
        "from your shell (with your R environment activated)."
    )

def run_r_limma_pipeline(adata: ad.AnnData, config: dict) -> ad.AnnData:
    limma = rpackages.importr("limma")
    stats = rpackages.importr("stats")

    X = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    X_design, design_info = DesignMatrixBuilder(adata.obs, config["design"]).build()
    contrast_df, contrast_names = ContrastBuilder(design_info).make_all_pairwise_contrasts()

    # Pass to R
    # Gene expression: genes x samples
    ro.globalenv["X"] = pandas2ri.py2rpy(pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names))

    # Design: samples x covariates
    ro.globalenv["design"] = pandas2ri.py2rpy(pd.DataFrame(X_design, columns=design_info.column_names))

    # Contrasts: covariates x contrasts
    ro.globalenv["contrasts"] = pandas2ri.py2rpy(
        pd.DataFrame(contrast_df.astype(float), columns=contrast_names, index=design_info.column_names)
    )

    ro.r("contrasts <- as.matrix(contrasts)") #for some reason gotta convert here

    ro.r("fit <- lmFit(X, design)")
    ro.r("fit <- contrasts.fit(fit, contrasts)")
    ro.r("fit_raw <- fit")  # pre-ebayes
    ro.r("fit <- eBayes(fit)")

    # print the first 6 genes of the three key vectors in R:
    #ro.r('cat("=== R:   first 6 s2.post  \\n");    print(head(fit$s2.post))')
    #ro.r('cat("=== R:   first 6 se_moderated  \\n");'
    #     ' print(head(fit$stdev.unscaled[,1] * sqrt(fit$s2.post)))')
    #ro.r('cat("=== R:   first 6 t_ebayes  \\n");     print(head(fit$t))')

    logfc_r = np.array(ro.r("fit$coefficients"))
    p_ebayes = np.array(ro.r("fit$p.value")).T  # shape: (n_contrasts, n_genes)

    q_ebayes = np.vstack([
        multipletests(p, method="fdr_bh")[1]
        for p in p_ebayes
    ])

    t_ebayes = np.atleast_2d(np.array(ro.r("fit$t")))  # shape: (n_genes, n_contrasts)

    # limma doesn't provide pre ebayes stats, gotta compute by hand
    ro.r("""
    fit_raw$t <- fit_raw$coefficients / fit_raw$stdev.unscaled / sqrt(fit_raw$sigma^2)
    fit_raw$p.value <- 2 * pt(-abs(fit_raw$t), df=fit_raw$df.residual)
    fit$t <- fit$coefficients / fit$stdev.unscaled / sqrt(fit$sigma^2)
    """)
    p_raw = np.array(ro.r("fit_raw$p.value"))
    p_raw = np.atleast_2d(p_raw).T
    q_raw = np.vstack([
        multipletests(p, method="fdr_bh")[1]
        for p in p_raw
    ])

    #print(np.array(ro.r("fit$df.prior")))
    #print(np.array(ro.r("summary(fit$sigma^2)")))
    #print(np.array(ro.r("var(log(fit$sigma^2))")))
    #print("R s2.prior:", ro.r("fit$s2.prior")[0])
    #print("R df.prior:", ro.r("fit$df.prior")[0])
    #print(ro.r("head(fit$p.value[, 1])"))

    t_raw = np.array(ro.r("fit$t"))

    adata.uns["contrast_names"] = contrast_names
    adata.varm["log2fc"] = logfc_r
    adata.varm["p_ebayes"] = p_ebayes.T
    adata.varm["q_ebayes"] = q_ebayes.T
    adata.varm["t_ebayes"] = t_ebayes
    adata.varm["p"] = p_raw.T
    adata.varm["q"] = q_raw.T
    adata.varm["t"] = t_raw

    # just to compute missingness
    qvals = adata.layers["qvalue"].T
    conditions = adata.obs[config["design"]["group_column"]].values

    # Create dummy tester to reuse the missingness logic
    tester = StatisticalTester(
        log2fc=adata.varm["log2fc"],
        se=None,  # Not needed for missingness
        df_residual=None, # Not needed
        contrast_names=adata.uns["contrast_names"],
        protein_ids=adata.var_names
    )

    miss = tester.compute_missingness(qvals, conditions)
    adata.uns["missingness"] = miss

    return adata

