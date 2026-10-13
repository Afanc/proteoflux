#!/usr/bin/env Rscript
# Run from the repository root. R/limma are reference tooling only.
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly=TRUE)
root <- if (length(args)) args[[1]] else "tests/reference/limma"
options(digits=17)
read_matrix <- function(path) as.matrix(read.csv(path, row.names=1, check.names=FALSE, na.strings="NA"))
bh_finite <- function(p) {
  q <- rep(NA_real_, length(p)); testable <- is.finite(p)
  q[testable] <- p.adjust(p[testable], method="BH")
  q
}
for (case in c("balanced", "shared", "batch", "equal_variance", "missing")) {
  folder <- file.path(root, case)
  y <- read_matrix(file.path(folder, "expression.csv"))
  design <- read_matrix(file.path(folder, "design.csv"))
  contrasts <- read_matrix(file.path(folder, "contrasts.csv"))
  stopifnot(identical(colnames(y), rownames(design)),
            identical(colnames(design), rownames(contrasts)))
  fit <- contrasts.fit(lmFit(y, design=design), contrasts=contrasts)
  moderated <- eBayes(fit) # ProteoFlux uses defaults: no trend, robust=FALSE.
  outputs <- lapply(seq_len(ncol(contrasts)), function(j) {
    se <- fit$stdev.unscaled[,j] * fit$sigma
    raw_t <- fit$coefficients[,j] / se
    raw_p <- 2 * pt(-abs(raw_t), df=fit$df.residual)
    data.frame(feature=rownames(y), contrast=colnames(contrasts)[j],
      log2fc=fit$coefficients[,j], df_residual=fit$df.residual,
      se_raw=se, t_raw=raw_t, p_raw=raw_p, q_raw=bh_finite(raw_p),
      se_ebayes=fit$stdev.unscaled[,j] * sqrt(moderated$s2.post),
      t_ebayes=moderated$t[,j], p_ebayes=moderated$p.value[,j],
      q_ebayes=bh_finite(moderated$p.value[,j]),
      s2_post=moderated$s2.post, df_total=moderated$df.total,
      df_prior=rep_len(moderated$df.prior, nrow(y)))
  })
  write.csv(do.call(rbind, outputs), file.path(folder, "expected.csv"), row.names=FALSE, na="NA")
}
inputs <- list.files(root, pattern="(expression|samples|design|contrasts)\\.csv$", recursive=TRUE, full.names=TRUE)
write.table(data.frame(file=sub(paste0(root, "/"), "", inputs, fixed=TRUE), md5=unname(tools::md5sum(inputs))),
            file.path(root, "input_checksums.tsv"), sep="\t", row.names=FALSE, quote=FALSE)
writeLines(c(paste("R:", R.version.string), paste("limma:", as.character(packageVersion("limma"))),
             capture.output(sessionInfo())), file.path(root, "sessionInfo.txt"))
