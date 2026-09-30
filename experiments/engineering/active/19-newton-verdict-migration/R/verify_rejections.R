#!/usr/bin/env Rscript
# Check every fit the Newton check newly rejects: the objective lavaan reaches
# with the same estimator (comparable for GLS and ULS; FIML's constant differs),
# and a refit under complete-data ML's tighter stopping controls.
# Usage: Rscript R/verify_rejections.R [results-dir]
suppressPackageStartupMessages({library(magmaanlab); library(lavaan)})
`%||%` <- function(x, y) if (is.null(x)) y else x

args <- commandArgs(TRUE)
here <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])))
source(file.path(here, "../../../../_support/R/helpers.R"))
set_single_threaded_math()
root <- normalizePath(corpus_root()); man <- read.csv(file.path(root, "manifest.csv"))
res <- if (length(args)) args[[1]] else file.path(here, "../results/current")
source(file.path(here, "corpus.R"))
f <- read.csv(file.path(res, "fits.csv"), stringsAsFactors = FALSE)
ok <- f[f$estimator != "none" & !is.na(f$converged), ]
r <- ok[ok$first_order %in% TRUE & !(ok$converged %in% TRUE), ]
tight <- list(nlopt = list(ftol_rel = 1e-12, xtol_rel = 1e-10, max_eval = 5000L))
out <- list()
for (i in seq_len(nrow(r))) {
  id <- r$case[i]; est <- r$estimator[i]
  dir <- man$case_dir[man$case_id == id]
  meta <- jsonlite::fromJSON(file.path(root, dir, "meta.json"))
  case <- read_case(root, dir)
  syn <- paste(readLines(file.path(root, dir, "model.lav")), collapse = "\n")
  if (identical(case$kind, "ordinal")) {
    fun <- getExportedValue("lavaan", meta$lavaan_function %||% "sem")
    lf <- tryCatch(suppressWarnings(fun(syn, data = case$raw, ordered = case$ordered,
                                        estimator = "WLSMV", parameterization = case$parameterization,
                                        se = "none", test = "standard")), error = function(e) NULL)
    lav_f <- if (is.null(lf) || !lavInspect(lf, "converged")) NA else
      tryCatch(unname(fitMeasures(lf, "fmin")), error = function(e) NA)
    refit <- tryCatch(suppressWarnings(fit_model(case$syntax, case$raw, estimator = "DWLS",
      ordered = case$ordered, parameterization = case$parameterization, control = tight)),
      error = function(e) NULL)
    out[[i]] <- data.frame(case = id, estimator = est, status = r$newton_status[i],
      d = signif(r$distance[i], 3), f = signif(r$fmin[i], 6), lavaan_f = signif(lav_f, 6),
      lav_conv = if (is.null(lf)) NA else lavInspect(lf, "converged"),
      tight_f = if (is.null(refit)) NA else signif(refit$fmin, 6),
      tight_conv = if (is.null(refit)) NA else isTRUE(refit$converged),
      tight_d = if (is.null(refit)) NA else signif(refit$diagnostics$newton_accuracy$distance %||% NA, 3))
    next
  }
  largs <- list(model = syn, estimator = if (est == "FIML") "ML" else est,
                meanstructure = isTRUE(meta$model_options$meanstructure), fixed.x = isTRUE(meta$model_options$fixed_x),
                se = "none", test = "standard")
  if (est == "FIML") { largs$data <- case$raw; largs$missing <- "ml" } else {
    largs$sample.cov <- if (length(case$sample$S) == 1) case$sample$S[[1]] else case$sample$S
    largs$sample.nobs <- case$sample$nobs
    if (!is.null(case$sample$mean)) largs$sample.mean <- if (length(case$sample$mean) == 1) case$sample$mean[[1]] else case$sample$mean
  }
  for (k in c("group_equal","group_partial")) if (length(meta$model_options[[k]])) largs[[gsub("_",".",k)]] <- meta$model_options[[k]]
  fun <- getExportedValue("lavaan", meta$lavaan_function %||% "sem")
  lf <- tryCatch(suppressWarnings(do.call(fun, largs)), error = function(e) NULL)
  lav_f <- if (is.null(lf) || !lavInspect(lf, "converged")) NA else
    tryCatch(unname(fitMeasures(lf, "fmin")), error = function(e) NA)
  lav_conv <- if (is.null(lf)) NA else lavInspect(lf, "converged")
  refit <- tryCatch(suppressWarnings(switch(est,
    GLS = magmaan_core$fit_gls(case$model, case$sample, control = tight),
    ULS = magmaan_core$fit_uls(case$model, case$sample, control = tight),
    FIML = magmaan_core$fit_fiml(case$model, df_to_fiml_data(case$raw, case$model), control = tight))), error = function(e) NULL)
  out[[i]] <- data.frame(case = id, estimator = est, status = r$newton_status[i],
    d = signif(r$distance[i], 3), f = signif(r$fmin[i], 6), lavaan_f = signif(lav_f, 6), lav_conv = lav_conv,
    tight_f = if (is.null(refit)) NA else signif(refit$fmin, 6),
    tight_conv = if (is.null(refit)) NA else isTRUE(refit$converged),
    tight_d = if (is.null(refit)) NA else signif(refit$diagnostics$newton_accuracy$distance %||% NA, 3))
}
rej <- do.call(rbind, out)
utils::write.csv(rej, file.path(res, "rejections.csv"), row.names = FALSE)
cat("Wrote", file.path(res, "rejections.csv"), "\n")
