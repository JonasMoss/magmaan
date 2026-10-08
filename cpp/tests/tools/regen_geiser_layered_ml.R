#!/usr/bin/env Rscript
# TASK-132: ML oracle on the retained GLS fixture's identical summary moments.
# Run from the repository root; no external corpus or magmaan package needed.
suppressPackageStartupMessages({library(jsonlite); library(lavaan)})
pin <- trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))
version <- as.character(packageVersion("lavaan"))
stopifnot(gsub("-", ".", pin, fixed = TRUE) == version)
source <- "cpp/tests/fixtures/geiser/gls_reference.json"
cases <- read_json(source, simplifyVector = FALSE)$cases
c <- Filter(function(x) x$id == "latent_ar_cross_lagged_extended", cases)[[1]]
S <- do.call(rbind, lapply(c$sample_cov, unlist))
mu <- unlist(c$sample_mean)
nm <- unlist(c$ov_names)
dimnames(S) <- list(nm, nm)
names(mu) <- nm
fit <- sem(c$model, sample.cov = S, sample.mean = mu,
           sample.nobs = c$n_obs, sample.cov.rescale = FALSE,
           meanstructure = c$meanstructure, fixed.x = c$fixed_x,
           estimator = "ML", se = "none")
stopifnot(lavInspect(fit, "converged"))
im <- fitted(fit)
c$estimator <- "ML"
c$lavaan <- list(version = version, converged = TRUE,
                 fx = unname(fitMeasures(fit, "fmin")),
                 sigma = unname(im$cov[nm, nm]),
                 mu = unname(im$mean[nm]), theta = unname(coef(fit)))
write_json(list(`_meta` = list(format_version = 1, fixture_kind = "geiser.ml",
           tool = "cpp/tests/tools/regen_geiser_layered_ml.R",
           source_fixture = source, lavaan_version = version), cases = list(c)),
           "cpp/tests/fixtures/geiser/layered_ml_reference.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 16)
