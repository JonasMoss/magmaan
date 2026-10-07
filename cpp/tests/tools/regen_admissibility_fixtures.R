#!/usr/bin/env Rscript
# Run from the root; optional output directory allows reproducibility checks.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) <= 1L)
out <- if (length(args)) args[[1L]] else "cpp/tests/fixtures/admissibility"
version <- as.character(utils::packageVersion("lavaan"))
pin <- gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt")))
stopifnot(version == pin)
model <- "f =~ x1 + x2 + x3"
options <- list(estimator = "ML", meanstructure = FALSE,
                sample.cov.rescale = FALSE, std.lv = FALSE,
                sample.nobs = 500L, se = "none", test = "none")
cases <- list(improper = c(.8, .8, .3), proper = c(.6, .5, .4))
references <- lapply(names(cases), function(id) {
  off <- cases[[id]]
  s <- diag(3)
  s[1, 2] <- s[2, 1] <- off[1]
  s[1, 3] <- s[3, 1] <- off[2]
  s[2, 3] <- s[3, 2] <- off[3]
  dimnames(s) <- list(paste0("x", 1:3), paste0("x", 1:3))
  # Independent marker algebra: s12=lambda2*psi, s13=lambda3*psi,
  # s23=lambda2*lambda3*psi; residuals complete the unit diagonal.
  psi <- off[1] * off[2] / off[3]
  lambda <- c(1, off[3] / off[2], off[3] / off[1])
  residual <- 1 - lambda^2 * psi
  determinant <- 1 + 2 * prod(off) - sum(off^2)
  stopifnot(determinant > 0, abs(det(s) - determinant) < 1e-14,
            max(abs(tcrossprod(lambda) * psi + diag(residual) - s)) < 1e-14)
  if (id == "improper") stopifnot(abs(determinant - .014) < 1e-14,
                                 abs(residual[1] + 17/15) < 1e-14)
  warnings <- character()
  capture <- function(expr) withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  fit <- capture(do.call(lavaan::cfa, c(list(model = model, sample.cov = s), options)))
  post_check <- capture(lavaan::lavInspect(fit, "post.check"))
  pt <- lavaan::parTable(fit)
  list(case_id = id, model = model, sample_cov = unname(s),
       analytic = list(determinant = determinant, lambda = lambda,
                       psi = psi, residual = residual),
       converged = lavaan::lavInspect(fit, "converged"),
       post_check = post_check, warnings = warnings,
       parameters = pt[, c("lhs", "op", "rhs", "group", "free", "start", "est")],
       implied_cov = unname(lavaan::lavInspect(fit, "sigma.hat")),
       criterion = list(name = "lavaan fmin (half ML discrepancy)",
                        value = as.numeric(fit@optim$fx)))
})
fixture <- list(provenance = list(generator = "cpp/tests/tools/regen_admissibility_fixtures.R",
                                 source = "synthetic complete-data summary covariance",
                                 lavaan_version = version), options = options, cases = references)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(fixture, file.path(out, "reference.json"),
                     pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat("Generated paired admissibility oracle with lavaan", version, "\n")
