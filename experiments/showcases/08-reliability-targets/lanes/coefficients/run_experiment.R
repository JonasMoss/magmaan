#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
root <- dirname(normalizePath(script, mustWork = TRUE))
if (identical(args, "--help")) {
  cat("Deterministic reliability targets: alpha, lambda6, SG omega and fitted omega.\n",
      "Usage: [--smoke] [--results-dir DIR] (default results)\n",
      "No finite-N simulation or interval coverage is computed.\n")
  quit(status = 0L)
}
out <- file.path(root, "results")
args <- args[args != "--smoke"]
if (length(args)) {
  if (length(args) != 2L || args[[1L]] != "--results-dir") stop("see --help")
  out <- if (startsWith(args[[2L]], "/")) args[[2L]] else file.path(root, args[[2L]])
}
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
suppressPackageStartupMessages(library(magmaanlab))
magmaan_core <- magmaanlab::magmaan_core
# Population construction and functionals retained from former research/16.
ov <- paste0("y", 1:6)
p <- length(ov)

omega_model_syntax <- function(ov) {
  p <- length(ov)
  loads <- paste0("l", seq_len(p), "*", ov)
  resid <- paste0(ov, " ~~ e", seq_len(p), "*", ov)
  lsum <- paste0("l", seq_len(p), collapse = " + ")
  esum <- paste0("e", seq_len(p), collapse = " + ")
  paste(c(
    paste0("f =~ ", paste(loads, collapse = " + ")),
    resid,
    paste0("omega := (", lsum, ")^2 / ((", lsum, ")^2 + ", esum, ")")
  ), collapse = "\n")
}
omega_model <- omega_model_syntax(ov)

unit_total_reliability <- function(common, Sigma) {
  sum(common) / sum(Sigma)
}

population_one_factor <- function() {
  lambda <- c(0.85, 0.75, 0.70, 0.65, 0.60, 0.55)
  common <- tcrossprod(lambda)
  psi <- 1.0 - lambda^2
  Sigma <- common + diag(psi, p)
  dimnames(Sigma) <- list(ov, ov)
  list(name = "one_factor", label = "Correct one-factor",
       Sigma = Sigma, common = common,
       true_reliability = unit_total_reliability(common, Sigma))
}

population_two_factor <- function() {
  Lambda <- matrix(0, p, 2)
  Lambda[1:3, 1] <- c(0.85, 0.75, 0.65)
  Lambda[4:6, 2] <- c(0.80, 0.70, 0.60)
  Phi <- matrix(c(1.0, 0.35, 0.35, 1.0), 2, 2)
  common <- Lambda %*% Phi %*% t(Lambda)
  psi <- 1.0 - diag(common)
  Sigma <- common + diag(psi, p)
  dimnames(Sigma) <- list(ov, ov)
  list(name = "two_factor_fit_as_one", label = "Two-factor fit as one",
       Sigma = Sigma, common = common,
       true_reliability = unit_total_reliability(common, Sigma))
}

sample_stats_from_cov <- function(Sigma, n = 1000000L) {
  list(S = list(Sigma), mean = list(rep(0, nrow(Sigma))), nobs = as.integer(n))
}

defined_omega <- function(fit, vcov) {
  d <- magmaanlab::compute_defined(omega_model, fit, vcov)
  as.numeric(d$est[d$lhs == "omega" & d$op == ":="][1L])
}

defined_omega_se <- function(fit, vcov) {
  d <- magmaanlab::compute_defined(omega_model, fit, vcov)
  as.numeric(d$se[d$lhs == "omega" & d$op == ":="][1L])
}

population_omega_target <- function(Sigma) {
  fit <- magmaanlab::fit_model(omega_model, sample_stats_from_cov(Sigma),
                          estimator = "ML", std_lv = TRUE)
  if (!isTRUE(fit$converged)) {
    stop("population one-factor omega target did not converge", call. = FALSE)
  }
  V <- magmaan_core$inference_vcov(
    magmaan_core$inference_information_expected(fit), fit)
  defined_omega(fit, V)
}

covariance_targets <- function(pop) {
  tab <- magmaan_core$measures_reliability_cov(pop$Sigma)$table
  out <- data.frame(
    population = pop$name,
    population_label = pop$label,
    method = as.character(tab$coefficient),
    target_functional = as.numeric(tab$value),
    target_kind = "covariance functional",
    true_reliability = pop$true_reliability,
    stringsAsFactors = FALSE
  )
  omega_target <- population_omega_target(pop$Sigma)
  rbind(out, data.frame(
    population = pop$name,
    population_label = pop$label,
    method = "fitted_omega",
    target_functional = omega_target,
    target_kind = "one-factor ML pseudo-target",
    true_reliability = pop$true_reliability,
    stringsAsFactors = FALSE
  ))
}


pops <- list(population_one_factor(), population_two_factor())
targets <- do.call(rbind, lapply(pops, covariance_targets))
stopifnot(all(is.finite(targets$target_functional)), all(is.finite(targets$true_reliability)))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.csv(targets, file.path(out, "population_targets.csv"), row.names = FALSE)
write.csv(data.frame(key = c("scope", "source", "package_version", "r_version", "command", "source_md5", "seed"),
  value = c("deterministic population targets; no coverage", "former research/16 populations",
    as.character(packageVersion("magmaanlab")), R.version.string,
    paste(commandArgs(), collapse = " "), unname(tools::md5sum(script)), "none; deterministic")),
  file.path(out, "metadata.csv"), row.names = FALSE)
cat("Wrote deterministic targets to", out, "\n")
print(targets)
