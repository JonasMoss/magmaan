#!/usr/bin/env Rscript
# Misspecification-robust standard errors for ordinal DWLS: does the
# observed-Hessian bread ("robust" regime) track the empirical sampling SD when
# the model is wrong, while the expected / Gauss-Newton bread ("model" regime)
# underestimates it -- and do the two coincide when the model is correct?
#
# Both regimes use the SAME (empirical NACOV) meat, so they differ ONLY in the
# bread. We report the raw focal parameter and its standardized counterpart.
#
# Usage:
#   Rscript run_experiment.R [--reps 300] [--n-total 1000]
#                            [--cross 0.3] [--seed-base S] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

parse_args <- function(args) {
  out <- list(reps = 300L, n_total = 1000L, cross = 0.3,
              seed_base = 20260619L, smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n-total N] ",
          "[--cross C] [--seed-base S] [--smoke]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n-total") { i <- i + 1L; out$n_total <- as.integer(args[[i]])
    } else if (startsWith(a, "--n-total=")) {
      out$n_total <- as.integer(sub("^--n-total=", "", a))
    } else if (a == "--cross") { i <- i + 1L; out$cross <- as.numeric(args[[i]])
    } else if (startsWith(a, "--cross=")) {
      out$cross <- as.numeric(sub("^--cross=", "", a))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else { stop("unknown argument: ", a, call. = FALSE) }
    i <- i + 1L
  }
  if (out$smoke) { out$reps <- 30L; out$n_total <- 500L }
  if (!is.finite(out$reps) || out$reps < 2L) stop("--reps must be >= 2")
  if (!is.finite(out$n_total) || out$n_total < 120L) stop("--n-total >= 120")
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
res_dir <- ensure_results_dir()

ov <- paste0("x", 1:6)
thresholds <- c(-0.8, 0.0, 0.8)        # four-category items
loading <- 0.7
fcor <- 0.3                            # true f1<->f2 correlation
model <- "f1 =~ x1 + x2 + x3\nf2 =~ x4 + x5 + x6"

# Data-generating model: a clean two-factor thresholded-normal CFA, optionally
# with a cross-loading of x4 on f1 (`cross`). The fitted model never includes
# that cross-loading, so `cross > 0` is a (Stage-2) structural misspecification
# while latent normality holds exactly.
gen <- function(n, cross) {
  f1 <- rnorm(n)
  f2 <- fcor * f1 + sqrt(1 - fcor^2) * rnorm(n)
  resid_sd <- function(load) sqrt(max(1e-3, 1 - load^2))
  x4_resid <- sqrt(max(1e-3, 1 - cross^2 - loading^2))
  z <- cbind(
    loading * f1 + rnorm(n, sd = resid_sd(loading)),
    loading * f1 + rnorm(n, sd = resid_sd(loading)),
    loading * f1 + rnorm(n, sd = resid_sd(loading)),
    cross * f1 + loading * f2 + rnorm(n, sd = x4_resid),
    loading * f2 + rnorm(n, sd = resid_sd(loading)),
    loading * f2 + rnorm(n, sd = resid_sd(loading)))
  d <- as.data.frame(lapply(seq_len(6), function(j)
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))))
  names(d) <- ov
  d
}

focal_index <- function(fit) {
  pt <- fit$partable
  row <- which(pt$op == "~~" & pt$lhs == "f1" & pt$rhs == "f2")
  list(row = row, free = pt$free[row])
}

one_rep <- function(n, cross, seed) {
  set.seed(seed)
  d <- gen(n, cross)
  fit <- tryCatch(magmaan::magmaan(model, d, estimator = "DWLS", ordered = ov),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  fx <- focal_index(fit)
  if (!length(fx$row) || is.na(fx$free) || fx$free < 1) return(NULL)
  Vm <- tryCatch(vcov(fit, regime = "model"), error = function(e) NULL)
  Vr <- tryCatch(vcov(fit, regime = "robust"), error = function(e) NULL)
  if (is.null(Vm) || is.null(Vr)) return(NULL)
  fi <- fx$free
  sm <- tryCatch(magmaan::standardized(fit, Vm, type = "all"), error = function(e) NULL)
  sr <- tryCatch(magmaan::standardized(fit, Vr, type = "all"), error = function(e) NULL)
  out <- data.frame(
    est_raw = fit$partable$est[fx$row],
    se_model_raw = sqrt(Vm[fi, fi]),
    se_robust_raw = sqrt(Vr[fi, fi]),
    est_std = if (!is.null(sm)) sm$theta[fi] else NA_real_,
    se_model_std = if (!is.null(sm)) sm$se[fi] else NA_real_,
    se_robust_std = if (!is.null(sr)) sr$se[fi] else NA_real_)
  if (any(!is.finite(unlist(out)))) return(NULL)
  out
}

run_cell <- function(cell, cross, seed_off) {
  rows <- vector("list", cfg$reps)
  for (r in seq_len(cfg$reps)) {
    rows[[r]] <- one_rep(cfg$n_total, cross, cfg$seed_base + seed_off + r)
  }
  rows <- Filter(Negate(is.null), rows)
  D <- do.call(rbind, rows)
  mk <- function(kind, est, sem, ser) data.frame(
    cell = cell, param_kind = kind, n_total = cfg$n_total, cross = cross,
    reps = nrow(D), emp_sd = sd(est),
    se_model = mean(sem), se_robust = mean(ser),
    ratio_model = mean(sem) / sd(est), ratio_robust = mean(ser) / sd(est),
    mean_est = mean(est), stringsAsFactors = FALSE)
  rbind(
    mk("raw", D$est_raw, D$se_model_raw, D$se_robust_raw),
    mk("std", D$est_std, D$se_model_std, D$se_robust_std))
}

summary <- rbind(
  run_cell("null", 0.0, 0L),
  run_cell("misspec", cfg$cross, 1000000L))
write_csv(summary, file.path(res_dir, "se_regime.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps, n_total = cfg$n_total, cross = cfg$cross,
    seed_base = cfg$seed_base, estimator = "DWLS_ordinal_delta",
    model = "two_factor_six_indicator_ordinal",
    focal = "f1~~f2 (raw) and its std.all counterpart",
    regimes = "model=expected/Gauss-Newton bread; robust=observed-Hessian bread (same empirical NACOV meat)",
    smoke = cfg$smoke),
  packages = "magmaan")

print(summary[, c("cell", "param_kind", "reps", "emp_sd", "se_model",
                  "se_robust", "ratio_model", "ratio_robust")],
      row.names = FALSE, digits = 4)
cat("\nWrote results to ", res_dir, "\n", sep = "")
