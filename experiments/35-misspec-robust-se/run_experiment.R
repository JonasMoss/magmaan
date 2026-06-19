#!/usr/bin/env Rscript
# Does the observed-Hessian bread ("robust" regime) recover the true sampling SD
# of a focal loading when the structural model is wrong, while the expected /
# Gauss-Newton bread ("model" regime) underestimates it -- and do the two
# coincide when the model is correct? We look at the raw loading AND its
# standardized counterpart, for continuous ML and ordinal DWLS on the same
# latent design. Both regimes use the SAME (empirical) meat, so they differ ONLY
# in the bread.
#
# Usage:
#   Rscript run_experiment.R [--reps 250] [--n-total 600]
#                            [--cross 0.4] [--seed-base S] [--smoke]

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
  out <- list(reps = 250L, n_total = 600L, cross = 0.4,
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
  if (out$smoke) { out$reps <- 30L; out$n_total <- 400L }
  if (!is.finite(out$reps) || out$reps < 2L) stop("--reps must be >= 2")
  if (!is.finite(out$n_total) || out$n_total < 120L) stop("--n-total >= 120")
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
res_dir <- ensure_results_dir()

ov <- paste0("y", 1:6)
thresholds <- c(-0.8, 0.0, 0.8)
loading <- 0.7
fcor <- 0.3
model <- "f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6"
focal <- list(lhs = "f2", op = "=~", rhs = "y5")  # free loading on the distorted factor

# Latent design: a clean two-factor model, optionally with a cross-loading of y4
# on f1 (`cross`) that the fitted model omits. The fitted model never includes
# the cross-loading, so `cross > 0` is a Stage-2 structural misspecification; the
# latent-normal measurement model is correct throughout. `gen_z` returns the
# continuous indicators; the ordinal arm thresholds them into four categories.
gen_z <- function(n, cross) {
  f1 <- rnorm(n)
  f2 <- fcor * f1 + sqrt(1 - fcor^2) * rnorm(n)
  rsd <- sqrt(max(1e-3, 1 - loading^2))
  y4sd <- sqrt(max(1e-3, 1 - cross^2 - loading^2))
  z <- cbind(
    loading * f1 + rnorm(n, sd = rsd), loading * f1 + rnorm(n, sd = rsd),
    loading * f1 + rnorm(n, sd = rsd),
    cross * f1 + loading * f2 + rnorm(n, sd = y4sd),
    loading * f2 + rnorm(n, sd = rsd), loading * f2 + rnorm(n, sd = rsd))
  colnames(z) <- ov
  z
}
as_continuous <- function(z) as.data.frame(z)
as_ordinal <- function(z) {
  d <- as.data.frame(lapply(seq_len(6), function(j)
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))))
  names(d) <- ov
  d
}

focal_free <- function(fit) {
  pt <- fit$partable
  r <- which(pt$lhs == focal$lhs & pt$op == focal$op & pt$rhs == focal$rhs)
  list(row = r, free = pt$free[r])
}

one_rep <- function(estimator, n, cross, seed) {
  set.seed(seed)
  z <- gen_z(n, cross)
  if (estimator == "ML") {
    d <- as_continuous(z)
    fit <- tryCatch(magmaan::magmaan(model, d, estimator = "ML"),
                    error = function(e) NULL)
  } else {
    d <- as_ordinal(z)
    fit <- tryCatch(magmaan::magmaan(model, d, estimator = "DWLS", ordered = ov),
                    error = function(e) NULL)
  }
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  fx <- focal_free(fit)
  if (!length(fx$row) || is.na(fx$free) || fx$free < 1) return(NULL)
  data_arg <- if (estimator == "ML") d else NULL
  Vm <- tryCatch(vcov(fit, regime = "model", data = data_arg), error = function(e) NULL)
  Vr <- tryCatch(vcov(fit, regime = "robust", data = data_arg), error = function(e) NULL)
  if (is.null(Vm) || is.null(Vr)) return(NULL)
  sm <- tryCatch(magmaan::standardized(fit, Vm, type = "all"), error = function(e) NULL)
  sr <- tryCatch(magmaan::standardized(fit, Vr, type = "all"), error = function(e) NULL)
  if (is.null(sm) || is.null(sr)) return(NULL)
  fi <- fx$free
  out <- data.frame(
    raw_est = fit$partable$est[fx$row],
    raw_model = sqrt(Vm[fi, fi]), raw_robust = sqrt(Vr[fi, fi]),
    std_est = sm$theta[fi], std_model = sm$se[fi], std_robust = sr$se[fi])
  if (any(!is.finite(unlist(out)))) return(NULL)
  out
}

run_cell <- function(estimator, cell, cross, seed_off) {
  rows <- lapply(seq_len(cfg$reps), function(r)
    one_rep(estimator, cfg$n_total, cross, cfg$seed_base + seed_off + r))
  D <- do.call(rbind, Filter(Negate(is.null), rows))
  mk <- function(kind, est, sem, ser) data.frame(
    estimator = estimator, cell = cell, param_kind = kind,
    n_total = cfg$n_total, cross = cross, reps = nrow(D),
    emp_sd = sd(est), se_model = mean(sem), se_robust = mean(ser),
    ratio_model = mean(sem) / sd(est), ratio_robust = mean(ser) / sd(est),
    mean_est = mean(est), stringsAsFactors = FALSE)
  rbind(mk("raw", D$raw_est, D$raw_model, D$raw_robust),
        mk("std", D$std_est, D$std_model, D$std_robust))
}

grid <- expand.grid(estimator = c("ML", "DWLS"),
                    cell = c("null", "misspec"), stringsAsFactors = FALSE)
summary <- do.call(rbind, Map(function(est, cell) {
  cross <- if (cell == "null") 0.0 else cfg$cross
  off <- (if (cell == "null") 0L else 1000000L) +
    (if (est == "DWLS") 500000L else 0L)
  run_cell(est, cell, cross, off)
}, grid$estimator, grid$cell))
write_csv(summary, file.path(res_dir, "se_regime.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps, n_total = cfg$n_total, cross = cfg$cross,
    seed_base = cfg$seed_base,
    estimators = "ML (continuous), DWLS (ordinal delta)",
    model = "two_factor_six_indicator; y4 cross-loads on f1 in the DGP, omitted in the fit",
    focal = "f2=~y5 loading (free; on the misspecification-distorted factor) and its std.all counterpart",
    regimes = "model=expected/Gauss-Newton bread; robust=observed-Hessian bread (same empirical meat)",
    smoke = cfg$smoke),
  packages = "magmaan")

print(summary[, c("estimator", "cell", "param_kind", "reps", "emp_sd",
                  "se_model", "se_robust", "ratio_model", "ratio_robust")],
      row.names = FALSE, digits = 4)
cat("\nWrote results to ", res_dir, "\n", sep = "")
