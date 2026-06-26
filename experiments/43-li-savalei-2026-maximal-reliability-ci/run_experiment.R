#!/usr/bin/env Rscript
# Finite-sample confidence intervals for maximal reliability in bifactor models:
# the inference Li & Savalei (2026, MBR 61:3, 469-490) leave as future work. They
# give the correct point formulas (OLC for the general factor, OLSC = corrected
# coefficient H for a group factor) and warn the group-factor coefficient is low
# and unstable; we ask whether a CI for it is trustworthy.
#
# For each (coefficient, indicators-per-subscale, N, distribution) cell we fit the
# correct orthogonal bifactor by ML, compute rho* and its delta-method SE under
# the normal-theory ("model") and sandwich ("robust") parameter covariance, and
# score four 95% CIs by coverage of the population rho*:
#   wald_model, wald_robust            -- rho* +/- z*se on the raw scale
#   logit_model, logit_robust          -- the same, Wald on the logit scale
# The two axes attribute any fix to the sandwich V (model vs robust) and to the
# boundary/skew transform (wald vs logit). The bootstrap percentile reference is
# the separate, heavier scripts/bootstrap.R.
#
# Prediction: for the general factor (rho* high) all four cover near 0.95; for the
# group factor (rho* low, near its floor) wald_model under-covers at small N and
# worsens under chi-square data, robust fixes the non-normality part, logit fixes
# the boundary/skew part, and only the two together approach nominal.
#
# Usage:
#   Rscript run_experiment.R [--reps 1000] [--ns 50,100,200,500]
#                            [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(sp)), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("run_experiment.R", mustWork = FALSE)
  dirname(sp)
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))

parse_args <- function(a) {
  o <- list(reps = 1000L, ns = c(50L, 100L, 200L, 500L),
            seed_base = 20260626L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (startsWith(x, "--reps=")) o$reps <- as.integer(sub("^--reps=", "", x))
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (startsWith(x, "--ns=")) o$ns <- as.integer(strsplit(sub("^--ns=", "", x), ",")[[1L]])
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 40L; o$ns <- c(100L) }
  if (!is.finite(o$reps) || o$reps < 2L) stop("--reps must be >= 2")
  if (any(!is.finite(o$ns)) || any(o$ns < 30L)) stop("--ns values must be >= 30")
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
res_dir <- ensure_results_dir()

# Two model sizes: 3 and 6 indicators per subscale, 3 orthogonal subscales. The
# general factor is reliably measured at both; the group factor is not, and its
# coefficient climbs with indicators (Li & Savalei's scaling axis in miniature).
pops <- list(
  p9  = bifactor_population(per = 3L, gen = 0.6, grp = c(0.5, 0.4, 0.45)),
  p18 = bifactor_population(per = 6L, gen = 0.6, grp = c(0.5, 0.4, 0.45)))
specs <- lapply(pops, function(pop)
  magmaan::model_spec(pop$syntax, orthogonal = TRUE, std_lv = TRUE))

dists <- c("normal", "chisq")
methods <- c("wald_model", "wald_robust", "logit_model", "logit_robust")

# Per-cell accumulators: coverage and width per method, plus point bias/SD.
one_cell <- function(pkey, n, dist, seed0) {
  pop <- pops[[pkey]]; spec <- specs[[pkey]]
  tgt <- c(gen = pop$rho_gen, grp = pop$rho_grp1)
  cov <- setNames(rep(0, 0), character(0))
  acc <- list()
  key <- function(coef, m) paste(coef, m, sep = ".")
  for (coef in c("gen", "grp")) for (m in methods) {
    acc[[key(coef, m)]] <- list(cov = 0L, width = 0)
  }
  est <- list(gen = numeric(0), grp = numeric(0))
  ok <- 0L; attempts <- 0L
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    dat <- gen_data(pop, n, dist)
    attempts <- attempts + 1L
    fc <- fit_coefficients(spec, dat, pop)
    if (is.null(fc)) next
    ok <- ok + 1L
    for (ci in seq_len(nrow(fc))) {
      coef <- fc$coef[ci]; rho <- fc$rho[ci]
      est[[coef]] <- c(est[[coef]], rho)
      cis <- list(
        wald_model   = ci_wald(rho, fc$se_model[ci]),
        wald_robust  = ci_wald(rho, fc$se_robust[ci]),
        logit_model  = ci_logit(rho, fc$se_model[ci]),
        logit_robust = ci_logit(rho, fc$se_robust[ci]))
      for (m in methods) {
        a <- acc[[key(coef, m)]]
        a$cov <- a$cov + as.integer(covers(cis[[m]], tgt[[coef]]))
        a$width <- a$width + width(cis[[m]])
        acc[[key(coef, m)]] <- a
      }
    }
  }
  rows <- list()
  for (coef in c("gen", "grp")) for (m in methods) {
    a <- acc[[key(coef, m)]]
    rows[[length(rows) + 1L]] <- data.frame(
      p = pop$p, per = pop$p / pop$k, coef = coef, dist = dist, n = n,
      method = m, reps_ok = ok, attempts = attempts,
      rho_pop = tgt[[coef]],
      mean_est = mean(est[[coef]]), sd_est = stats::sd(est[[coef]]),
      bias = mean(est[[coef]]) - tgt[[coef]],
      coverage = a$cov / ok, mean_width = a$width / ok,
      stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

grid <- expand.grid(pkey = names(pops), n = cfg$ns, dist = dists,
                    stringsAsFactors = FALSE)
seed_for <- function(i) cfg$seed_base + i * 100000L
all_rows <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  message(sprintf("cell %d/%d: %s n=%d %s", i, nrow(grid), g$pkey, g$n, g$dist))
  one_cell(g$pkey, g$n, g$dist, seed_for(i))
}))

write_csv(all_rows, file.path(res_dir, "coverage.csv"))
write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
    seed_base = cfg$seed_base, smoke = cfg$smoke,
    model = "orthogonal std.lv bifactor, 3 subscales, 3 and 6 indicators/subscale",
    loadings = "general 0.6; group 0.5/0.4/0.45; indicator variance 1",
    coefficients = "gen = OLC general-factor max reliability; grp = OLSC group-1 (corrected coefficient H)",
    estimator = "ML (continuous)",
    se = "delta method; V = model (normal-theory) or robust (sandwich) from vcov(fit, regime=)",
    methods = paste(methods, collapse = ","),
    distributions = "normal; chisq (standardized chi-square_1 factors+errors, same Sigma)",
    note = "bootstrap percentile reference in scripts/bootstrap.R"),
  packages = "magmaan")

cat("\nGeneral factor (rho*_pop):\n")
print(all_rows[all_rows$coef == "gen",
               c("p","dist","n","method","coverage","mean_width","bias")],
      row.names = FALSE, digits = 3)
cat("\nGroup factor OLSC (rho*_pop):\n")
print(all_rows[all_rows$coef == "grp",
               c("p","dist","n","method","coverage","mean_width","bias")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "coverage.csv"), "\n", sep = "")
