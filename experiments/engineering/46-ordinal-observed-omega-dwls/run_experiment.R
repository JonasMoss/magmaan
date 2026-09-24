#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--reps N] [--n N] [--n-pop N] [--seed-base N]

Probe observed-category-score omega from an all-ordinal DWLS fit.

Options:
  --smoke        Small run: reps=2, n=240, n-pop=20000.
  --reps N       Replications per threshold regime (default 50).
  --n N          Sample size per replication (default 500).
  --n-pop N      Monte-Carlo population size for the target covariance (default 200000).
  --seed-base N  Base seed (default 20260702).
  --help         Show this help.
")
}

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  usage()
  quit(status = 0)
}

opt <- list(smoke = FALSE, reps = 50L, n = 500L, n_pop = 200000L,
            seed_base = 20260702L)
take <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(flag, " needs a value", call. = FALSE)
  args[[i + 1L]]
}
if ("--smoke" %in% args) {
  opt$smoke <- TRUE
  opt$reps <- 2L
  opt$n <- 240L
  opt$n_pop <- 20000L
}
opt$reps <- as.integer(take("--reps", opt$reps))
opt$n <- as.integer(take("--n", opt$n))
opt$n_pop <- as.integer(take("--n-pop", opt$n_pop))
opt$seed_base <- as.integer(take("--seed-base", opt$seed_base))

stopifnot(opt$reps > 0L, opt$n > 0L, opt$n_pop > 1000L)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

loadings <- c(0.85, 0.78, 0.70, 0.62)
model <- "f =~ x1 + x2 + x3 + x4"
ordered_vars <- paste0("x", seq_along(loadings))
block <- rep(1L, length(loadings))
regimes <- list(
  balanced = c(-0.65, 0.45),
  extreme = c(-1.35, 0.95)
)

simulate_ord <- function(n, cuts, seed) {
  set.seed(seed)
  eta <- rnorm(n)
  X <- matrix(NA_real_, n, length(loadings))
  for (j in seq_along(loadings)) {
    eps <- sqrt(1 - loadings[[j]]^2) * rnorm(n)
    y <- loadings[[j]] * eta + eps
    X[, j] <- as.integer(cut(y, c(-Inf, cuts, Inf), labels = FALSE))
  }
  out <- as.data.frame(X)
  names(out) <- ordered_vars
  for (nm in ordered_vars) out[[nm]] <- ordered(out[[nm]])
  out
}

target_for <- function(cuts, seed) {
  pop <- simulate_ord(opt$n_pop, cuts, seed)
  scores <- as.data.frame(lapply(pop[ordered_vars], function(x) as.integer(x) - 1L))
  S <- stats::cov(scores)
  core$measures_reliability_omega_multidim(S, block = block, target = "total")$value
}

fit_once <- function(regime, cuts, rep_id) {
  seed <- opt$seed_base + match(regime, names(regimes)) * 100000L + rep_id
  dat <- simulate_ord(opt$n, cuts, seed)
  spec <- magmaan::model_spec(model, ordered = ordered_vars, parameterization = "delta")
  stats <- core$data_ordinal_stats_from_df(dat, spec)
  fit <- tryCatch(
    core$fit_dwls_ordinal(
      spec, stats,
      control = list(max_iter = 3000, ftol = 1e-12, gtol = 1e-8)),
    error = identity)
  if (inherits(fit, "error")) {
    return(data.frame(regime = regime, rep = rep_id, converged = FALSE,
                      value = NA_real_, se = NA_real_, target = NA_real_,
                      covered = NA, width = NA_real_, message = fit$message))
  }
  omega <- tryCatch(
    core$measures_reliability_ordinal_observed_omega(fit, block = block),
    error = identity)
  if (inherits(omega, "error")) {
    return(data.frame(regime = regime, rep = rep_id, converged = FALSE,
                      value = NA_real_, se = NA_real_, target = NA_real_,
                      covered = NA, width = NA_real_, message = omega$message))
  }
  target <- targets[[regime]]
  lo <- omega$value - 1.96 * omega$se
  hi <- omega$value + 1.96 * omega$se
  data.frame(regime = regime, rep = rep_id, converged = TRUE,
             value = omega$value, se = omega$se, target = target,
             covered = lo <= target && target <= hi, width = hi - lo,
             message = "")
}

message("computing population targets")
targets <- setNames(
  lapply(names(regimes), function(nm) {
    target_for(regimes[[nm]], opt$seed_base + match(nm, names(regimes)) * 900000L)
  }),
  names(regimes))

rows <- vector("list", length(regimes) * opt$reps)
k <- 1L
for (regime in names(regimes)) {
  for (rep_id in seq_len(opt$reps)) {
    if ((rep_id %% max(1L, opt$reps %/% 10L)) == 0L) {
      message("regime=", regime, " rep=", rep_id, "/", opt$reps)
    }
    rows[[k]] <- fit_once(regime, regimes[[regime]], rep_id)
    k <- k + 1L
  }
}
raw <- do.call(rbind, rows)
ok <- raw[raw$converged, , drop = FALSE]
summary <- do.call(rbind, lapply(split(ok, ok$regime), function(x) {
  data.frame(
    regime = x$regime[[1]],
    reps = nrow(x),
    failures = sum(raw$regime == x$regime[[1]] & !raw$converged),
    target = x$target[[1]],
    mean_value = mean(x$value),
    bias = mean(x$value - x$target),
    mean_se = mean(x$se),
    sd_value = stats::sd(x$value),
    coverage_95 = mean(x$covered),
    mean_width_95 = mean(x$width))
}))

write.csv(raw, "results/simulation_raw.csv", row.names = FALSE)
write.csv(summary, "results/simulation_summary.csv", row.names = FALSE)
write.csv(data.frame(
  smoke = opt$smoke,
  reps = opt$reps,
  n = opt$n,
  n_pop = opt$n_pop,
  seed_base = opt$seed_base,
  magmaan_version = as.character(utils::packageVersion("magmaan"))),
  "results/metadata.csv", row.names = FALSE)
message("wrote results/simulation_raw.csv")
message("wrote results/simulation_summary.csv")
message("wrote results/metadata.csv")
