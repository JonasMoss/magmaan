#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--reps N] [--n-grid LIST] [--seed-base N]

Probe robust delta CI coverage for ordinal polychoric omega.

Options:
  --smoke        Small run: reps=20, n-grid=50,100.
  --reps N       Replications per cell (default 200).
  --n-grid LIST  Comma-separated sample sizes (default 50,100,250).
  --seed-base N  Base seed (default 20260702).
  --help         Show this help.
")
}

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  usage()
  quit(status = 0)
}

has_flag <- function(flag) flag %in% args
take <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(flag, " needs a value", call. = FALSE)
  args[[i + 1L]]
}

smoke <- has_flag("--smoke")
reps <- as.integer(take("--reps", if (smoke) "20" else "200"))
n_grid <- strsplit(take("--n-grid", if (smoke) "50,100" else "50,100,250"),
                   ",", fixed = TRUE)[[1L]]
n_grid <- as.integer(n_grid)
seed_base <- as.integer(take("--seed-base", "20260702"))
stopifnot(reps > 0L, all(n_grid > 2L), is.finite(seed_base))

dir.create("results", showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

loadings <- c(0.85, 0.78, 0.70, 0.62)
ordered_vars <- paste0("x", seq_along(loadings))
block <- rep(1L, length(loadings))
model <- paste("f =~", paste(ordered_vars, collapse = " + "))
spec <- magmaan::model_spec(model, ordered = ordered_vars, parameterization = "delta")
regimes <- list(
  balanced = rep(list(c(-0.65, 0.45)), length(loadings)),
  threshold_extreme = rep(list(c(-1.35, 0.95)), length(loadings))
)

latent_R <- tcrossprod(loadings)
diag(latent_R) <- 1
dimnames(latent_R) <- list(ordered_vars, ordered_vars)
target <- core$measures_reliability_omega_multidim(
  latent_R, block = block, target = "total")$value

simulate_ord <- function(n, cuts_by_var, seed) {
  set.seed(seed)
  eta <- rnorm(n)
  X <- matrix(NA_integer_, nrow = n, ncol = length(loadings))
  for (j in seq_along(loadings)) {
    eps <- sqrt(1 - loadings[[j]]^2) * rnorm(n)
    y <- loadings[[j]] * eta + eps
    X[, j] <- as.integer(cut(y, c(-Inf, cuts_by_var[[j]], Inf),
                             labels = FALSE))
  }
  out <- as.data.frame(X)
  names(out) <- ordered_vars
  for (j in seq_along(ordered_vars)) {
    out[[ordered_vars[[j]]]] <- ordered(
      out[[ordered_vars[[j]]]],
      levels = seq_len(length(cuts_by_var[[j]]) + 1L))
  }
  out
}

min_variable_category_count <- function(dat) {
  min(vapply(dat, function(x) {
    min(tabulate(as.integer(x), nbins = nlevels(x)))
  }, integer(1L)))
}

fit_one <- function(regime, n, rep_id) {
  regime_id <- match(regime, names(regimes))
  seed <- seed_base + 1000000L * regime_id + 10000L * n + rep_id
  dat <- simulate_ord(n, regimes[[regime]], seed)
  min_cat_count <- min_variable_category_count(dat)
  stats <- tryCatch(
    core$data_ordinal_stats_from_df(dat, spec),
    error = identity)
  if (inherits(stats, "error")) {
    return(data.frame(
      regime = regime, n = n, rep = rep_id, ok = FALSE,
      value = NA_real_, se = NA_real_, target = target,
      ci_lo = NA_real_, ci_hi = NA_real_, covered = NA,
      width = NA_real_, min_variable_category_count = min_cat_count,
      message = conditionMessage(stats), stringsAsFactors = FALSE))
  }
  omega <- tryCatch(
    core$measures_reliability_ordinal_polychoric_omega(stats, block = block),
    error = identity)
  if (inherits(omega, "error")) {
    return(data.frame(
      regime = regime, n = n, rep = rep_id, ok = FALSE,
      value = NA_real_, se = NA_real_, target = target,
      ci_lo = NA_real_, ci_hi = NA_real_, covered = NA,
      width = NA_real_, min_variable_category_count = min_cat_count,
      message = conditionMessage(omega), stringsAsFactors = FALSE))
  }
  lo <- omega$value - 1.96 * omega$se
  hi <- omega$value + 1.96 * omega$se
  data.frame(
    regime = regime,
    n = n,
    rep = rep_id,
    ok = TRUE,
    value = omega$value,
    se = omega$se,
    target = target,
    ci_lo = lo,
    ci_hi = hi,
    covered = lo <= target && target <= hi,
    width = hi - lo,
    min_variable_category_count = min_cat_count,
    message = "",
    stringsAsFactors = FALSE)
}

rows <- vector("list", length(regimes) * length(n_grid) * reps)
k <- 1L
for (regime in names(regimes)) {
  for (n in n_grid) {
    for (r in seq_len(reps)) {
      if (r %% max(1L, reps %/% 10L) == 0L) {
        message(sprintf("regime=%s n=%d rep=%d/%d", regime, n, r, reps))
      }
      rows[[k]] <- fit_one(regime, n, r)
      k <- k + 1L
    }
  }
}
raw <- do.call(rbind, rows)

cell_summary <- do.call(rbind, lapply(split(raw, list(raw$regime, raw$n), drop = TRUE),
                                      function(x) {
  ok <- x[x$ok, , drop = FALSE]
  data.frame(
    regime = x$regime[[1]],
    n = x$n[[1]],
    reps = nrow(x),
    failures = sum(!x$ok),
    target = target,
    mean_value = if (nrow(ok)) mean(ok$value) else NA_real_,
    bias = if (nrow(ok)) mean(ok$value - target) else NA_real_,
    sd_value = if (nrow(ok)) stats::sd(ok$value) else NA_real_,
    mean_se = if (nrow(ok)) mean(ok$se) else NA_real_,
    coverage_95 = if (nrow(ok)) mean(ok$covered) else NA_real_,
    mean_width_95 = if (nrow(ok)) mean(ok$width) else NA_real_,
    min_variable_category_count_p05 = if (nrow(ok)) {
      unname(stats::quantile(ok$min_variable_category_count, 0.05))
    } else NA_real_,
    stringsAsFactors = FALSE)
}))
cell_summary <- cell_summary[order(cell_summary$regime, cell_summary$n), ]
row.names(cell_summary) <- NULL

write.csv(raw, "results/simulation_raw.csv", row.names = FALSE)
write.csv(cell_summary, "results/simulation_summary.csv", row.names = FALSE)
write.csv(data.frame(
  smoke = smoke,
  reps = reps,
  n_grid = paste(n_grid, collapse = ","),
  seed_base = seed_base,
  target = target,
  loadings = paste(loadings, collapse = ","),
  regimes = paste(names(regimes), collapse = ","),
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE
), "results/metadata.csv", row.names = FALSE)
message("wrote results/simulation_raw.csv")
message("wrote results/simulation_summary.csv")
message("wrote results/metadata.csv")
