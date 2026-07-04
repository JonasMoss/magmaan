#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--reps N] [--n-grid LIST] [--seed-base N]

DWLS-only taxonomy run for ordinal polychoric-omega profile-LRT calibration.

Options:
  --smoke        Small run: reps=20, n-grid=50,100.
  --reps N       Replications per cell (default 1000).
  --n-grid LIST  Comma-separated sample sizes (default 50,100,250,500).
  --seed-base N  Base seed (default 20260704).
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
reps <- as.integer(take("--reps", if (smoke) "20" else "1000"))
n_grid <- strsplit(take("--n-grid", if (smoke) "50,100" else "50,100,250,500"),
                   ",", fixed = TRUE)[[1L]]
n_grid <- as.integer(n_grid)
seed_base <- as.integer(take("--seed-base", "20260704"))
stopifnot(reps > 0L, all(n_grid > 2L), is.finite(seed_base))

dir.create("results", showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

loadings <- c(0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
ordered_vars <- paste0("x", seq_along(loadings))
block <- rep(1L, length(loadings))
model <- paste("f =~", paste(ordered_vars, collapse = " + "))
spec <- magmaan::model_spec(model, ordered = ordered_vars, parameterization = "delta")
fit_control <- list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
q95 <- stats::qchisq(0.95, df = 1)

cut_regimes <- list(
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
  eta <- stats::rnorm(n)
  X <- matrix(NA_integer_, nrow = n, ncol = length(loadings))
  for (j in seq_along(loadings)) {
    eps <- sqrt(1 - loadings[[j]]^2) * stats::rnorm(n)
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

active_count <- function(fit, side) {
  if (is.null(fit$diagnostics)) return(NA_integer_)
  field <- paste0("active_bounds_", side)
  if (is.null(fit$diagnostics[[field]])) return(NA_integer_)
  length(fit$diagnostics[[field]])
}

failure_row <- function(regime, n, rep_id, arm, estimator, min_cat_count,
                        stage, msg) {
  data.frame(
    regime = regime, n = n, rep = rep_id, arm = arm, estimator = estimator,
    ok = FALSE, stage = stage, message = msg,
    omega_wald = NA_real_, omega_wald_se = NA_real_,
    omega_wald_ci_lo = NA_real_, omega_wald_ci_hi = NA_real_,
    omega_wald_covered = NA,
    target = target, T = NA_real_, reference_stat = NA_real_,
    reference_cutoff = NA_real_, covered = NA,
    scaling_factor = NA_real_, misspec_scaling_factor = NA_real_,
    misspec_mixture_cutoff = NA_real_,
    constrained_value = NA_real_, constraint_residual = NA_real_,
    fmin_unrestricted = NA_real_, fmin_constrained = NA_real_,
    fit_converged = NA, optimizer_status = NA_character_,
    active_lower = NA_integer_, active_upper = NA_integer_,
    min_variable_category_count = min_cat_count,
    stringsAsFactors = FALSE)
}

profile_row <- function(regime, n, rep_id, arm, estimator, fit, stats, omega,
                        min_cat_count, reference = NULL) {
  lrt <- tryCatch(
    core$frontier_profile_lrt_ordinal_polychoric_omega(
      fit, block = block, omega0 = target, weight = "fit",
      ordinal_stats = stats, reference = reference),
    error = identity)
  if (inherits(lrt, "error")) {
    return(failure_row(regime, n, rep_id, arm, estimator, min_cat_count,
                       "profile_lrt", conditionMessage(lrt)))
  }

  if (identical(reference, "robust_scaled")) {
    reference_stat <- lrt$T_scaled
    reference_cutoff <- q95
  } else if (identical(reference, "misspec_mixture")) {
    reference_stat <- lrt$T_misspec_scaled
    reference_cutoff <- q95
  } else {
    reference_stat <- lrt$T
    reference_cutoff <- q95
  }

  lo <- omega$value - 1.96 * omega$se
  hi <- omega$value + 1.96 * omega$se
  data.frame(
    regime = regime, n = n, rep = rep_id, arm = arm, estimator = estimator,
    ok = TRUE, stage = "", message = "",
    omega_wald = omega$value, omega_wald_se = omega$se,
    omega_wald_ci_lo = lo, omega_wald_ci_hi = hi,
    omega_wald_covered = lo <= target && target <= hi,
    target = target, T = lrt$T, reference_stat = reference_stat,
    reference_cutoff = reference_cutoff,
    covered = is.finite(reference_stat) && is.finite(reference_cutoff) &&
      reference_stat <= reference_cutoff,
    scaling_factor = lrt$scaling_factor,
    misspec_scaling_factor = lrt$misspec_scaling_factor,
    misspec_mixture_cutoff = lrt$misspec_mixture_cutoff,
    constrained_value = lrt$constrained_value,
    constraint_residual = lrt$constraint_residual,
    fmin_unrestricted = lrt$fmin_unrestricted,
    fmin_constrained = lrt$fmin_constrained,
    fit_converged = isTRUE(fit$converged),
    optimizer_status = as.character(fit$optimizer_status),
    active_lower = active_count(fit, "lower"),
    active_upper = active_count(fit, "upper"),
    min_variable_category_count = min_cat_count,
    stringsAsFactors = FALSE)
}

fit_one <- function(regime, n, rep_id) {
  regime_id <- match(regime, names(cut_regimes))
  seed <- seed_base + 1000000L * regime_id + 10000L * n + rep_id
  dat <- simulate_ord(n, cut_regimes[[regime]], seed)
  min_cat_count <- min_variable_category_count(dat)

  stats <- tryCatch(core$data_ordinal_stats_from_df(dat, spec), error = identity)
  arms <- data.frame(
    arm = c("dwls_ordinary", "dwls_robust_scaled", "dwls_misspec_mixture"),
    estimator = rep("DWLS", 3L),
    reference = c(NA_character_, "robust_scaled", "misspec_mixture"),
    stringsAsFactors = FALSE)
  if (inherits(stats, "error")) {
    return(do.call(rbind, lapply(seq_len(nrow(arms)), function(i) {
      failure_row(regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]],
                  min_cat_count, "ordinal_stats", conditionMessage(stats))
    })))
  }

  omega <- tryCatch(
    core$measures_reliability_ordinal_polychoric_omega(stats, block = block),
    error = identity)
  if (inherits(omega, "error")) {
    return(do.call(rbind, lapply(seq_len(nrow(arms)), function(i) {
      failure_row(regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]],
                  min_cat_count, "omega_wald", conditionMessage(omega))
    })))
  }

  fit_dwls <- tryCatch(core$fit_dwls_ordinal(spec, stats, control = fit_control),
                       error = identity)

  rows <- vector("list", nrow(arms))
  for (i in seq_len(nrow(arms))) {
    fit <- fit_dwls
    if (inherits(fit, "error")) {
      rows[[i]] <- failure_row(
        regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]], min_cat_count,
        paste0(tolower(arms$estimator[[i]]), "_fit"), conditionMessage(fit))
    } else {
      ref <- arms$reference[[i]]
      if (is.na(ref)) ref <- NULL
      rows[[i]] <- profile_row(
        regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]], fit, stats,
        omega, min_cat_count, reference = ref)
    }
  }
  do.call(rbind, rows)
}

rows <- vector("list", length(cut_regimes) * length(n_grid) * reps)
k <- 1L
for (regime in names(cut_regimes)) {
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

cell_summary <- do.call(rbind, lapply(
  split(raw, list(raw$regime, raw$n, raw$arm), drop = TRUE),
  function(x) {
    ok <- x[x$ok, , drop = FALSE]
    c_oracle <- if (nrow(ok)) mean(ok$reference_stat, na.rm = TRUE) else NA_real_
    data.frame(
      regime = x$regime[[1]],
      n = x$n[[1]],
      arm = x$arm[[1]],
      estimator = x$estimator[[1]],
      reps = length(unique(x$rep)),
      failures = sum(!x$ok),
      target = target,
      mean_omega_wald = if (nrow(ok)) mean(ok$omega_wald) else NA_real_,
      omega_bias = if (nrow(ok)) mean(ok$omega_wald - target) else NA_real_,
      omega_wald_coverage_95 = if (nrow(ok)) mean(ok$omega_wald_covered) else NA_real_,
      mean_T = if (nrow(ok)) mean(ok$T) else NA_real_,
      mean_reference_stat = c_oracle,
      reference_coverage_95 = if (nrow(ok)) mean(ok$covered) else NA_real_,
      oracle_constant_coverage_95 = if (nrow(ok)) {
        mean(ok$reference_stat <= c_oracle * q95, na.rm = TRUE)
      } else NA_real_,
      mean_scaling_factor = if (nrow(ok)) mean(ok$scaling_factor, na.rm = TRUE) else NA_real_,
      mean_misspec_scaling_factor = if (nrow(ok)) {
        mean(ok$misspec_scaling_factor, na.rm = TRUE)
      } else NA_real_,
      min_variable_category_count_p05 = if (nrow(ok)) {
        unname(stats::quantile(ok$min_variable_category_count, 0.05))
      } else NA_real_,
      stringsAsFactors = FALSE)
  }))
cell_summary <- cell_summary[order(cell_summary$regime, cell_summary$n,
                                   cell_summary$arm), ]
row.names(cell_summary) <- NULL

write.csv(raw, "results/simulation_raw.csv", row.names = FALSE)
write.csv(cell_summary, "results/simulation_summary.csv", row.names = FALSE)
write.csv(data.frame(
  smoke = smoke,
  reps = reps,
  n_grid = paste(n_grid, collapse = ","),
  seed_base = seed_base,
  target = target,
  q95 = q95,
  loadings = paste(loadings, collapse = ","),
  cut_regimes = paste(names(cut_regimes), collapse = ","),
  arms = paste(unique(raw$arm), collapse = ","),
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE
), "results/metadata.csv", row.names = FALSE)
message("wrote results/simulation_raw.csv")
message("wrote results/simulation_summary.csv")
message("wrote results/metadata.csv")
