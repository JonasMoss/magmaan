#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--ci] [--reps N] [--n-grid LIST] [--dgp LIST] [--pseudo-n N] [--ci-references LIST] [--seed-base N]

DWLS-only taxonomy run for ordinal polychoric-omega profile-LRT calibration.

Options:
  --smoke       Small run: reps=20, n-grid=50,100, pseudo-n=10000.
  --reps N      Replications per cell (default 1000).
  --n-grid LIST Comma-separated sample sizes (default 50,100,250,500).
  --dgp LIST    Comma-separated DGPs: one_factor,local_dependence (default both).
  --pseudo-n N  Large-sample size for pseudo-targets (default 100000).
  --ci          Also compute profile-LRT CI inversion for selected references.
  --ci-references LIST
               CI references (default robust_scaled,misspec_mixture).
  --seed-base N Base seed (default 20260704).
  --help        Show this help.
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

split_list <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

normalize_reference_key <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("-", "_", x, fixed = TRUE)
  if (x %in% c("chisq", "chi_square")) return("ordinary")
  if (x %in% c("satorra_scaled", "scaled")) return("robust_scaled")
  if (x == "misspecification_scaled") return("misspec_scaled")
  if (x %in% c("misspecification_mixture", "mixture")) return("misspec_mixture")
  x
}

smoke <- has_flag("--smoke")
ci_enabled <- has_flag("--ci")
reps <- as.integer(take("--reps", if (smoke) "20" else "1000"))
n_grid <- split_list(take("--n-grid", if (smoke) "50,100" else "50,100,250,500"))
n_grid <- as.integer(n_grid)
pseudo_n <- as.integer(take("--pseudo-n", if (smoke) "10000" else "100000"))
seed_base <- as.integer(take("--seed-base", "20260704"))
all_dgps <- c("one_factor", "local_dependence")
dgp_grid <- split_list(take("--dgp", paste(all_dgps, collapse = ",")))
reference_levels <- c("ordinary", "robust_scaled", "misspec_scaled",
                      "misspec_mixture")
ci_references <- vapply(
  split_list(take("--ci-references", "robust_scaled,misspec_mixture")),
  normalize_reference_key, character(1L), USE.NAMES = FALSE)
stopifnot(reps > 0L, all(n_grid > 2L), pseudo_n > 1000L,
          is.finite(seed_base), all(dgp_grid %in% all_dgps),
          all(ci_references %in% reference_levels))

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
ci_initial_step <- 0.02
ci_root_tol <- 1e-5
ci_statistic_tol <- 1e-5

cut_regimes <- list(
  balanced = rep(list(c(-0.65, 0.45)), length(loadings)),
  threshold_extreme = rep(list(c(-1.35, 0.95)), length(loadings))
)

latent_response <- function(n, dgp, seed) {
  set.seed(seed)
  eta <- stats::rnorm(n)
  theta_sqrt <- sqrt(1 - loadings^2)
  residual_R <- diag(length(loadings))

  if (dgp == "local_dependence") {
    residual_R[1L, 2L] <- residual_R[2L, 1L] <- 0.35
    residual_R[5L, 6L] <- residual_R[6L, 5L] <- 0.25
  } else if (dgp != "one_factor") {
    stop("unknown dgp: ", dgp, call. = FALSE)
  }

  eps <- matrix(stats::rnorm(n * length(loadings)), nrow = n) %*%
    chol(residual_R)
  sweep(eps, 2L, theta_sqrt, `*`) + tcrossprod(eta, loadings)
}

discretize_ord <- function(Y, cuts_by_var) {
  X <- matrix(NA_integer_, nrow = nrow(Y), ncol = ncol(Y))
  for (j in seq_along(cuts_by_var)) {
    X[, j] <- as.integer(cut(Y[, j], c(-Inf, cuts_by_var[[j]], Inf),
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

simulate_ord <- function(n, dgp, cuts_by_var, seed) {
  discretize_ord(latent_response(n, dgp, seed), cuts_by_var)
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

arm_name <- function(reference_key) {
  switch(reference_key,
         ordinary = "dwls_ordinary",
         robust_scaled = "dwls_robust_scaled",
         misspec_scaled = "dwls_misspec_scaled",
         misspec_mixture = "dwls_misspec_mixture",
         stop("unknown reference key: ", reference_key, call. = FALSE))
}

reference_arg <- function(reference_key) {
  if (identical(reference_key, "ordinary")) NA_character_ else reference_key
}

make_arms <- function(reference_keys) {
  reference_keys <- unique(reference_keys)
  data.frame(
    arm = vapply(reference_keys, arm_name, character(1L), USE.NAMES = FALSE),
    estimator = rep("DWLS", length(reference_keys)),
    reference_key = reference_keys,
    reference = vapply(reference_keys, reference_arg, character(1L),
                       USE.NAMES = FALSE),
    stringsAsFactors = FALSE)
}

base_references <- c("ordinary", "robust_scaled", "misspec_mixture")
arms <- make_arms(c(base_references, if (ci_enabled) ci_references))

profile_omega_unrestricted <- function(fit, stats, omega0) {
  lrt <- core$frontier_profile_lrt_ordinal_polychoric_omega(
    fit, block = block, omega0 = omega0, weight = "fit",
    ordinal_stats = stats)
  lrt$unrestricted_value
}

estimate_pseudo_target <- function(dgp, regime) {
  dgp_id <- match(dgp, all_dgps)
  regime_id <- match(regime, names(cut_regimes))
  seed <- seed_base + 70000000L + 1000000L * dgp_id + 100000L * regime_id
  dat <- simulate_ord(pseudo_n, dgp, cut_regimes[[regime]], seed)
  stats <- core$data_ordinal_stats_from_df(dat, spec)
  omega_sample <- core$measures_reliability_ordinal_polychoric_omega(
    stats, block = block)
  fit <- core$fit_dwls_ordinal(spec, stats, control = fit_control)
  target <- profile_omega_unrestricted(fit, stats, omega_sample$value)
  data.frame(
    dgp = dgp,
    regime = regime,
    pseudo_n = pseudo_n,
    target = target,
    sample_polychoric_target = omega_sample$value,
    target_minus_sample_polychoric = target - omega_sample$value,
    stringsAsFactors = FALSE)
}

message("estimating pseudo-targets")
targets <- do.call(rbind, lapply(dgp_grid, function(dgp) {
  do.call(rbind, lapply(names(cut_regimes), function(regime) {
    message(sprintf("target dgp=%s regime=%s", dgp, regime))
    estimate_pseudo_target(dgp, regime)
  }))
}))

target_lookup <- function(dgp, regime) {
  targets[targets$dgp == dgp & targets$regime == regime, , drop = FALSE]
}

ci_blank <- function(attempted = FALSE, ok = NA, stage = "", msg = "") {
  data.frame(
    ci_attempted = attempted,
    ci_ok = ok,
    ci_stage = stage,
    ci_message = msg,
    ci_estimate = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    ci_width = NA_real_,
    ci_covered = NA,
    ci_lower_evals = NA_integer_,
    ci_upper_evals = NA_integer_,
    ci_lower_at_bound = NA,
    ci_upper_at_bound = NA,
    ci_lower_stat = NA_real_,
    ci_upper_stat = NA_real_,
    ci_lower_cutoff = NA_real_,
    ci_upper_cutoff = NA_real_,
    ci_lower_gap = NA_real_,
    ci_upper_gap = NA_real_,
    ci_lrt_agrees = NA,
    stringsAsFactors = FALSE)
}

ci_endpoint_stat <- function(profile, reference_key) {
  if (identical(reference_key, "robust_scaled")) return(profile$T_scaled)
  if (identical(reference_key, "misspec_scaled")) return(profile$T_misspec_scaled)
  profile$T
}

run_ci <- function(fit, stats, target, covered, reference_key) {
  if (!ci_enabled || !(reference_key %in% ci_references)) return(ci_blank())
  ref <- reference_arg(reference_key)
  if (is.na(ref)) ref <- NULL
  ci <- tryCatch(
    core$frontier_profile_lrt_ci_ordinal_polychoric_omega(
      fit, block = block, weight = "fit", ordinal_stats = stats,
      initial_step = ci_initial_step, root_tol = ci_root_tol,
      statistic_tol = ci_statistic_tol, reference = ref),
    error = identity)
  if (inherits(ci, "error")) {
    return(ci_blank(TRUE, FALSE, "profile_ci", conditionMessage(ci)))
  }

  lower_stat <- ci_endpoint_stat(ci$lower_profile, reference_key)
  upper_stat <- ci_endpoint_stat(ci$upper_profile, reference_key)
  lower_cutoff <- if (identical(reference_key, "misspec_mixture")) {
    ci$lower_cutoff
  } else {
    ci$cutoff
  }
  upper_cutoff <- if (identical(reference_key, "misspec_mixture")) {
    ci$upper_cutoff
  } else {
    ci$cutoff
  }
  ci_covered <- is.finite(ci$lower) && is.finite(ci$upper) &&
    ci$lower <= target && target <= ci$upper
  data.frame(
    ci_attempted = TRUE,
    ci_ok = TRUE,
    ci_stage = "",
    ci_message = "",
    ci_estimate = ci$estimate,
    ci_lower = ci$lower,
    ci_upper = ci$upper,
    ci_width = ci$upper - ci$lower,
    ci_covered = ci_covered,
    ci_lower_evals = ci$lower_evals,
    ci_upper_evals = ci$upper_evals,
    ci_lower_at_bound = ci$lower_at_bound,
    ci_upper_at_bound = ci$upper_at_bound,
    ci_lower_stat = lower_stat,
    ci_upper_stat = upper_stat,
    ci_lower_cutoff = lower_cutoff,
    ci_upper_cutoff = upper_cutoff,
    ci_lower_gap = lower_stat - lower_cutoff,
    ci_upper_gap = upper_stat - upper_cutoff,
    ci_lrt_agrees = if (is.na(covered)) NA else identical(ci_covered, covered),
    stringsAsFactors = FALSE)
}

failure_row <- function(dgp, regime, n, rep_id, arm, estimator, reference_key,
                        target_row, min_cat_count, stage, msg) {
  data.frame(
    dgp = dgp, regime = regime, n = n, rep = rep_id, arm = arm,
    estimator = estimator, reference = reference_key,
    ok = FALSE, stage = stage, message = msg,
    omega_wald = NA_real_, omega_wald_se = NA_real_,
    omega_wald_ci_lo = NA_real_, omega_wald_ci_hi = NA_real_,
    omega_wald_covered = NA,
    target = target_row$target,
    sample_polychoric_target = target_row$sample_polychoric_target,
    target_minus_sample_polychoric = target_row$target_minus_sample_polychoric,
    T = NA_real_, reference_stat = NA_real_,
    reference_cutoff = NA_real_, covered = NA,
    scaling_factor = NA_real_, misspec_scaling_factor = NA_real_,
    misspec_mixture_cutoff = NA_real_,
    constrained_value = NA_real_, constraint_residual = NA_real_,
    fmin_unrestricted = NA_real_, fmin_constrained = NA_real_,
    fit_converged = NA, optimizer_status = NA_character_,
    active_lower = NA_integer_, active_upper = NA_integer_,
    min_variable_category_count = min_cat_count,
    stringsAsFactors = FALSE) |>
    cbind(ci_blank())
}

profile_row <- function(dgp, regime, n, rep_id, arm, estimator, fit, stats,
                        omega, target_row, min_cat_count, reference_key,
                        reference = NULL) {
  target <- target_row$target
  lrt <- tryCatch(
    core$frontier_profile_lrt_ordinal_polychoric_omega(
      fit, block = block, omega0 = target, weight = "fit",
      ordinal_stats = stats, reference = reference),
    error = identity)
  if (inherits(lrt, "error")) {
    return(failure_row(dgp, regime, n, rep_id, arm, estimator, reference_key,
                       target_row, min_cat_count,
                       "profile_lrt", conditionMessage(lrt)))
  }

  if (identical(reference, "robust_scaled")) {
    reference_stat <- lrt$T_scaled
    reference_cutoff <- q95
  } else if (identical(reference, "misspec_scaled")) {
    reference_stat <- lrt$T_misspec_scaled
    reference_cutoff <- q95
  } else if (identical(reference, "misspec_mixture")) {
    reference_stat <- lrt$T_misspec_scaled
    reference_cutoff <- q95
  } else {
    reference_stat <- lrt$T
    reference_cutoff <- q95
  }
  covered <- is.finite(reference_stat) && is.finite(reference_cutoff) &&
    reference_stat <= reference_cutoff

  lo <- omega$value - 1.96 * omega$se
  hi <- omega$value + 1.96 * omega$se
  row <- data.frame(
    dgp = dgp, regime = regime, n = n, rep = rep_id, arm = arm,
    estimator = estimator, reference = reference_key,
    ok = TRUE, stage = "", message = "",
    omega_wald = omega$value, omega_wald_se = omega$se,
    omega_wald_ci_lo = lo, omega_wald_ci_hi = hi,
    omega_wald_covered = lo <= target && target <= hi,
    target = target,
    sample_polychoric_target = target_row$sample_polychoric_target,
    target_minus_sample_polychoric = target_row$target_minus_sample_polychoric,
    T = lrt$T, reference_stat = reference_stat,
    reference_cutoff = reference_cutoff,
    covered = covered,
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
  cbind(row, run_ci(fit, stats, target, covered, reference_key))
}

fit_one <- function(dgp, regime, n, rep_id) {
  dgp_id <- match(dgp, all_dgps)
  regime_id <- match(regime, names(cut_regimes))
  seed <- seed_base + 10000000L * dgp_id + 1000000L * regime_id +
    10000L * n + rep_id
  target_row <- target_lookup(dgp, regime)
  dat <- simulate_ord(n, dgp, cut_regimes[[regime]], seed)
  min_cat_count <- min_variable_category_count(dat)

  stats <- tryCatch(core$data_ordinal_stats_from_df(dat, spec), error = identity)
  if (inherits(stats, "error")) {
    return(do.call(rbind, lapply(seq_len(nrow(arms)), function(i) {
      failure_row(dgp, regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]],
                  arms$reference_key[[i]], target_row,
                  min_cat_count, "ordinal_stats", conditionMessage(stats))
    })))
  }

  omega <- tryCatch(
    core$measures_reliability_ordinal_polychoric_omega(stats, block = block),
    error = identity)
  if (inherits(omega, "error")) {
    return(do.call(rbind, lapply(seq_len(nrow(arms)), function(i) {
      failure_row(dgp, regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]],
                  arms$reference_key[[i]], target_row,
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
        dgp, regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]],
        arms$reference_key[[i]], target_row, min_cat_count,
        paste0(tolower(arms$estimator[[i]]), "_fit"), conditionMessage(fit))
    } else {
      ref <- arms$reference[[i]]
      if (is.na(ref)) ref <- NULL
      rows[[i]] <- profile_row(
        dgp, regime, n, rep_id, arms$arm[[i]], arms$estimator[[i]], fit,
        stats, omega, target_row, min_cat_count, arms$reference_key[[i]],
        reference = ref)
    }
  }
  do.call(rbind, rows)
}

rows <- vector("list", length(dgp_grid) * length(cut_regimes) *
                 length(n_grid) * reps)
k <- 1L
for (dgp in dgp_grid) {
  for (regime in names(cut_regimes)) {
    for (n in n_grid) {
      for (r in seq_len(reps)) {
        if (r %% max(1L, reps %/% 10L) == 0L) {
          message(sprintf("dgp=%s regime=%s n=%d rep=%d/%d",
                          dgp, regime, n, r, reps))
        }
        rows[[k]] <- fit_one(dgp, regime, n, r)
        k <- k + 1L
      }
    }
  }
}
raw <- do.call(rbind, rows)

cell_summary <- do.call(rbind, lapply(
  split(raw, list(raw$dgp, raw$regime, raw$n, raw$arm), drop = TRUE),
  function(x) {
    ok <- x[x$ok, , drop = FALSE]
    c_oracle <- if (nrow(ok)) mean(ok$reference_stat, na.rm = TRUE) else NA_real_
    ci_success <- x$ci_attempted & !is.na(x$ci_ok) & x$ci_ok
    ci_failed <- x$ci_attempted & (is.na(x$ci_ok) | !x$ci_ok)
    ci_ok <- x[ci_success, , drop = FALSE]
    ci_eval_count <- ci_ok$ci_lower_evals + ci_ok$ci_upper_evals
    ci_endpoint_gap <- c(ci_ok$ci_lower_gap, ci_ok$ci_upper_gap)
    finite_ci_endpoint_gap <- ci_endpoint_gap[is.finite(ci_endpoint_gap)]
    data.frame(
      dgp = x$dgp[[1]],
      regime = x$regime[[1]],
      n = x$n[[1]],
      arm = x$arm[[1]],
      estimator = x$estimator[[1]],
      reference = x$reference[[1]],
      reps = length(unique(x$rep)),
      failures = sum(!x$ok),
      target = x$target[[1]],
      sample_polychoric_target = x$sample_polychoric_target[[1]],
      target_minus_sample_polychoric = x$target_minus_sample_polychoric[[1]],
      mean_omega_wald = if (nrow(ok)) mean(ok$omega_wald) else NA_real_,
      omega_bias = if (nrow(ok)) mean(ok$omega_wald - ok$target) else NA_real_,
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
      ci_attempted = sum(x$ci_attempted),
      ci_failures = sum(ci_failed),
      ci_coverage_95 = if (nrow(ci_ok)) mean(ci_ok$ci_covered) else NA_real_,
      mean_ci_width = if (nrow(ci_ok)) mean(ci_ok$ci_width) else NA_real_,
      median_ci_width = if (nrow(ci_ok)) stats::median(ci_ok$ci_width) else NA_real_,
      mean_ci_evals = if (nrow(ci_ok)) mean(ci_eval_count) else NA_real_,
      ci_bound_rate = if (nrow(ci_ok)) {
        mean(ci_ok$ci_lower_at_bound | ci_ok$ci_upper_at_bound)
      } else NA_real_,
      max_abs_ci_endpoint_gap = if (length(finite_ci_endpoint_gap)) {
        max(abs(finite_ci_endpoint_gap))
      } else NA_real_,
      ci_lrt_disagreements = if (nrow(ci_ok)) {
        sum(!is.na(ci_ok$ci_lrt_agrees) & !ci_ok$ci_lrt_agrees)
      } else NA_integer_,
      ci_lrt_disagreement_rate = if (nrow(ci_ok)) {
        mean(!is.na(ci_ok$ci_lrt_agrees) & !ci_ok$ci_lrt_agrees)
      } else NA_real_,
      stringsAsFactors = FALSE)
  }))
cell_summary <- cell_summary[order(cell_summary$dgp, cell_summary$regime,
                                   cell_summary$n, cell_summary$arm), ]
row.names(cell_summary) <- NULL

write.csv(raw, "results/simulation_raw.csv", row.names = FALSE)
write.csv(targets, "results/pseudo_targets.csv", row.names = FALSE)
write.csv(cell_summary, "results/simulation_summary.csv", row.names = FALSE)
write.csv(data.frame(
  smoke = smoke,
  reps = reps,
  n_grid = paste(n_grid, collapse = ","),
  pseudo_n = pseudo_n,
  seed_base = seed_base,
  q95 = q95,
  ci = ci_enabled,
  ci_references = if (ci_enabled) paste(ci_references, collapse = ",") else "",
  ci_initial_step = ci_initial_step,
  ci_root_tol = ci_root_tol,
  ci_statistic_tol = ci_statistic_tol,
  loadings = paste(loadings, collapse = ","),
  dgps = paste(dgp_grid, collapse = ","),
  cut_regimes = paste(names(cut_regimes), collapse = ","),
  arms = paste(unique(raw$arm), collapse = ","),
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE
), "results/metadata.csv", row.names = FALSE)
message("wrote results/pseudo_targets.csv")
message("wrote results/simulation_raw.csv")
message("wrote results/simulation_summary.csv")
message("wrote results/metadata.csv")
