#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
has_flag <- function(flag) any(args == flag)
arg_value <- function(flag, default) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1L]]
}
parse_grid <- function(x) as.integer(strsplit(x, ",", fixed = TRUE)[[1L]])

smoke <- has_flag("--smoke")
reps <- as.integer(arg_value("--reps", if (smoke) "20" else "200"))
n_grid <- parse_grid(arg_value("--n-grid", if (smoke) "30,50" else "30,50,100,500"))
ci_reps <- as.integer(arg_value("--ci-reps", if (smoke) "4" else "80"))
ci_n_grid <- parse_grid(arg_value("--ci-n-grid", if (smoke) "100" else "100,200,500"))
ci_df <- as.numeric(arg_value("--ci-df", "5"))
seed_base <- as.integer(arg_value("--seed-base", "20260702"))
out_dir <- arg_value("--out-dir", "results")

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

model <- "f =~ x1 + x2 + x3 + x4"
lambda <- c(1.0, 0.8, 0.7, 0.6)
theta <- c(0.36, 0.55, 0.65, 0.75)
Sigma <- tcrossprod(lambda) + diag(theta)
dimnames(Sigma) <- list(paste0("x", 1:4), paste0("x", 1:4))
target_loading <- lambda[[2L]]

pt <- core$lavaan_lavaanify(model)
loading_free_index <- function(fit) {
  row <- with(fit$partable, lhs == "f" & op == "=~" & rhs == "x2")
  idx <- fit$partable$free[row]
  if (length(idx) != 1L || is.na(idx) || idx <= 0L) {
    stop("could not find free loading f =~ x2")
  }
  idx
}

draw_normal_data <- function(n, seed) {
  set.seed(seed)
  Z <- matrix(rnorm(n * nrow(Sigma)), nrow = n)
  X <- Z %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  X
}

draw_t_data <- function(n, seed, df) {
  set.seed(seed)
  Z <- matrix(rnorm(n * nrow(Sigma)), nrow = n)
  scale <- sqrt(df / stats::rchisq(n, df = df)) * sqrt((df - 2) / df)
  X <- (Z * scale) %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  X
}

fit_and_test_lrt <- function(n, rep_id) {
  X <- draw_normal_data(n, seed_base + 100000L * n + rep_id)
  ss <- core$data_sample_stats_from_raw(list(X))
  fit <- core$fit_ml(
    pt, ss,
    optimizer = "nlopt-lbfgs",
    control = list(max_iter = 2000L, ftol = 1e-10, gtol = 1e-7)
  )
  k <- loading_free_index(fit)
  lrt <- core$frontier_profile_lrt_parameter_ml(
    fit, parameter = k, target = target_loading,
    control = list(max_iter = 3000L, ftol = 1e-10, gtol = 1e-7)
  )
  data.frame(
    n = n,
    rep = rep_id,
    parameter = k,
    estimate = fit$theta[[k]],
    target = target_loading,
    T = lrt$T,
    p_value = lrt$p_value,
    reject_05 = lrt$T > stats::qchisq(0.95, df = 1),
    constraint_residual = lrt$constraint_residual,
    abs_constraint_residual = abs(lrt$constraint_residual),
    constrained_converged = isTRUE(lrt$constrained$converged),
    constrained_status = lrt$constrained$optimizer_status,
    stringsAsFactors = FALSE
  )
}

mean_na <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

profile_stat <- function(profile, robust) {
  if (isTRUE(robust)) profile$T_scaled else profile$T
}

profile_scale <- function(profile, robust) {
  if (isTRUE(robust)) profile$scaling_factor else NA_real_
}

ci_methods <- data.frame(
  estimator = c(rep("ULS", 4L), rep("GLS", 3L), rep("WLS", 3L)),
  profile_weight = c(
    "fixed", "fixed", "fitted", "fitted",
    "fixed", "fixed", "fixed",
    "fixed", "fixed", "fixed"
  ),
  reference = c(
    "ordinary", "robust_fixed", "ordinary", "robust_fixed",
    "ordinary", "robust_fixed", "robust_estimated_weight",
    "ordinary", "robust_fixed", "robust_estimated_weight"
  ),
  stringsAsFactors = FALSE
)

is_robust_reference <- function(reference) !identical(reference, "ordinary")
is_estimated_weight_reference <- function(reference) {
  identical(reference, "robust_estimated_weight")
}

ci_failure_rows <- function(n, rep_id, stage, error, estimator = NULL) {
  methods <- if (is.null(estimator)) {
    ci_methods
  } else {
    ci_methods[ci_methods$estimator == estimator, , drop = FALSE]
  }
  data.frame(
    n = n,
    rep = rep_id,
    estimator = methods$estimator,
    profile_weight = methods$profile_weight,
    reference = methods$reference,
    stage = stage,
    error = error,
    stringsAsFactors = FALSE
  )
}

ci_to_row <- function(ci, n, rep_id, estimator, profile_weight, reference) {
  robust <- is_robust_reference(reference)
  lower_stat <- profile_stat(ci$lower_profile, robust)
  upper_stat <- profile_stat(ci$upper_profile, robust)
  data.frame(
    n = n,
    rep = rep_id,
    generator = sprintf("t%g", ci_df),
    estimator = estimator,
    profile_weight = profile_weight,
    reference = reference,
    parameter = ci$parameter,
    estimate = ci$estimate,
    target = target_loading,
    lower = ci$lower,
    upper = ci$upper,
    width = ci$upper - ci$lower,
    covers = ci$lower <= target_loading && target_loading <= ci$upper,
    cutoff = ci$cutoff,
    lower_stat = lower_stat,
    upper_stat = upper_stat,
    endpoint_error = max(abs(c(lower_stat, upper_stat) - ci$cutoff), na.rm = TRUE),
    lower_scaling_factor = profile_scale(ci$lower_profile, robust),
    upper_scaling_factor = profile_scale(ci$upper_profile, robust),
    lower_evals = ci$lower_evals,
    upper_evals = ci$upper_evals,
    lower_at_bound = isTRUE(ci$lower_at_bound),
    upper_at_bound = isTRUE(ci$upper_at_bound),
    lower_converged = isTRUE(ci$lower_profile$constrained$converged),
    upper_converged = isTRUE(ci$upper_profile$constrained$converged),
    stringsAsFactors = FALSE
  )
}

fit_ci_model <- function(estimator, dat, X) {
  if (identical(estimator, "WLS")) {
    W <- tryCatch(solve(core$robust_empirical_gamma(X)), error = function(e) e)
    if (inherits(W, "error")) return(list(error = conditionMessage(W)))
    fit <- tryCatch(
      magmaan::magmaan(model, dat, estimator = "WLS", W = W),
      error = function(e) e
    )
    if (inherits(fit, "error")) return(list(error = conditionMessage(fit)))
    return(list(fit = fit, weight = W))
  }
  fit <- tryCatch(
    magmaan::magmaan(model, dat, estimator = estimator),
    error = function(e) e
  )
  if (inherits(fit, "error")) return(list(error = conditionMessage(fit)))
  list(fit = fit, weight = NULL)
}

run_fixed_ci <- function(fit, weight, k, step, X, reference, ci_control) {
  robust <- is_robust_reference(reference)
  args <- list(
    fit = fit,
    parameter = k,
    initial_step = step,
    control = ci_control,
    root_tol = if (robust) 1e-5 else 1e-4,
    statistic_tol = if (robust) 1e-5 else 1e-4
  )
  if (!is.null(weight)) args$weight <- weight
  if (robust) {
    args$raw_data <- X
    args$robust <- TRUE
  }
  if (is_estimated_weight_reference(reference)) {
    args$estimated_weight <- TRUE
  }
  do.call(core$frontier_profile_lrt_ci_parameter_gmm, args)
}

run_fitted_ci <- function(fit, k, step, X, reference, ci_control) {
  robust <- is_robust_reference(reference)
  args <- list(
    fit = fit,
    parameter = k,
    initial_step = step,
    control = ci_control,
    root_tol = if (robust) 1e-5 else 1e-4,
    statistic_tol = if (robust) 1e-5 else 1e-4
  )
  if (robust) {
    args$raw_data <- X
    args$robust <- TRUE
  }
  do.call(core$frontier_profile_lrt_ci_parameter_gmm_fitted_weight, args)
}

fit_and_test_ci <- function(n, rep_id) {
  X <- draw_t_data(n, seed_base + 200000L * n + rep_id, df = ci_df)
  dat <- as.data.frame(X)
  ci_control <- list(max_iter = 10000L, ftol = 1e-10, gtol = 1e-7)
  raw_rows <- list()
  fail_rows <- list()

  for (estimator in unique(ci_methods$estimator)) {
    fit_info <- fit_ci_model(estimator, dat, X)
    if (!is.null(fit_info$error)) {
      fail_rows[[length(fail_rows) + 1L]] <- ci_failure_rows(
        n, rep_id, "fit", fit_info$error, estimator = estimator
      )
      next
    }

    fit <- fit_info$fit
    k <- loading_free_index(fit)
    step <- max(0.02, 0.1 * abs(as.numeric(fit$theta[[k]])))
    methods <- ci_methods[ci_methods$estimator == estimator, , drop = FALSE]
    for (i in seq_len(nrow(methods))) {
      method <- methods[i, ]
      out <- tryCatch({
        if (identical(method$profile_weight, "fitted")) {
          run_fitted_ci(fit, k, step, X, method$reference, ci_control)
        } else {
          run_fixed_ci(fit, fit_info$weight, k, step, X,
                       method$reference, ci_control)
        }
      }, error = function(e) e)
      if (inherits(out, "error")) {
        fail_rows[[length(fail_rows) + 1L]] <- data.frame(
          n = n,
          rep = rep_id,
          estimator = estimator,
          profile_weight = method$profile_weight,
          reference = method$reference,
          stage = "ci",
          error = conditionMessage(out),
          stringsAsFactors = FALSE
        )
      } else {
        raw_rows[[length(raw_rows) + 1L]] <- ci_to_row(
          out, n, rep_id, estimator, method$profile_weight, method$reference
        )
      }
    }
  }

  list(
    raw = if (length(raw_rows)) do.call(rbind, raw_rows) else data.frame(),
    fail = if (length(fail_rows)) {
      do.call(rbind, fail_rows)
    } else {
      data.frame(
        n = integer(), rep = integer(), estimator = character(),
        profile_weight = character(), reference = character(),
        stage = character(), error = character()
      )
    }
  )
}

summarize_lrt <- function(raw, fail) {
  summary <- if (nrow(raw)) {
    aggregate(
      cbind(T, reject_05, estimate, abs_constraint_residual,
            constrained_converged = as.numeric(raw$constrained_converged)) ~ n,
      data = raw,
      FUN = function(x) c(mean = mean(x), sd = stats::sd(x))
    )
  } else {
    data.frame()
  }
  if (nrow(summary)) {
    summary <- data.frame(
      n = summary$n,
      reps = as.integer(tabulate(match(raw$n, summary$n), nbins = nrow(summary))),
      failures = as.integer(tabulate(match(fail$n, summary$n), nbins = nrow(summary))),
      mean_T = summary$T[, "mean"],
      sd_T = summary$T[, "sd"],
      type1_05 = summary$reject_05[, "mean"],
      mean_estimate = summary$estimate[, "mean"],
      sd_estimate = summary$estimate[, "sd"],
      mean_abs_constraint_residual = summary$abs_constraint_residual[, "mean"],
      constrained_convergence_rate = summary$constrained_converged[, "mean"]
    )
  }
  summary
}

summarize_ci <- function(raw, fail) {
  if (!nrow(raw)) return(data.frame())
  groups <- unique(raw[c("n", "estimator", "profile_weight", "reference")])
  rows <- vector("list", nrow(groups))
  for (i in seq_len(nrow(groups))) {
    g <- groups[i, ]
    keep <- raw$n == g$n &
      raw$estimator == g$estimator &
      raw$profile_weight == g$profile_weight &
      raw$reference == g$reference
    sub <- raw[keep, ]
    rows[[i]] <- data.frame(
      n = g$n,
      estimator = g$estimator,
      profile_weight = g$profile_weight,
      reference = g$reference,
      reps = nrow(sub),
      failures = sum(
        fail$n == g$n &
          fail$estimator == g$estimator &
          fail$profile_weight == g$profile_weight &
          fail$reference == g$reference
      ),
      coverage = mean(sub$covers),
      mean_width = mean(sub$width),
      sd_width = stats::sd(sub$width),
      mean_endpoint_error = mean(sub$endpoint_error),
      mean_lower_scaling_factor = mean_na(sub$lower_scaling_factor),
      mean_upper_scaling_factor = mean_na(sub$upper_scaling_factor),
      lower_bound_rate = mean(sub$lower_at_bound),
      upper_bound_rate = mean(sub$upper_at_bound),
      lower_convergence_rate = mean(sub$lower_converged),
      upper_convergence_rate = mean(sub$upper_converged),
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)
  out[order(out$n, out$estimator, out$profile_weight, out$reference), ]
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

lrt_raw_rows <- list()
lrt_fail_rows <- list()
for (n in n_grid) {
  for (r in seq_len(reps)) {
    if (r %% max(1L, reps %/% 10L) == 0L) {
      message(sprintf("LRT n=%d rep=%d/%d", n, r, reps))
    }
    out <- tryCatch(
      fit_and_test_lrt(n, r),
      error = function(e) {
        data.frame(n = n, rep = r, error = conditionMessage(e),
                   stringsAsFactors = FALSE)
      }
    )
    if ("error" %in% names(out)) {
      lrt_fail_rows[[length(lrt_fail_rows) + 1L]] <- out
    } else {
      lrt_raw_rows[[length(lrt_raw_rows) + 1L]] <- out
    }
  }
}

ci_raw_rows <- list()
ci_fail_rows <- list()
for (n in ci_n_grid) {
  for (r in seq_len(ci_reps)) {
    if (r %% max(1L, ci_reps %/% 10L) == 0L) {
      message(sprintf("CI n=%d rep=%d/%d", n, r, ci_reps))
    }
    out <- fit_and_test_ci(n, r)
    if (nrow(out$raw)) ci_raw_rows[[length(ci_raw_rows) + 1L]] <- out$raw
    if (nrow(out$fail)) ci_fail_rows[[length(ci_fail_rows) + 1L]] <- out$fail
  }
}

lrt_raw <- if (length(lrt_raw_rows)) do.call(rbind, lrt_raw_rows) else data.frame()
lrt_fail <- if (length(lrt_fail_rows)) {
  do.call(rbind, lrt_fail_rows)
} else {
  data.frame(n = integer(), rep = integer(), error = character())
}
lrt_summary <- summarize_lrt(lrt_raw, lrt_fail)

ci_raw <- if (length(ci_raw_rows)) do.call(rbind, ci_raw_rows) else data.frame()
ci_fail <- if (length(ci_fail_rows)) {
  do.call(rbind, ci_fail_rows)
} else {
  data.frame(
    n = integer(), rep = integer(), estimator = character(),
    profile_weight = character(), reference = character(),
    stage = character(), error = character()
  )
}
ci_summary <- summarize_ci(ci_raw, ci_fail)

meta <- data.frame(
  smoke = smoke,
  reps = reps,
  n_grid = paste(n_grid, collapse = ","),
  ci_reps = ci_reps,
  ci_n_grid = paste(ci_n_grid, collapse = ","),
  ci_generator = "multivariate_t",
  ci_df = ci_df,
  ci_estimators = paste(unique(ci_methods$estimator), collapse = ","),
  ci_references = paste(unique(ci_methods$reference), collapse = ","),
  seed_base = seed_base,
  target_loading = target_loading,
  stringsAsFactors = FALSE
)

write.csv(lrt_raw, file.path(out_dir, "profile_lrt_raw.csv"), row.names = FALSE)
write.csv(lrt_summary, file.path(out_dir, "profile_lrt_summary.csv"), row.names = FALSE)
write.csv(lrt_fail, file.path(out_dir, "profile_lrt_failures.csv"), row.names = FALSE)
write.csv(ci_raw, file.path(out_dir, "profile_ci_raw.csv"), row.names = FALSE)
write.csv(ci_summary, file.path(out_dir, "profile_ci_summary.csv"), row.names = FALSE)
write.csv(ci_fail, file.path(out_dir, "profile_ci_failures.csv"), row.names = FALSE)
write.csv(meta, file.path(out_dir, "metadata.csv"), row.names = FALSE)
message(sprintf("wrote %s/profile_lrt_raw.csv", out_dir))
message(sprintf("wrote %s/profile_lrt_summary.csv", out_dir))
message(sprintf("wrote %s/profile_lrt_failures.csv", out_dir))
message(sprintf("wrote %s/profile_ci_raw.csv", out_dir))
message(sprintf("wrote %s/profile_ci_summary.csv", out_dir))
message(sprintf("wrote %s/profile_ci_failures.csv", out_dir))
message(sprintf("wrote %s/metadata.csv", out_dir))
