#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--reps N] [--n-grid LIST] [--pseudo-n N] [--seed-base N]

Stress robust delta CI coverage for ordinal polychoric omega.

Options:
  --smoke       Small run: reps=20, n-grid=50,100, pseudo-n=10000.
  --reps N      Replications per cell (default 200).
  --n-grid LIST Comma-separated sample sizes (default 50,100,250).
  --pseudo-n N  Large-sample size for pseudo-targets (default 100000).
  --seed-base N Base seed (default 20260703).
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

smoke <- has_flag("--smoke")
reps <- as.integer(take("--reps", if (smoke) "20" else "200"))
n_grid <- strsplit(take("--n-grid", if (smoke) "50,100" else "50,100,250"),
                   ",", fixed = TRUE)[[1L]]
n_grid <- as.integer(n_grid)
pseudo_n <- as.integer(take("--pseudo-n", if (smoke) "10000" else "100000"))
seed_base <- as.integer(take("--seed-base", "20260703"))
stopifnot(reps > 0L, all(n_grid > 2L), pseudo_n > 1000L, is.finite(seed_base))

dir.create("results", showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

loadings <- c(0.85, 0.78, 0.70, 0.62)
ordered_vars <- paste0("x", seq_along(loadings))
block <- rep(1L, length(loadings))
model <- paste("f =~", paste(ordered_vars, collapse = " + "))
spec <- magmaan::model_spec(model, ordered = ordered_vars, parameterization = "delta")

cut_regimes <- list(
  balanced = rep(list(c(-0.65, 0.45)), length(loadings)),
  threshold_extreme = rep(list(c(-1.35, 0.95)), length(loadings))
)

dgp_names <- c("skew_eta", "t_eta_eps", "local_dependence")

latent_response <- function(n, dgp, seed) {
  set.seed(seed)
  theta_sqrt <- sqrt(1 - loadings^2)

  if (dgp == "skew_eta") {
    eta <- (stats::rchisq(n, df = 3) - 3) / sqrt(6)
    eps <- matrix(stats::rnorm(n * length(loadings)), nrow = n)
    return(sweep(eps, 2L, theta_sqrt, `*`) +
             tcrossprod(eta, loadings))
  }

  if (dgp == "t_eta_eps") {
    eta <- stats::rt(n, df = 3) / sqrt(3)
    eps <- matrix(stats::rt(n * length(loadings), df = 3) / sqrt(3), nrow = n)
    return(sweep(eps, 2L, theta_sqrt, `*`) +
             tcrossprod(eta, loadings))
  }

  if (dgp == "local_dependence") {
    eta <- stats::rnorm(n)
    residual_R <- diag(length(loadings))
    residual_R[1L, 2L] <- residual_R[2L, 1L] <- 0.35
    eps <- matrix(stats::rnorm(n * length(loadings)), nrow = n) %*%
      chol(residual_R)
    return(sweep(eps, 2L, theta_sqrt, `*`) +
             tcrossprod(eta, loadings))
  }

  stop("unknown dgp: ", dgp, call. = FALSE)
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

min_variable_category_count <- function(dat) {
  min(vapply(dat, function(x) {
    min(tabulate(as.integer(x), nbins = nlevels(x)))
  }, integer(1L)))
}

omega_value <- function(R) {
  dimnames(R) <- list(ordered_vars, ordered_vars)
  core$measures_reliability_omega_multidim(R, block = block, target = "total")$value
}

estimate_pseudo_target <- function(dgp, cut_name) {
  dgp_id <- match(dgp, dgp_names)
  cut_id <- match(cut_name, names(cut_regimes))
  seed <- seed_base + 70000000L + 1000000L * dgp_id + 100000L * cut_id
  Y <- latent_response(pseudo_n, dgp, seed)
  dat <- discretize_ord(Y, cut_regimes[[cut_name]])
  stats <- core$data_ordinal_stats_from_df(dat, spec)
  omega <- core$measures_reliability_ordinal_polychoric_omega(stats, block = block)
  data.frame(
    dgp = dgp,
    cuts = cut_name,
    pseudo_n = pseudo_n,
    target_pseudo = omega$value,
    target_continuous = omega_value(stats::cor(Y)),
    pseudo_minus_continuous = omega$value - omega_value(stats::cor(Y)),
    stringsAsFactors = FALSE)
}

message("estimating pseudo-targets")
targets <- do.call(rbind, lapply(dgp_names, function(dgp) {
  do.call(rbind, lapply(names(cut_regimes), function(cut_name) {
    message(sprintf("target dgp=%s cuts=%s", dgp, cut_name))
    estimate_pseudo_target(dgp, cut_name)
  }))
}))

target_lookup <- function(dgp, cut_name) {
  targets[targets$dgp == dgp & targets$cuts == cut_name, , drop = FALSE]
}

fit_one <- function(dgp, cut_name, n, rep_id) {
  dgp_id <- match(dgp, dgp_names)
  cut_id <- match(cut_name, names(cut_regimes))
  seed <- seed_base + 10000000L * dgp_id + 1000000L * cut_id + 10000L * n + rep_id
  Y <- latent_response(n, dgp, seed)
  dat <- discretize_ord(Y, cut_regimes[[cut_name]])
  min_cat_count <- min_variable_category_count(dat)
  target_row <- target_lookup(dgp, cut_name)

  stats <- tryCatch(
    core$data_ordinal_stats_from_df(dat, spec),
    error = identity)
  if (inherits(stats, "error")) {
    return(data.frame(
      dgp = dgp, cuts = cut_name, n = n, rep = rep_id, ok = FALSE,
      value = NA_real_, se = NA_real_,
      target_pseudo = target_row$target_pseudo,
      target_continuous = target_row$target_continuous,
      ci_lo = NA_real_, ci_hi = NA_real_,
      covered_pseudo = NA, covered_continuous = NA,
      width = NA_real_, min_variable_category_count = min_cat_count,
      message = conditionMessage(stats), stringsAsFactors = FALSE))
  }

  omega <- tryCatch(
    core$measures_reliability_ordinal_polychoric_omega(stats, block = block),
    error = identity)
  if (inherits(omega, "error")) {
    return(data.frame(
      dgp = dgp, cuts = cut_name, n = n, rep = rep_id, ok = FALSE,
      value = NA_real_, se = NA_real_,
      target_pseudo = target_row$target_pseudo,
      target_continuous = target_row$target_continuous,
      ci_lo = NA_real_, ci_hi = NA_real_,
      covered_pseudo = NA, covered_continuous = NA,
      width = NA_real_, min_variable_category_count = min_cat_count,
      message = conditionMessage(omega), stringsAsFactors = FALSE))
  }

  lo <- omega$value - 1.96 * omega$se
  hi <- omega$value + 1.96 * omega$se
  data.frame(
    dgp = dgp,
    cuts = cut_name,
    n = n,
    rep = rep_id,
    ok = TRUE,
    value = omega$value,
    se = omega$se,
    target_pseudo = target_row$target_pseudo,
    target_continuous = target_row$target_continuous,
    ci_lo = lo,
    ci_hi = hi,
    covered_pseudo = lo <= target_row$target_pseudo &&
      target_row$target_pseudo <= hi,
    covered_continuous = lo <= target_row$target_continuous &&
      target_row$target_continuous <= hi,
    width = hi - lo,
    min_variable_category_count = min_cat_count,
    message = "",
    stringsAsFactors = FALSE)
}

rows <- vector("list", length(dgp_names) * length(cut_regimes) * length(n_grid) * reps)
k <- 1L
for (dgp in dgp_names) {
  for (cut_name in names(cut_regimes)) {
    for (n in n_grid) {
      for (r in seq_len(reps)) {
        if (r %% max(1L, reps %/% 10L) == 0L) {
          message(sprintf("dgp=%s cuts=%s n=%d rep=%d/%d",
                          dgp, cut_name, n, r, reps))
        }
        rows[[k]] <- fit_one(dgp, cut_name, n, r)
        k <- k + 1L
      }
    }
  }
}
raw <- do.call(rbind, rows)

cell_summary <- do.call(rbind, lapply(
  split(raw, list(raw$dgp, raw$cuts, raw$n), drop = TRUE),
  function(x) {
    ok <- x[x$ok, , drop = FALSE]
    data.frame(
      dgp = x$dgp[[1]],
      cuts = x$cuts[[1]],
      n = x$n[[1]],
      reps = nrow(x),
      failures = sum(!x$ok),
      target_pseudo = x$target_pseudo[[1]],
      target_continuous = x$target_continuous[[1]],
      pseudo_minus_continuous = x$target_pseudo[[1]] - x$target_continuous[[1]],
      mean_value = if (nrow(ok)) mean(ok$value) else NA_real_,
      bias_pseudo = if (nrow(ok)) mean(ok$value - ok$target_pseudo) else NA_real_,
      sd_value = if (nrow(ok)) stats::sd(ok$value) else NA_real_,
      mean_se = if (nrow(ok)) mean(ok$se) else NA_real_,
      median_se = if (nrow(ok)) stats::median(ok$se) else NA_real_,
      coverage_pseudo_95 = if (nrow(ok)) mean(ok$covered_pseudo) else NA_real_,
      coverage_continuous_95 = if (nrow(ok)) mean(ok$covered_continuous) else NA_real_,
      mean_width_95 = if (nrow(ok)) mean(ok$width) else NA_real_,
      median_width_95 = if (nrow(ok)) stats::median(ok$width) else NA_real_,
      min_variable_category_count_p05 = if (nrow(ok)) {
        unname(stats::quantile(ok$min_variable_category_count, 0.05))
      } else NA_real_,
      stringsAsFactors = FALSE)
  }))
cell_summary <- cell_summary[order(cell_summary$dgp, cell_summary$cuts, cell_summary$n), ]
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
  loadings = paste(loadings, collapse = ","),
  dgps = paste(dgp_names, collapse = ","),
  cut_regimes = paste(names(cut_regimes), collapse = ","),
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE
), "results/metadata.csv", row.names = FALSE)
message("wrote results/pseudo_targets.csv")
message("wrote results/simulation_raw.csv")
message("wrote results/simulation_summary.csv")
message("wrote results/metadata.csv")
