#!/usr/bin/env Rscript

usage <- function() {
  cat(
"Usage: Rscript run_experiment.R [--smoke] [--tol X] [--seed-base N]

Audit whether the current ordinal observed-score covariance omega equals the
direct one-factor ordinal true-score target Var(sum E[Y_j | eta]) / Var(sum Y_j).

Options:
  --smoke        Run four representative cells instead of the full grid.
  --tol X        Relative tolerance for one-dimensional integrations (default 1e-10).
  --seed-base N  Recorded for reproducibility metadata (default 20260702).
  --help         Show this help.
")
}

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  usage()
  quit(status = 0)
}

take <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(flag, " needs a value", call. = FALSE)
  args[[i + 1L]]
}

opt <- list(
  smoke = "--smoke" %in% args,
  tol = as.numeric(take("--tol", "1e-10")),
  seed_base = as.integer(take("--seed-base", "20260702"))
)
stopifnot(is.finite(opt$tol), opt$tol > 0)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core

loading_patterns <- list(
  tau_equal = rep(0.75, 4),
  congeneric = c(0.85, 0.78, 0.70, 0.62),
  weak_tail = c(0.80, 0.68, 0.55, 0.42),
  high = c(0.95, 0.90, 0.84, 0.78)
)

threshold_patterns <- list(
  equal_balanced = rep(list(c(-0.65, 0.45)), 4),
  equal_extreme = rep(list(c(-1.35, 0.95)), 4),
  mixed_three = list(
    c(-1.35, 0.95),
    c(-0.65, 0.45),
    c(-0.20, 1.20),
    c(-1.10, -0.10)
  ),
  mixed_four = list(
    c(-1.20, -0.20, 0.80),
    c(-0.80, 0.00, 1.30),
    c(-1.60, -0.60, 0.20),
    c(-0.20, 0.70, 1.60)
  )
)

integrate_normal <- function(f) {
  stats::integrate(
    function(z) f(z) * stats::dnorm(z),
    lower = -Inf, upper = Inf,
    rel.tol = opt$tol, subdivisions = 500L
  )$value
}

item_moments <- function(lambda, cuts, eta) {
  eta <- as.numeric(eta)
  sd_eps <- sqrt(1 - lambda^2)
  bounds <- c(-Inf, cuts, Inf)
  scores <- seq_len(length(bounds) - 1L) - 1L
  probs <- vapply(seq_along(scores), function(k) {
    lo <- (bounds[[k]] - lambda * eta) / sd_eps
    hi <- (bounds[[k + 1L]] - lambda * eta) / sd_eps
    stats::pnorm(hi) - stats::pnorm(lo)
  }, numeric(length(eta)))
  probs[probs < 0] <- 0
  list(
    mean = as.numeric(probs %*% scores),
    second = as.numeric(probs %*% (scores^2))
  )
}

communality_ratio_of_sums <- function(S, item) {
  others <- setdiff(seq_len(nrow(S)), item)
  pairs <- utils::combn(others, 2L)
  numerator <- sum(S[item, pairs[1L, ]] * S[item, pairs[2L, ]])
  denominator <- sum(S[cbind(pairs[1L, ], pairs[2L, ])])
  numerator / denominator
}

cell_audit <- function(loading_name, threshold_name) {
  lambda <- loading_patterns[[loading_name]]
  cuts <- threshold_patterns[[threshold_name]]
  p <- length(lambda)
  stopifnot(length(cuts) == p)

  means <- numeric(p)
  second <- numeric(p)
  true_diag <- numeric(p)
  S <- matrix(0, p, p)
  true_score_cov <- matrix(0, p, p)

  for (j in seq_len(p)) {
    means[[j]] <- integrate_normal(function(z) {
      item_moments(lambda[[j]], cuts[[j]], z)$mean
    })
    second[[j]] <- integrate_normal(function(z) {
      item_moments(lambda[[j]], cuts[[j]], z)$second
    })
    true_second <- integrate_normal(function(z) {
      m <- item_moments(lambda[[j]], cuts[[j]], z)$mean
      m * m
    })
    S[j, j] <- second[[j]] - means[[j]]^2
    true_diag[[j]] <- true_second - means[[j]]^2
    true_score_cov[j, j] <- true_diag[[j]]
  }

  for (j in seq_len(p - 1L)) {
    for (i in (j + 1L):p) {
      cross <- integrate_normal(function(z) {
        mi <- item_moments(lambda[[i]], cuts[[i]], z)$mean
        mj <- item_moments(lambda[[j]], cuts[[j]], z)$mean
        mi * mj
      })
      S[i, j] <- cross - means[[i]] * means[[j]]
      S[j, i] <- S[i, j]
      true_score_cov[i, j] <- S[i, j]
      true_score_cov[j, i] <- S[i, j]
    }
  }

  h <- vapply(seq_len(p), function(j) communality_ratio_of_sums(S, j), numeric(1))
  C <- S
  diag(C) <- h

  denominator <- sum(S)
  direct_numerator <- sum(true_score_cov)
  covariance_numerator <- sum(C)
  direct_omega <- direct_numerator / denominator
  covariance_omega_manual <- covariance_numerator / denominator
  covariance_omega_core <- core$measures_reliability_omega_multidim(
    S, block = rep(1L, p), target = "total")$value

  probs <- unlist(lapply(cuts, function(th) {
    diff(stats::pnorm(c(-Inf, th, Inf)))
  }))

  summary <- data.frame(
    cell = paste(loading_name, threshold_name, sep = "__"),
    loading_pattern = loading_name,
    threshold_pattern = threshold_name,
    n_items = p,
    min_category_prob = min(probs),
    current_covariance_omega = covariance_omega_core,
    direct_true_score_omega = direct_omega,
    difference = covariance_omega_core - direct_omega,
    abs_difference = abs(covariance_omega_core - direct_omega),
    denominator = denominator,
    current_numerator = covariance_numerator,
    direct_numerator = direct_numerator,
    numerator_ratio = covariance_numerator / direct_numerator,
    max_abs_communality_gap = max(abs(h - true_diag)),
    mean_abs_communality_gap = mean(abs(h - true_diag)),
    manual_core_gap = covariance_omega_manual - covariance_omega_core,
    stringsAsFactors = FALSE
  )

  item <- data.frame(
    cell = summary$cell,
    item = seq_len(p),
    loading = lambda,
    n_categories = vapply(cuts, length, integer(1)) + 1L,
    observed_variance = diag(S),
    direct_true_score_variance = true_diag,
    covariance_omega_communality = h,
    communality_gap = h - true_diag,
    stringsAsFactors = FALSE
  )
  list(summary = summary, item = item)
}

grid <- expand.grid(
  loading_pattern = names(loading_patterns),
  threshold_pattern = names(threshold_patterns),
  stringsAsFactors = FALSE
)
if (opt$smoke) {
  keep <- with(grid,
               (loading_pattern == "tau_equal" & threshold_pattern == "equal_balanced") |
               (loading_pattern == "congeneric" & threshold_pattern == "equal_balanced") |
               (loading_pattern == "congeneric" & threshold_pattern == "mixed_three") |
               (loading_pattern == "high" & threshold_pattern == "mixed_four"))
  grid <- grid[keep, , drop = FALSE]
}

summary_rows <- vector("list", nrow(grid))
item_rows <- vector("list", nrow(grid))
for (r in seq_len(nrow(grid))) {
  message(sprintf("cell %d/%d: %s x %s", r, nrow(grid),
                  grid$loading_pattern[[r]], grid$threshold_pattern[[r]]))
  out <- cell_audit(grid$loading_pattern[[r]], grid$threshold_pattern[[r]])
  summary_rows[[r]] <- out$summary
  item_rows[[r]] <- out$item
}

summary <- do.call(rbind, summary_rows)
item <- do.call(rbind, item_rows)
summary <- summary[order(-summary$abs_difference), ]
row.names(summary) <- NULL

write.csv(summary, "results/target_summary.csv", row.names = FALSE)
write.csv(item, "results/item_diagnostics.csv", row.names = FALSE)
write.csv(data.frame(
  smoke = opt$smoke,
  n_cells = nrow(grid),
  integration_tol = opt$tol,
  seed_base = opt$seed_base,
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE
), "results/metadata.csv", row.names = FALSE)
message("wrote results/target_summary.csv")
message("wrote results/item_diagnostics.csv")
message("wrote results/metadata.csv")
