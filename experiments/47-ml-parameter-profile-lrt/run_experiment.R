#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
has_flag <- function(flag) any(args == flag)
arg_value <- function(flag, default) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1L]]
}

smoke <- has_flag("--smoke")
reps <- as.integer(arg_value("--reps", if (smoke) "20" else "200"))
n_grid <- strsplit(arg_value("--n-grid", if (smoke) "30,50" else "30,50,100,500"), ",", fixed = TRUE)[[1L]]
n_grid <- as.integer(n_grid)
seed_base <- as.integer(arg_value("--seed-base", "20260702"))

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

draw_data <- function(n, seed) {
  set.seed(seed)
  Z <- matrix(rnorm(n * nrow(Sigma)), nrow = n)
  X <- Z %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  X
}

fit_and_test <- function(n, rep_id) {
  X <- draw_data(n, seed_base + 100000L * n + rep_id)
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

dir.create("results", showWarnings = FALSE)
raw_rows <- list()
fail_rows <- list()
row_id <- 1L
fail_id <- 1L
for (n in n_grid) {
  for (r in seq_len(reps)) {
    if (r %% max(1L, reps %/% 10L) == 0L) {
      message(sprintf("n=%d rep=%d/%d", n, r, reps))
    }
    out <- tryCatch(
      fit_and_test(n, r),
      error = function(e) {
        data.frame(n = n, rep = r, error = conditionMessage(e),
                   stringsAsFactors = FALSE)
      }
    )
    if ("error" %in% names(out)) {
      fail_rows[[fail_id]] <- out
      fail_id <- fail_id + 1L
    } else {
      raw_rows[[row_id]] <- out
      row_id <- row_id + 1L
    }
  }
}

raw <- if (length(raw_rows)) do.call(rbind, raw_rows) else data.frame()
fail <- if (length(fail_rows)) {
  do.call(rbind, fail_rows)
} else {
  data.frame(n = integer(), rep = integer(), error = character())
}
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
meta <- data.frame(
  smoke = smoke,
  reps = reps,
  n_grid = paste(n_grid, collapse = ","),
  seed_base = seed_base,
  target_loading = target_loading,
  stringsAsFactors = FALSE
)

write.csv(raw, "results/profile_lrt_raw.csv", row.names = FALSE)
write.csv(summary, "results/profile_lrt_summary.csv", row.names = FALSE)
write.csv(fail, "results/profile_lrt_failures.csv", row.names = FALSE)
write.csv(meta, "results/metadata.csv", row.names = FALSE)
message("wrote results/profile_lrt_raw.csv")
message("wrote results/profile_lrt_summary.csv")
message("wrote results/profile_lrt_failures.csv")
message("wrote results/metadata.csv")
