#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "score_sandwich.R"))
set_single_threaded_math()

sem_seed <- function(seed) {
  as.integer(seed %% (.Machine$integer.max - 1L))
}

usage <- function() cat(
  "Usage: Rscript investigate_pseudonull_information.R [options]\n\n",
  "Compares expected-H0, observed-H0, and observed-H1 global-score\n",
  "geometries under an exact nonnormal-MAR Gaussian-FIML pseudo-null.\n\n",
  "  --reps N          Replications per cell (default 1000).\n",
  "  --n CSV           Sample sizes (default 200,500,2000).\n",
  "  --beta CSV        MAR logit slopes (default 0,1.5,3,5).\n",
  "  --missing P       Marginal joint-missing rate (default 0.5).\n",
  "  --population-n N  Large-sample pseudo-null gate (default 200000).\n",
  "  --cores N         Parallel cell workers (default up to 4).\n",
  "  --seed-base N     Deterministic seed base.\n",
  "  --results-dir P   Output directory.\n",
  "  --help            Show this help.\n", sep = "")

opts <- list(
  reps = 1000L,
  n = c(200L, 500L, 2000L),
  beta = c(0, 1.5, 3, 5),
  missing = 0.5,
  population_n = 200000L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  seed_base = 20260903L,
  results_dir = NULL)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) {
    usage()
    quit(save = "no", status = 0L)
  } else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(parse_csv_arg(take()))
  else if (arg == "--beta") opts$beta <- as.numeric(parse_csv_arg(take()))
  else if (arg == "--missing") opts$missing <- as.numeric(take())
  else if (arg == "--population-n") opts$population_n <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(
  opts$reps > 0L,
  all(opts$n >= 80L),
  all(is.finite(opts$beta)),
  all(opts$beta >= 0),
  is.finite(opts$missing),
  opts$missing > 0,
  opts$missing < 1,
  opts$population_n >= 10000L,
  opts$cores > 0L)

results <- opts$results_dir %||% file.path(
  script_dir, "results", "pseudonull-information")
dir.create(results, recursive = TRUE, showWarnings = FALSE)

ov <- paste0("x", 1:4)
model <- paste(c(
  paste0(ov, " ~~ ", ov),
  "x2 ~~ x3 + x4",
  "x3 ~~ x4",
  paste0(ov, " ~ 1")), collapse = "\n")

# Every variable is an independent standardized exponential. Variables x2:x4
# are jointly missing with probability logit^{-1}(a + beta*x1), where x1 is
# always observed. This is MAR with strict positivity. Independence makes the
# saturated Gaussian-FIML pseudo-target diagonal for every beta: selection on
# x1 cannot change the marginal moments of the independent variables x2:x4 or
# create covariances with them. The fitted model is saturated within the
# x2:x4 block and tests only x1's three covariances with that block. It is
# therefore the exact estimator-level pseudo-null while concentrating all
# three test directions on the information mismatch.
driver_grid <- stats::qexp((seq_len(1000000L) - 0.5) / 1000000L) - 1
calibrate_intercept <- function(beta) {
  if (beta == 0) return(stats::qlogis(opts$missing))
  stats::uniroot(
    function(intercept) {
      mean(stats::plogis(intercept + beta * driver_grid)) - opts$missing
    }, c(-40, 40), tol = 1e-12)$root
}
intercepts <- vapply(opts$beta, calibrate_intercept, numeric(1L))

population_check <- do.call(rbind, lapply(seq_along(opts$beta), function(k) {
  beta <- opts$beta[[k]]
  intercept <- intercepts[[k]]
  p_missing <- stats::plogis(intercept + beta * driver_grid)
  p_observed <- 1 - p_missing
  observed_mean <- weighted.mean(driver_grid, p_observed)
  observed_variance <- weighted.mean(
    (driver_grid - observed_mean)^2, p_observed)
  data.frame(
    beta = beta,
    intercept = intercept,
    missing_rate = mean(p_missing),
    driver_mean_when_others_observed = observed_mean,
    driver_variance_when_others_observed = observed_variance,
    fisher_to_actual_variance_ratio = 1 / observed_variance,
    stringsAsFactors = FALSE)
}))
write_csv(population_check, file.path(results, "population_check.csv"))

geometries <- data.frame(
  geometry = c(
    "expected-H0", "observed-H0/expected-metric",
    "observed-H0-light/expected-metric",
    "observed-H0-sqrt/expected-metric",
    "observed-H1/expected-metric", "observed-H0", "observed-H1"),
  sensitivity = c(
    "expected", "observed", "observed-shrink-light",
    "observed-shrink-sqrt", "observed-h1", "observed", "observed-h1"),
  metric = c(
    "expected", "expected", "expected", "expected", "expected",
    "observed", "observed-h1"),
  stringsAsFactors = FALSE)

fmg_peba4 <- function(score) {
  magmaan:::infer_fmg_test(
    score$statistic_effective, score$df, score$eigenvalues,
    method = "peba", param = 4, truncate_negative = TRUE)$p_value
}

large_sample_check <- do.call(rbind, lapply(seq_along(opts$beta), function(k) {
  beta <- opts$beta[[k]]
  intercept <- intercepts[[k]]
  set.seed(sem_seed(opts$seed_base + 900000001L + k))
  X <- matrix(
    stats::rexp(opts$population_n * 4L) - 1,
    nrow = opts$population_n, ncol = 4L)
  colnames(X) <- ov
  jointly_missing <- stats::runif(opts$population_n) <
    stats::plogis(intercept + beta * X[, 1L])
  X[jointly_missing, 2:4] <- NA_real_
  fit <- magmaan(
    model, as.data.frame(X), estimator = "FIML",
    optimizer = "nlopt-lbfgs-slsqp-fallback")
  if (!isTRUE(fit$converged)) {
    stop("large-sample pseudo-null fit did not converge", call. = FALSE)
  }
  lrt <- fmg_tests(fit, tests = "all")
  stage1 <- magmaan_core$estimate_saturated_em_moments(fit$raw_data)
  data.frame(
    beta = beta,
    n = opts$population_n,
    missing_rate = mean(jointly_missing),
    lrt = lrt$base_statistic[[1L]],
    lrt_per_case = lrt$base_statistic[[1L]] / opts$population_n,
    max_abs_tested_h1_covariance = max(abs(stage1$cov[[1L]][1L, 2:4])),
    stringsAsFactors = FALSE)
}))
write_csv(large_sample_check, file.path(results, "large_sample_check.csv"))

one_rep <- function(n, beta, intercept, rep_id, cell_id) {
  seed <- sem_seed(opts$seed_base + cell_id * 100003L + rep_id)
  set.seed(seed)
  X <- matrix(stats::rexp(n * 4L) - 1, nrow = n, ncol = 4L)
  colnames(X) <- ov
  p_missing <- stats::plogis(intercept + beta * X[, 1L])
  jointly_missing <- stats::runif(n) < p_missing
  X[jointly_missing, 2:4] <- NA_real_

  fit <- tryCatch(
    magmaan(
      model, as.data.frame(X), estimator = "FIML",
      optimizer = "nlopt-lbfgs-slsqp-fallback"),
    error = function(e) e)
  if (inherits(fit, "error") || !isTRUE(fit$converged)) {
    error <- if (inherits(fit, "error")) conditionMessage(fit) else
      "FIML fit did not converge"
    return(data.frame(
      cell_id = cell_id, n = n, beta = beta, rep = rep_id, seed = seed,
      missing_rate = mean(jointly_missing), geometry = geometries$geometry,
      ok = FALSE, error = error, statistic = NA_real_, df = NA_integer_,
      eigen_min = NA_real_, eigen_mean = NA_real_, eigen_max = NA_real_,
      p_sb = NA_real_, p_peba4 = NA_real_, p_all = NA_real_,
      p_sandwich = NA_real_, sandwich_ok = FALSE, sandwich_error = error,
      p_sandwich_centered_chisq = NA_real_,
      p_sandwich_hotelling = NA_real_,
      p_sandwich_shrink_light = NA_real_,
      p_sandwich_shrink_sqrt = NA_real_,
      sandwich_rho_light = NA_real_, sandwich_rho_sqrt = NA_real_,
      sandwich_centered_condition = NA_real_, stringsAsFactors = FALSE))
  }

  do.call(rbind, lapply(seq_len(nrow(geometries)), function(g) {
    spec <- geometries[g, ]
    score <- tryCatch(
      global_score_flip_test(
        fit, n_flips = 1L, seed = seed + 700001L,
        sensitivity = spec$sensitivity, metric = spec$metric),
      error = function(e) e)
    if (inherits(score, "error")) {
      return(data.frame(
        cell_id = cell_id, n = n, beta = beta, rep = rep_id, seed = seed,
        missing_rate = mean(jointly_missing), geometry = spec$geometry,
        ok = FALSE, error = conditionMessage(score), statistic = NA_real_,
        df = NA_integer_, eigen_min = NA_real_, eigen_mean = NA_real_,
        eigen_max = NA_real_, p_sb = NA_real_, p_peba4 = NA_real_,
        p_all = NA_real_, p_sandwich = NA_real_, sandwich_ok = FALSE,
        sandwich_error = conditionMessage(score),
        p_sandwich_centered_chisq = NA_real_,
        p_sandwich_hotelling = NA_real_,
        p_sandwich_shrink_light = NA_real_,
        p_sandwich_shrink_sqrt = NA_real_,
        sandwich_rho_light = NA_real_, sandwich_rho_sqrt = NA_real_,
        sandwich_centered_condition = NA_real_, stringsAsFactors = FALSE))
    }
    eigenvalues <- score$eigenvalues
    out <- data.frame(
      cell_id = cell_id, n = n, beta = beta, rep = rep_id, seed = seed,
      missing_rate = mean(jointly_missing), geometry = spec$geometry,
      ok = TRUE, error = "", statistic = score$statistic_effective,
      df = as.integer(score$df), eigen_min = min(eigenvalues),
      eigen_mean = mean(eigenvalues), eigen_max = max(eigenvalues),
      p_sb = score$p_mean_scaled, p_peba4 = fmg_peba4(score),
      p_all = score$p_mixture, p_sandwich = score$p_sandwich,
      stringsAsFactors = FALSE)
    sandwich <- tryCatch(
      score_sandwich_diagnostics(score), error = function(e) e)
    if (inherits(sandwich, "error")) {
      out$sandwich_ok <- FALSE
      out$sandwich_error <- conditionMessage(sandwich)
      out$p_sandwich_centered_chisq <- NA_real_
      out$p_sandwich_hotelling <- NA_real_
      out$p_sandwich_shrink_light <- NA_real_
      out$p_sandwich_shrink_sqrt <- NA_real_
      out$sandwich_rho_light <- NA_real_
      out$sandwich_rho_sqrt <- NA_real_
      out$sandwich_centered_condition <- NA_real_
    } else {
      out$sandwich_ok <- TRUE
      out$sandwich_error <- ""
      out$p_sandwich_centered_chisq <-
        sandwich$sandwich_p_centered_chisq
      out$p_sandwich_hotelling <- sandwich$sandwich_p_hotelling
      out$p_sandwich_shrink_light <- sandwich$sandwich_p_shrink_light
      out$p_sandwich_shrink_sqrt <- sandwich$sandwich_p_shrink_sqrt
      out$sandwich_rho_light <- sandwich$sandwich_rho_light
      out$sandwich_rho_sqrt <- sandwich$sandwich_rho_sqrt
      out$sandwich_centered_condition <-
        sandwich$sandwich_centered_condition
    }
    out$ok <- all(is.finite(unlist(out[c(
      "statistic", "df", "eigen_min", "eigen_mean", "eigen_max",
      "p_sb", "p_peba4", "p_all")])))
    if (!out$ok) out$error <- "non-finite score result"
    out
  }))
}

grid <- expand.grid(
  n = opts$n,
  beta = opts$beta,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid$intercept <- intercepts[match(grid$beta, opts$beta)]

one_cell <- function(index) {
  cell <- grid[index, ]
  message(sprintf(
    "cell %d/%d: n=%d beta=%g", index, nrow(grid), cell$n, cell$beta))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(
      cell$n, cell$beta, cell$intercept, rep_id,
      cell$cell_id)
  }))
}

cat(sprintf(
  "cells=%d reps=%d geometries=%d cores=%d\n",
  nrow(grid), opts$reps, nrow(geometries), opts$cores))
begin <- proc.time()[["elapsed"]]
pieces <- if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  parallel::mclapply(
    seq_len(nrow(grid)), one_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else lapply(seq_len(nrow(grid)), one_cell)
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep, raw$geometry), ]
row.names(raw) <- NULL
write_csv(raw, file.path(results, "replications.csv"))

summary_groups <- split(
  raw, interaction(raw$n, raw$beta, raw$geometry, drop = TRUE))
summary <- do.call(rbind, lapply(summary_groups, function(z) {
  good <- z[z$ok, ]
  data.frame(
    n = z$n[[1L]], beta = z$beta[[1L]], geometry = z$geometry[[1L]],
    attempted = nrow(z), usable = nrow(good),
    missing_rate = mean(z$missing_rate),
    reject_sb = mean(good$p_sb < .05),
    reject_peba4 = mean(good$p_peba4 < .05),
    reject_all = mean(good$p_all < .05),
    reject_sandwich = mean(good$p_sandwich < .05, na.rm = TRUE),
    reject_sandwich_centered_chisq = mean(
      good$p_sandwich_centered_chisq < .05, na.rm = TRUE),
    reject_sandwich_hotelling = mean(
      good$p_sandwich_hotelling < .05, na.rm = TRUE),
    reject_sandwich_shrink_light = mean(
      good$p_sandwich_shrink_light < .05, na.rm = TRUE),
    reject_sandwich_shrink_sqrt = mean(
      good$p_sandwich_shrink_sqrt < .05, na.rm = TRUE),
    sandwich_usable = sum(good$sandwich_ok),
    mean_sandwich_centered_condition = mean(
      good$sandwich_centered_condition, na.rm = TRUE),
    mean_statistic = mean(good$statistic),
    mean_eigen_min = mean(good$eigen_min),
    mean_eigen_mean = mean(good$eigen_mean),
    mean_eigen_max = mean(good$eigen_max),
    stringsAsFactors = FALSE)
}))
summary <- summary[order(summary$beta, summary$n, summary$geometry), ]
row.names(summary) <- NULL
write_csv(summary, file.path(results, "summary.csv"))

metadata <- data.frame(
  key = c(
    "reps", "sample_sizes", "beta", "target_missing", "population_n", "cores",
    "seed_base", "elapsed_seconds"),
  value = c(
    opts$reps, paste(opts$n, collapse = ","), paste(opts$beta, collapse = ","),
    opts$missing, opts$population_n, opts$cores, opts$seed_base,
    proc.time()[["elapsed"]] - begin),
  stringsAsFactors = FALSE)
write_csv(metadata, file.path(results, "metadata.csv"))

print(population_check, row.names = FALSE)
print(large_sample_check, row.names = FALSE)
print(summary, row.names = FALSE)
cat(sprintf("elapsed %.1f seconds\n", proc.time()[["elapsed"]] - begin))
