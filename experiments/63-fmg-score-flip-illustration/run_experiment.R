#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "design.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--probe] [options]\n\n",
  "Focused FMG weak-invariance score-flip comparison.\n\n",
  "  --smoke          32 cells, 2 reps, 19 flips (default).\n",
  "  --probe          32 cells, 200 reps, 199 flips.\n",
  "  --reps N         Override replications per cell.\n",
  "  --flips N        Override random flips per replication.\n",
  "  --cores N        Parallel cell workers.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(mode = "smoke", reps = NULL, flips = NULL,
             cores = min(8L, max(1L, parallel::detectCores() - 2L)),
             seed_base = 20260713L, results_dir = NULL)
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--probe") opts$mode <- "probe"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 2L else 200L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 19L else 199L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L,
          opts$seed_base >= 0L)
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

fmg_validate_design()
grid <- fmg_design_grid()
spec_cache <- new.env(parent = emptyenv())
get_specs <- function(p, groups) {
  key <- paste(p, groups, sep = "_")
  if (is.null(spec_cache[[key]])) spec_cache[[key]] <- fmg_model_specs(p, groups)
  spec_cache[[key]]
}

nested_tests <- c("std_ml", "sb_ml", "mv_ml", "pall_ml", "all_ml",
                  "std_rls", "ss_rls", "peba4_rls")
p_columns <- c(
  "p_flip_basic", "p_flip_effective", "p_flip_standardized",
  "p_score_chisq", "p_score_sb", "p_score_mv", "p_score_ss",
  "p_score_peba4", "p_score_pall", "p_score_all", "p_score_sandwich",
  paste0("p_nested_", nested_tests))

empty_p <- function() stats::setNames(rep(NA_real_, length(p_columns)), p_columns)

one_rep <- function(cell, rep_id, sampler, specs) {
  total_begin <- proc.time()[["elapsed"]]
  draw_begin <- proc.time()[["elapsed"]]
  sample <- sampler$draw(rep_id)
  draw_seconds <- proc.time()[["elapsed"]] - draw_begin

  fit_begin <- proc.time()[["elapsed"]]
  fits <- tryCatch(lapply(specs, function(spec) magmaan(
    spec, sample$data, estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none")),
    error = function(e) e)
  fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  p <- empty_p()
  base <- data.frame(
    rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
    fit_error = "", flip_error = "", nested_error = "",
    draw_seconds = draw_seconds, fit_seconds = fit_seconds,
    flip_seconds = NA_real_, nested_seconds = NA_real_, total_seconds = NA_real_,
    flip_setup_seconds = NA_real_, flip_score_seconds = NA_real_,
    flip_standardization_seconds = NA_real_, flip_asymptotic_seconds = NA_real_,
    nuisance_stationarity_norm = NA_real_,
    mean_variance_relative_shift = NA_real_,
    max_variance_relative_shift = NA_real_,
    min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
    sandwich_available = NA, sandwich_condition = NA_real_,
    score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
    score_eigen_ratio = NA_real_)
  if (inherits(fits, "error")) {
    base$fit_error <- conditionMessage(fits)
    base$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(cbind(base, as.data.frame(as.list(p))))
  }
  names(fits) <- names(specs)
  if (!all(vapply(fits, function(x) isTRUE(x$converged), logical(1)))) {
    base$fit_error <- "one or both ML fits did not converge"
    base$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(cbind(base, as.data.frame(as.list(p))))
  }
  base$fit_ok <- TRUE

  flip_begin <- proc.time()[["elapsed"]]
  flip <- tryCatch(score_flip_test(
    fits$H1, fits$H0, sample$blocks, n_flips = opts$flips,
    seed = opts$seed_base + cell$cell_id * 100000L + rep_id),
    error = function(e) e)
  base$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
  if (inherits(flip, "error")) {
    base$flip_error <- conditionMessage(flip)
  } else {
    stopifnot(flip$df == cell$df)
    base$flip_ok <- TRUE
    score_fmg <- function(method, param = 4) tryCatch(
      magmaan:::infer_fmg_test(
        flip$statistic_effective, flip$df, flip$eigenvalues,
        method = method, param = param)$p_value,
      error = function(e) NA_real_)
    p[c("p_flip_basic", "p_flip_effective", "p_flip_standardized",
        "p_score_chisq", "p_score_sb", "p_score_all",
        "p_score_sandwich")] <-
      c(flip$p_basic, flip$p_effective, flip$p_standardized,
        flip$p_chisq, flip$p_mean_scaled, flip$p_mixture, flip$p_sandwich)
    p["p_score_mv"] <- score_fmg("mv")
    p["p_score_ss"] <- score_fmg("ss")
    p["p_score_peba4"] <- score_fmg("peba", 4)
    p["p_score_pall"] <- score_fmg("penalized_all")
    base$flip_setup_seconds <- flip$setup_seconds
    base$flip_score_seconds <- flip$resampling_score_seconds
    base$flip_standardization_seconds <-
      flip$resampling_standardization_seconds
    base$flip_asymptotic_seconds <- flip$asymptotic_seconds
    base$nuisance_stationarity_norm <- flip$nuisance_stationarity_norm
    base$mean_variance_relative_shift <- flip$mean_variance_relative_shift
    base$max_variance_relative_shift <- flip$max_variance_relative_shift
    base$min_variance_eigenvalue <- flip$min_variance_eigenvalue
    base$max_variance_condition <- flip$max_variance_condition
    base$sandwich_available <- flip$sandwich_available
    base$sandwich_condition <- flip$sandwich_condition
    base$score_eigen_mean <- mean(flip$eigenvalues)
    base$score_eigen_cv <- if (length(flip$eigenvalues) > 1L)
      stats::sd(flip$eigenvalues) / mean(flip$eigenvalues) else 0
    base$score_eigen_ratio <- max(flip$eigenvalues) / min(flip$eigenvalues)
  }

  nested_begin <- proc.time()[["elapsed"]]
  nested <- tryCatch(fmg_nested(
    fits$H1, fits$H0, data = sample$blocks, tests = nested_tests,
    A.method = "exact"), error = function(e) e)
  base$nested_seconds <- proc.time()[["elapsed"]] - nested_begin
  if (inherits(nested, "error")) {
    base$nested_error <- conditionMessage(nested)
  } else {
    stopifnot(all(nested$df == cell$df), identical(nested$label, nested_tests))
    base$nested_ok <- TRUE
    p[paste0("p_nested_", nested$label)] <- nested$p_value
  }
  base$total_seconds <- proc.time()[["elapsed"]] - total_begin
  cbind(base, as.data.frame(as.list(p)))
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  pop <- fmg_population(cell$p, cell$groups, cell$truth)
  # The DGP seed excludes truth, giving null/power common random numbers within
  # each p x G x distribution corner.
  dist_id <- match(cell$distribution, c("normal", "vm", "ig", "pl"))
  dgp_seed <- opts$seed_base + cell$p * 10000L + cell$groups * 1000L +
    dist_id * 100L
  sampler <- tryCatch(fmg_make_sampler(
    pop, cell$n_group, opts$reps, cell$distribution, dgp_seed),
    error = function(e) e)
  if (inherits(sampler, "error")) {
    rows <- lapply(seq_len(opts$reps), function(rep_id) {
      p <- empty_p()
      base <- data.frame(
        rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
        fit_error = paste0("generator: ", conditionMessage(sampler)),
        flip_error = "", nested_error = "", draw_seconds = NA_real_,
        fit_seconds = NA_real_, flip_seconds = NA_real_, nested_seconds = NA_real_,
        total_seconds = NA_real_, flip_setup_seconds = NA_real_,
        flip_score_seconds = NA_real_, flip_standardization_seconds = NA_real_,
        flip_asymptotic_seconds = NA_real_, nuisance_stationarity_norm = NA_real_,
        mean_variance_relative_shift = NA_real_,
        max_variance_relative_shift = NA_real_,
        min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
        sandwich_available = NA, sandwich_condition = NA_real_,
        score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
        score_eigen_ratio = NA_real_)
      cbind(base, as.data.frame(as.list(p)))
    })
    generator_setup_seconds <- NA_real_
  } else {
    specs <- get_specs(cell$p, cell$groups)
    rows <- lapply(seq_len(opts$reps), function(rep_id) tryCatch(
      one_rep(cell, rep_id, sampler, specs),
      error = function(e) {
        p <- empty_p()
        base <- data.frame(
          rep = rep_id, fit_ok = FALSE, flip_ok = FALSE, nested_ok = FALSE,
          fit_error = conditionMessage(e), flip_error = "", nested_error = "",
          draw_seconds = NA_real_, fit_seconds = NA_real_,
          flip_seconds = NA_real_, nested_seconds = NA_real_,
          total_seconds = NA_real_, flip_setup_seconds = NA_real_,
          flip_score_seconds = NA_real_, flip_standardization_seconds = NA_real_,
          flip_asymptotic_seconds = NA_real_, nuisance_stationarity_norm = NA_real_,
          mean_variance_relative_shift = NA_real_,
          max_variance_relative_shift = NA_real_,
          min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
          sandwich_available = NA, sandwich_condition = NA_real_,
          score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
          score_eigen_ratio = NA_real_)
        cbind(base, as.data.frame(as.list(p)))
      }))
    generator_setup_seconds <- sampler$setup_seconds
  }
  out <- do.call(rbind, rows)
  for (name in names(cell)) out[[name]] <- cell[[name]]
  out$generator_setup_seconds <- generator_setup_seconds
  out
}

wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  phat <- x / n; den <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / den
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores))
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(grid)), run_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), function(k) {
    cat(sprintf("  cell %d/%d\n", k, nrow(grid))); flush.console()
    run_cell(k)
  })
}
raw <- do.call(rbind, pieces)
write.csv(raw, file.path(results_dir, "replications.csv"), row.names = FALSE)
write.csv(grid, file.path(results_dir, "design.csv"), row.names = FALSE)

method_rows <- do.call(rbind, lapply(p_columns, function(column) {
  data.frame(raw[c("cell_id", "p", "groups", "n_group", "n_total", "df",
                   "distribution", "truth", "rep")],
             method = sub("^p_", "", column), p_value = raw[[column]],
             stringsAsFactors = FALSE)
}))
method_split <- split(method_rows, interaction(method_rows$cell_id,
                                                method_rows$method, drop = TRUE))
method_summary <- do.call(rbind, lapply(method_split, function(x) {
  valid <- is.finite(x$p_value); n <- sum(valid)
  rejected <- sum(x$p_value[valid] < .05)
  ci <- wilson(rejected, n)
  data.frame(x[1L, c("cell_id", "p", "groups", "n_group", "n_total", "df",
                     "distribution", "truth", "method")],
             n = n, rejected = rejected,
             rejection_rate = if (n) rejected / n else NA_real_,
             ci_lower = ci[[1L]], ci_upper = ci[[2L]])
}))
method_summary <- method_summary[order(method_summary$cell_id,
                                       method_summary$method), ]
row.names(method_summary) <- NULL
write.csv(method_summary, file.path(results_dir, "method_summary.csv"),
          row.names = FALSE)

paired_split <- split(raw, raw$cell_id)
paired_summary <- do.call(rbind, lapply(paired_split, function(x) {
  keep <- is.finite(x$p_flip_effective) & is.finite(x$p_flip_standardized)
  y <- x[keep, , drop = FALSE]; n <- nrow(y)
  effective <- y$p_flip_effective < .05
  standardized <- y$p_flip_standardized < .05
  delta <- as.numeric(standardized) - as.numeric(effective)
  cor_shift <- if (n >= 3L && stats::sd(abs(y$p_flip_standardized -
                                            y$p_flip_effective)) > 0 &&
                          stats::sd(y$mean_variance_relative_shift) > 0) {
    stats::cor(abs(y$p_flip_standardized - y$p_flip_effective),
               y$mean_variance_relative_shift)
  } else NA_real_
  data.frame(x[1L, c("cell_id", "p", "groups", "n_group", "n_total", "df",
                     "distribution", "truth")],
             n = n,
             rejection_effective = if (n) mean(effective) else NA_real_,
             rejection_standardized = if (n) mean(standardized) else NA_real_,
             rejection_delta = if (n) mean(delta) else NA_real_,
             paired_mcse = if (n > 1L) stats::sd(delta) / sqrt(n) else NA_real_,
             disagreement_rate = if (n) mean(effective != standardized) else NA_real_,
             changed_to_accept = if (n) mean(effective & !standardized) else NA_real_,
             changed_to_reject = if (n) mean(!effective & standardized) else NA_real_,
             mean_abs_delta_p = if (n) mean(abs(y$p_flip_standardized -
                                                  y$p_flip_effective)) else NA_real_,
             median_abs_delta_p = if (n) stats::median(abs(
               y$p_flip_standardized - y$p_flip_effective)) else NA_real_,
             median_variance_shift = if (n) stats::median(
               y$mean_variance_relative_shift) else NA_real_,
             median_max_condition = if (n) stats::median(
               y$max_variance_condition) else NA_real_,
             delta_shift_correlation = cor_shift)
}))
row.names(paired_summary) <- NULL
write.csv(paired_summary, file.path(results_dir, "paired_summary.csv"),
          row.names = FALSE)

power_rows <- method_rows[method_rows$truth == "power" &
                            is.finite(method_rows$p_value), , drop = FALSE]
matched_split <- split(power_rows, interaction(
  power_rows$p, power_rows$groups, power_rows$distribution,
  power_rows$method, drop = TRUE))
matched_power <- do.call(rbind, lapply(matched_split, function(x) {
  null <- method_rows[
    method_rows$truth == "null" & method_rows$p == x$p[[1L]] &
      method_rows$groups == x$groups[[1L]] &
      method_rows$distribution == x$distribution[[1L]] &
      method_rows$method == x$method[[1L]] & is.finite(method_rows$p_value),
    "p_value"]
  empirical_p <- vapply(x$p_value, function(pv)
    (1 + sum(null <= pv)) / (length(null) + 1), numeric(1L))
  rejected <- sum(empirical_p <= .05); n <- length(empirical_p)
  ci <- wilson(rejected, n)
  data.frame(x[1L, c("p", "groups", "n_group", "n_total", "df",
                     "distribution", "method")],
             n_null = length(null), n_power = n, rejected = rejected,
             matched_power = if (n) rejected / n else NA_real_,
             ci_lower = ci[[1L]], ci_upper = ci[[2L]])
}))
row.names(matched_power) <- NULL
write.csv(matched_power, file.path(results_dir, "matched_power.csv"),
          row.names = FALSE)

timing_columns <- c(
  generator_setup = "generator_setup_seconds", draw = "draw_seconds",
  fit = "fit_seconds", flip_total = "flip_seconds",
  flip_setup = "flip_setup_seconds", flip_score = "flip_score_seconds",
  flip_standardization = "flip_standardization_seconds",
  flip_asymptotic = "flip_asymptotic_seconds", nested_fmg = "nested_seconds",
  replication_total = "total_seconds")
timing_summary <- do.call(rbind, lapply(paired_split, function(x) {
  do.call(rbind, lapply(names(timing_columns), function(phase) {
    values <- x[[timing_columns[[phase]]]]
    if (phase == "generator_setup") values <- values[[1L]]
    values <- values[is.finite(values)]
    data.frame(x[1L, c("cell_id", "p", "groups", "n_group", "n_total", "df",
                       "distribution", "truth")],
               phase = phase, n = length(values),
               median_seconds = if (length(values)) stats::median(values) else NA_real_,
               p90_seconds = if (length(values)) unname(stats::quantile(
                 values, .9, type = 8)) else NA_real_)
  }))
}))
row.names(timing_summary) <- NULL
write.csv(timing_summary, file.path(results_dir, "timing_summary.csv"),
          row.names = FALSE)

failures <- raw[!raw$fit_ok | !raw$flip_ok | !raw$nested_ok,
                c("cell_id", "rep", "p", "groups", "distribution", "truth",
                  "fit_ok", "flip_ok", "nested_ok", "fit_error", "flip_error",
                  "nested_error")]
write.csv(failures, file.path(results_dir, "failures.csv"), row.names = FALSE)

methods <- data.frame(
  method = sub("^p_", "", p_columns),
  family = c(rep("flip", 3L), rep("score", 8L), rep("nested", 8L)),
  stringsAsFactors = FALSE)
write.csv(methods, file.path(results_dir, "methods.csv"), row.names = FALSE)
elapsed <- proc.time()[["elapsed"]] - wall_begin
metadata <- data.frame(
  key = c("mode", "cells", "reps", "flips", "cores", "seed_base",
          "elapsed_seconds", "magmaan_version", "R_version",
          "paper_doi", "osf"),
  value = c(opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$seed_base, elapsed, as.character(packageVersion("magmaan")),
            R.version.string, "10.3758/s13428-026-02968-4",
            "https://osf.io/h2y3n/"))
write.csv(metadata, file.path(results_dir, "metadata.csv"), row.names = FALSE)
cat(sprintf("wrote results to %s (%.1fs, %d failures)\n",
            results_dir, elapsed, nrow(failures)))
