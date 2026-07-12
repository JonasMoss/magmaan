#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "expansion_design.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_expansion.R [--smoke|--probe] [options]\n\n",
  "Factorial calibration design for when flip standardization matters.\n\n",
  "Options:\n",
  "  --smoke          Hardest cell, 4 reps, 39 flips (default).\n",
  "  --probe          162-cell null grid.\n",
  "  --reps N         Replications per cell (probe default: 200).\n",
  "  --flips N        Random flips per replication (probe default: 499).\n",
  "  --cores N        Parallel cell workers.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(mode = "smoke", reps = NULL, flips = NULL,
             cores = max(1L, parallel::detectCores() - 2L),
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
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 4L else 200L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 39L else 499L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L)
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

specs <- setNames(lapply(c(1L, 4L, 8L), flip_expansion_specs), c("1", "4", "8"))
grid <- expand.grid(
  df = c(1L, 4L, 8L), n_total = c(60L, 100L, 200L),
  balance = c("1:1", "1:3"),
  heterogeneity = c("homogeneous", "scale", "geometry"),
  distribution = c("normal", "t5", "skew"), stringsAsFactors = FALSE)
if (opts$mode == "smoke") {
  grid <- subset(grid, df == 8L & n_total == 60L & balance == "1:3" &
                   heterogeneity == "geometry" & distribution == "skew")
}
grid$cell_id <- seq_len(nrow(grid))

empty_rep <- function(rep_id, error) data.frame(
  rep = rep_id, ok = FALSE, error = error,
  p_basic = NA_real_, p_effective = NA_real_, p_standardized = NA_real_,
  p_chisq = NA_real_, p_mean_scaled = NA_real_, p_mixture = NA_real_,
  reject_effective = NA, reject_standardized = NA,
  changed_to_accept = NA, changed_to_reject = NA,
  abs_delta_p = NA_real_, nuisance_norm = NA_real_,
  mean_variance_relative_shift = NA_real_,
  max_variance_relative_shift = NA_real_,
  min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
  fit_seconds = NA_real_, flip_elapsed_seconds = NA_real_,
  setup_seconds = NA_real_, resampling_score_seconds = NA_real_,
  resampling_standardization_seconds = NA_real_,
  asymptotic_seconds = NA_real_, core_total_seconds = NA_real_)

one_rep <- function(cell, rep_id) {
  seed <- opts$seed_base + cell$cell_id * 100000L + rep_id
  dat <- flip_expansion_draw_data(
    cell$n_total, cell$balance, cell$heterogeneity, cell$distribution, seed)
  pair <- specs[[as.character(cell$df)]]
  fit_begin <- proc.time()[["elapsed"]]
  fit <- tryCatch(lapply(
    pair, magmaan, data = dat, estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback"),
                  error = function(e) e)
  fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fit, "error")) return(empty_rep(rep_id, conditionMessage(fit)))

  flip_begin <- proc.time()[["elapsed"]]
  flip <- tryCatch(score_flip_test(
    fit$configural, fit$restricted, dat, n_flips = opts$flips,
    seed = opts$seed_base + cell$cell_id * 1000000L + rep_id),
    error = function(e) e)
  flip_elapsed <- proc.time()[["elapsed"]] - flip_begin
  if (inherits(flip, "error")) return(empty_rep(rep_id, conditionMessage(flip)))
  stopifnot(flip$df == cell$df)

  reject_effective <- flip$p_effective < .05
  reject_standardized <- flip$p_standardized < .05
  data.frame(
    rep = rep_id, ok = TRUE, error = "",
    p_basic = flip$p_basic, p_effective = flip$p_effective,
    p_standardized = flip$p_standardized, p_chisq = flip$p_chisq,
    p_mean_scaled = flip$p_mean_scaled, p_mixture = flip$p_mixture,
    reject_effective = reject_effective,
    reject_standardized = reject_standardized,
    changed_to_accept = reject_effective && !reject_standardized,
    changed_to_reject = !reject_effective && reject_standardized,
    abs_delta_p = abs(flip$p_standardized - flip$p_effective),
    nuisance_norm = flip$nuisance_stationarity_norm,
    mean_variance_relative_shift = flip$mean_variance_relative_shift,
    max_variance_relative_shift = flip$max_variance_relative_shift,
    min_variance_eigenvalue = flip$min_variance_eigenvalue,
    max_variance_condition = flip$max_variance_condition,
    fit_seconds = fit_seconds, flip_elapsed_seconds = flip_elapsed,
    setup_seconds = flip$setup_seconds,
    resampling_score_seconds = flip$resampling_score_seconds,
    resampling_standardization_seconds =
      flip$resampling_standardization_seconds,
    asymptotic_seconds = flip$asymptotic_seconds,
    core_total_seconds = flip$total_seconds)
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  out <- do.call(rbind, lapply(seq_len(opts$reps), function(r) one_rep(cell, r)))
  for (name in names(cell)) out[[name]] <- cell[[name]]
  out
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores))
t0 <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L && nrow(grid) > 1L) {
  pieces <- parallel::mclapply(seq_len(nrow(grid)), run_cell,
                               mc.cores = min(opts$cores, nrow(grid)),
                               mc.preschedule = TRUE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), function(k) {
    cat(sprintf("  cell %d/%d\n", k, nrow(grid))); flush.console()
    run_cell(k)
  })
}
raw <- do.call(rbind, pieces)
write.csv(raw, file.path(results_dir, "expansion_replications.csv"),
          row.names = FALSE)

summarize_cell <- function(x) {
  ok <- x[x$ok, , drop = FALSE]
  data.frame(
    reps = nrow(x), reps_ok = nrow(ok), failure_rate = mean(!x$ok),
    rejection_basic = mean(ok$p_basic < .05),
    rejection_effective = mean(ok$reject_effective),
    rejection_standardized = mean(ok$reject_standardized),
    rejection_chisq = mean(ok$p_chisq < .05),
    rejection_mean_scaled = mean(ok$p_mean_scaled < .05),
    rejection_mixture = mean(ok$p_mixture < .05),
    changed_to_accept = mean(ok$changed_to_accept),
    changed_to_reject = mean(ok$changed_to_reject),
    mean_abs_delta_p = mean(ok$abs_delta_p),
    mean_variance_relative_shift = mean(ok$mean_variance_relative_shift),
    max_variance_relative_shift = max(ok$max_variance_relative_shift),
    median_variance_condition = median(ok$max_variance_condition),
    median_fit_ms = 1000 * median(ok$fit_seconds),
    median_flip_ms = 1000 * median(ok$flip_elapsed_seconds),
    median_setup_ms = 1000 * median(ok$setup_seconds),
    median_resampling_score_ms = 1000 * median(ok$resampling_score_seconds),
    median_standardization_ms =
      1000 * median(ok$resampling_standardization_seconds),
    median_asymptotic_ms = 1000 * median(ok$asymptotic_seconds))
}
split_raw <- split(raw, raw$cell_id)
summary <- do.call(rbind, lapply(split_raw, summarize_cell))
summary$cell_id <- as.integer(names(split_raw))
summary <- merge(grid, summary, by = "cell_id", sort = TRUE)
write.csv(summary, file.path(results_dir, "expansion_summary.csv"),
          row.names = FALSE)

metadata <- data.frame(
  key = c("mode", "cells", "reps", "flips", "cores", "seed_base",
          "elapsed_seconds", "magmaan_version", "R_version"),
  value = c(opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$seed_base, proc.time()[["elapsed"]] - t0,
            as.character(packageVersion("magmaan")), R.version.string))
write.csv(metadata, file.path(results_dir, "expansion_metadata.csv"),
          row.names = FALSE)
cat(sprintf("wrote expansion results to %s (%.1fs)\n", results_dir,
            proc.time()[["elapsed"]] - t0))
