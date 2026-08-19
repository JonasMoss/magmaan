#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "pilot.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot] [options]\n\n",
  "FIML global-GOF stress grid.\n\n",
  "  --smoke          70 cells, 1 replication, 19 flips (default).\n",
  "  --pilot          70 cells, 500 replications, 199 flips.\n",
  "  --reps N         Override replications per cell.\n",
  "  --flips N        Override multiplier draws per replication.\n",
  "  --cores N        Parallel cell workers.\n",
  "  --n N            Sample size (default 120).\n",
  "  --p P            Number of variables (default 5).\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(
  mode = "smoke", reps = NULL, flips = NULL,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  n = 120L, p = 5L, seed_base = 20260818L, results_dir = NULL
)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
  else if (arg == "--smoke") opts$mode <- "smoke"
  else if (arg == "--pilot") opts$mode <- "pilot"
  else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--flips") opts$flips <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(take())
  else if (arg == "--p") opts$p <- as.integer(take())
  else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 1L else 500L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 19L else 199L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L,
          opts$n >= 40L, opts$p >= 3L, opts$p <= 8L, opts$seed_base >= 0L)

results <- opts$results_dir %||%
  file.path(script_dir, "results", opts$mode)
dir.create(results, recursive = TRUE, showWarnings = FALSE)
grid <- pilot_design(opts$n, opts$p)
specs <- pilot_specs(opts$p)
stopifnot(max(specs$H1$partable$free) == max(specs$H0$partable$free),
          all(grid$df == ncol(pilot_pairs(opts$p))))

run_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf(
    "  cell %d: %s, %s, %s",
    cell$cell_id, cell$distribution, cell$missingness, cell$truth))
  rows <- lapply(seq_len(opts$reps), function(rep_id) {
    out <- tryCatch(
      pilot_one_rep(cell, rep_id, specs, opts$flips, opts$seed_base),
      error = function(e) {
        p_values <- pilot_empty_p()
        base <- data.frame(
          rep = rep_id, fit_h0_ok = FALSE, lrt_ok = FALSE,
          score_ok = FALSE, fit_h1_ok = FALSE, wald_ok = FALSE,
          error_h0 = paste0("replication: ", conditionMessage(e)),
          error_lrt = "", error_score = "", error_h1 = "", error_wald = "",
          realized_missing = NA_real_, realized_missing_eligible = NA_real_,
          complete_r12 = NA_real_, observed_r12 = NA_real_,
          h0_seconds = NA_real_,
          lrt_seconds = NA_real_, score_seconds = NA_real_,
          h1_seconds = NA_real_, wald_seconds = NA_real_,
          total_seconds = NA_real_, score_df = NA_integer_,
          score_eigen_min = NA_real_, score_eigen_mean = NA_real_,
          score_eigen_cv = NA_real_, score_sandwich_condition = NA_real_,
          stringsAsFactors = FALSE)
        cbind(base, as.data.frame(as.list(p_values)))
      })
    for (name in names(cell)) out[[name]] <- cell[[name]]
    out
  })
  do.call(rbind, rows)
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d n=%d p=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$n, opts$p))
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(grid)), run_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), run_cell)
}
raw <- do.call(rbind, pieces)
row.names(raw) <- NULL
wall_seconds <- proc.time()[["elapsed"]] - wall_begin

summaries <- pilot_method_summary(raw)
write_csv(grid, file.path(results, "design.csv"))
write_csv(raw, file.path(results, "replications.csv"))
write_csv(summaries$long, file.path(results, "method_pvalues.csv"))
write_csv(summaries$summary, file.path(results, "method_summary.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  mode = opts$mode, reps = opts$reps, flips = opts$flips,
  cores = opts$cores, n = opts$n, p = opts$p,
  cells = nrow(grid), scenarios = length(unique(grid$scenario_id)),
  seed_base = opts$seed_base, wall_seconds = wall_seconds,
  replication_failures = sum(!raw$fit_h0_ok),
  lrt_failures = sum(raw$fit_h0_ok & !raw$lrt_ok),
  score_failures = sum(raw$fit_h0_ok & !raw$score_ok),
  wald_failures = sum(raw$fit_h1_ok & !raw$wald_ok)
), packages = "magmaan")

pooled <- aggregate(
  cbind(valid, rejected) ~ truth + method,
  data = summaries$summary, FUN = sum)
pooled$rejection_rate <- pooled$rejected / pooled$valid
pooled <- pooled[order(pooled$truth, pooled$method), ]
write_csv(pooled, file.path(results, "pooled_summary.csv"))

region_summary <- aggregate(
  cbind(valid, rejected) ~ analysis_region + truth + method,
  data = summaries$summary, FUN = sum)
region_summary$rejection_rate <-
  region_summary$rejected / region_summary$valid
region_summary <- region_summary[
  order(region_summary$analysis_region, region_summary$truth,
        region_summary$method), ]
write_csv(region_summary, file.path(results, "region_summary.csv"))

diagnostics <- aggregate(
  cbind(realized_missing, realized_missing_eligible, complete_r12,
        observed_r12, fit_h0_ok, lrt_ok, score_ok, fit_h1_ok, wald_ok) ~
    scenario_id + distribution + missingness + analysis_region + truth,
  data = raw, FUN = function(x) mean(as.numeric(x), na.rm = TRUE))
write_csv(diagnostics, file.path(results, "design_diagnostics.csv"))

# Diagnostic size-adjusted power: for each distribution/missingness/method,
# use the empirical fifth percentile of its null p-values as the rejection
# cutoff in the matched power cell. These cutoffs are unavailable in practice;
# they only compare detection after removing each method's size distortion.
key_columns <- c(
  "scenario_id", "distribution", "distribution_label", "missingness",
  "missingness_label", "analysis_region", "null_contract", "method")
keys <- unique(summaries$long[
  summaries$long$truth == "null", key_columns])
size_adjusted <- do.call(rbind, lapply(seq_len(nrow(keys)), function(index) {
  key <- keys[index, ]
  take <- function(truth) {
    x <- summaries$long[
      summaries$long$truth == truth &
        summaries$long$scenario_id == key$scenario_id &
        summaries$long$method == key$method,
      "p_value"]
    x[is.finite(x)]
  }
  null_p <- take("null")
  power_p <- take("power")
  cutoff <- if (length(null_p)) {
    unname(stats::quantile(null_p, 0.05, type = 8))
  } else NA_real_
  data.frame(
    key, null_n = length(null_p), power_n = length(power_p), cutoff = cutoff,
    empirical_size = if (length(null_p)) mean(null_p <= cutoff) else NA_real_,
    raw_power = if (length(power_p)) mean(power_p <= 0.05) else NA_real_,
    size_adjusted_power = if (length(power_p) && is.finite(cutoff))
      mean(power_p <= cutoff) else NA_real_,
    stringsAsFactors = FALSE)
}))
write_csv(size_adjusted, file.path(results, "size_adjusted_power.csv"))
size_adjusted_pooled <- aggregate(
  size_adjusted_power ~ method, size_adjusted, mean)
size_adjusted_pooled <- size_adjusted_pooled[
  order(-size_adjusted_pooled$size_adjusted_power), ]
write_csv(size_adjusted_pooled,
          file.path(results, "size_adjusted_power_pooled.csv"))

size_adjusted_region <- aggregate(
  cbind(raw_power, size_adjusted_power) ~ analysis_region + method,
  size_adjusted, mean)
write_csv(size_adjusted_region,
          file.path(results, "size_adjusted_power_by_region.csv"))

primary_regions <- c("Gaussian reference", "finite-moment robustness")
scorecard <- do.call(rbind, lapply(unique(summaries$summary$method),
  function(method) {
    null <- summaries$summary[
      summaries$summary$truth == "null" &
        summaries$summary$method == method, ]
    primary <- null[null$analysis_region %in% primary_regions, ]
    detection <- size_adjusted[
      size_adjusted$method == method &
        size_adjusted$analysis_region %in% primary_regions, ]
    data.frame(
      method = method,
      primary_null_cells = nrow(primary),
      primary_mean_abs_size_error = mean(abs(primary$rejection_rate - 0.05),
                                         na.rm = TRUE),
      primary_worst_abs_size_error = max(abs(primary$rejection_rate - 0.05),
                                         na.rm = TRUE),
      primary_min_size = min(primary$rejection_rate, na.rm = TRUE),
      primary_max_size = max(primary$rejection_rate, na.rm = TRUE),
      primary_near_nominal_fraction = mean(
        primary$rejection_rate >= 0.025 & primary$rejection_rate <= 0.075,
        na.rm = TRUE),
      primary_raw_power = mean(detection$raw_power, na.rm = TRUE),
      primary_size_adjusted_power = mean(
        detection$size_adjusted_power, na.rm = TRUE),
      all_null_mean_abs_size_error = mean(abs(null$rejection_rate - 0.05),
                                          na.rm = TRUE),
      stringsAsFactors = FALSE)
  }))
scorecard <- scorecard[order(
  scorecard$primary_mean_abs_size_error,
  -scorecard$primary_size_adjusted_power), ]
write_csv(scorecard, file.path(results, "method_scorecard.csv"))

cat(sprintf("wall_seconds=%.1f failures=%d/%d\n", wall_seconds,
            sum(!raw$fit_h0_ok), nrow(raw)))
print(pooled[, c("truth", "method", "valid", "rejection_rate")],
      row.names = FALSE, digits = 3)
cat("\nMean size-adjusted power over distribution/missingness cells:\n")
print(size_adjusted_pooled, row.names = FALSE, digits = 3)
cat("\nPrimary-region scorecard (ordered by calibration):\n")
print(scorecard, row.names = FALSE, digits = 3)
