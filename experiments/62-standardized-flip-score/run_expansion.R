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
  "Usage: Rscript run_expansion.R [--smoke|--probe|--power|--full] [options]\n\n",
  "Factorial calibration and power comparison for score flips and FMG.\n\n",
  "Options:\n",
  "  --smoke          Hardest cell, 4 reps, 39 flips (default).\n",
  "  --probe          162-cell null grid.\n",
  "  --power          540-cell sparse/dense power grid.\n",
  "  --stress         24-cell severe non-normal copula stress block.\n",
  "  --full           Null and power grids in one output.\n",
  "  --reps N         Replications per cell (null: 200; power: 150).\n",
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
  else if (a == "--power") opts$mode <- "power"
  else if (a == "--stress") opts$mode <- "stress"
  else if (a == "--full") opts$mode <- "full"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- switch(
  opts$mode, smoke = 4L, power = 150L, probe = 200L, full = 150L, stress = 200L)
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 39L else 499L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L)
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

specs <- setNames(lapply(c(1L, 4L, 8L), flip_expansion_specs), c("1", "4", "8"))
null_grid <- expand.grid(
  df = c(1L, 4L, 8L), n_total = c(60L, 100L, 200L),
  balance = c("1:1", "1:3"),
  heterogeneity = c("homogeneous", "scale", "geometry"),
  distribution = c("normal", "t5", "skew"), stringsAsFactors = FALSE)
null_grid$alternative <- "null"
null_grid$effect <- 0
power_grid <- expand.grid(
  df = c(1L, 4L, 8L), n_total = c(60L, 100L, 200L),
  balance = c("1:1", "1:3"),
  heterogeneity = c("homogeneous", "scale", "geometry"),
  distribution = c("normal", "t5", "skew"),
  alternative = c("sparse", "dense"), effect = c(.25, .50),
  stringsAsFactors = FALSE)
power_grid <- subset(power_grid, !(df == 1L & alternative == "dense"))
# Severe non-normal copula stress: same marginals (skew 3, excess kurtosis 21),
# two copulas (vm = Gaussian, pl = non-Gaussian). Homogeneous isolates the
# copula from group heterogeneity; geometry is the hardest information geometry.
# Kept as its own output so it does not reseed or unbalance the main null grid.
stress_grid <- expand.grid(
  df = c(1L, 8L), n_total = c(60L, 100L, 200L),
  balance = "1:1", heterogeneity = c("homogeneous", "geometry"),
  distribution = c("vm", "pl"), stringsAsFactors = FALSE)
stress_grid$alternative <- "null"
stress_grid$effect <- 0
grid <- switch(opts$mode,
  probe = null_grid,
  power = power_grid,
  stress = stress_grid,
  full = rbind(null_grid, power_grid),
  smoke = rbind(
    subset(null_grid, df == 8L & n_total == 60L & balance == "1:3" &
             heterogeneity == "geometry" & distribution == "skew"),
    subset(power_grid, df == 8L & n_total == 60L & balance == "1:3" &
             heterogeneity == "geometry" & distribution == "skew" &
             alternative == "dense" & effect == .50)))
grid$cell_id <- seq_len(nrow(grid))

output_prefix <- switch(opts$mode, probe = "expansion", power = "power",
                        stress = "stress", full = "combined", smoke = "smoke")
fmg_tests <- c("SB", "MV", "SS", "SF", "EBA2", "EBA4", "EBA6",
               "pEBA2", "pEBA4", "pEBA6", "PALL", "pOLS2", "ALL")
p_columns <- c(
  "p_basic", "p_effective", "p_standardized", "p_chisq",
  "p_mean_scaled", "p_mixture",
  "p_score_ss", "p_score_mv", "p_score_sf", "p_score_eba4",
  "p_score_peba4", "p_score_pols",
  "p_lr_unscaled", "p_lr_scaled",
  "p_lr_mv_adjusted", "p_lr_scaled_shifted", "p_lr_mixture",
  "p_fmg_sb", "p_fmg_mv", "p_fmg_ss", "p_fmg_sf", "p_fmg_eba2",
  "p_fmg_eba4", "p_fmg_eba6", "p_fmg_peba2", "p_fmg_peba4",
  "p_fmg_peba6", "p_fmg_pall", "p_fmg_pols", "p_fmg_all")

empty_rep <- function(rep_id, error) {
  out <- data.frame(rep = rep_id, ok = FALSE, error = error,
                    nested_error = NA_character_, fmg_error = NA_character_)
  for (name in p_columns) out[[name]] <- NA_real_
  cbind(out, data.frame(
  reject_effective = NA, reject_standardized = NA,
  changed_to_accept = NA, changed_to_reject = NA,
  abs_delta_p = NA_real_, nuisance_norm = NA_real_,
  mean_variance_relative_shift = NA_real_,
  max_variance_relative_shift = NA_real_,
  min_variance_eigenvalue = NA_real_, max_variance_condition = NA_real_,
  fit_seconds = NA_real_, flip_elapsed_seconds = NA_real_,
  setup_seconds = NA_real_, resampling_score_seconds = NA_real_,
  resampling_standardization_seconds = NA_real_,
  asymptotic_seconds = NA_real_, core_total_seconds = NA_real_,
  nested_seconds = NA_real_, fmg_seconds = NA_real_,
  fmg_duplicate_max_gap = NA_real_))
}

one_rep <- function(cell, rep_id) {
  seed <- opts$seed_base + cell$cell_id * 100000L + rep_id
  dat <- flip_expansion_draw_data(
    cell$n_total, cell$balance, cell$heterogeneity, cell$distribution, seed,
    cell$df, cell$alternative, cell$effect)
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

  # FMG spectrum transforms applied to the SCORE statistic's own eigenvalues.
  # score-SB/ALL/standard already exist as p_mean_scaled/p_mixture/p_chisq;
  # these add the SS/MV/SF/EBA/pEBA/pOLS members so the full FMG family runs on
  # the score base statistic, not only on the Satorra-2000 LR construction.
  score_fmg <- function(method, param = 4) tryCatch(
    magmaan:::infer_fmg_test(flip$statistic_effective, flip$df,
                             flip$eigenvalues, method = method,
                             param = param)$p_value,
    error = function(e) NA_real_)
  p_score_ss <- score_fmg("ss")
  p_score_mv <- score_fmg("mv")
  p_score_sf <- score_fmg("scaled_f")
  p_score_eba4 <- score_fmg("eba", 4)
  p_score_peba4 <- score_fmg("peba", 4)
  p_score_pols <- score_fmg("pols", 2)

  raw_blocks <- lapply(c("A", "B"), function(g)
    as.matrix(dat[dat$school == g, flip_expansion_ov, drop = FALSE]))
  nested_begin <- proc.time()[["elapsed"]]
  nested <- tryCatch(nestedTest(
    fit$configural, fit$restricted, data = raw_blocks,
    method = "satorra.2000", A.method = "exact"), error = function(e) e)
  nested_seconds <- proc.time()[["elapsed"]] - nested_begin
  nested_error <- if (inherits(nested, "error")) conditionMessage(nested) else ""

  fmg_begin <- proc.time()[["elapsed"]]
  fmg <- tryCatch(fmg_nested(
    fit$configural, fit$restricted, data = raw_blocks, tests = fmg_tests,
    A.method = "exact"), error = function(e) e)
  fmg_seconds <- proc.time()[["elapsed"]] - fmg_begin
  fmg_error <- if (inherits(fmg, "error")) conditionMessage(fmg) else ""

  p_nested <- rep(NA_real_, 5L)
  names(p_nested) <- c("p_lr_unscaled", "p_lr_scaled", "p_lr_mv_adjusted",
                       "p_lr_scaled_shifted", "p_lr_mixture")
  if (!inherits(nested, "error")) {
    p_nested[] <- c(nested$p_unscaled, nested$p_scaled, nested$p_adjusted,
                    nested$p_scaled_shifted, nested$p_mixture)
  }
  p_fmg <- setNames(rep(NA_real_, 13L), c(
    "p_fmg_sb", "p_fmg_mv", "p_fmg_ss", "p_fmg_sf", "p_fmg_eba2",
    "p_fmg_eba4", "p_fmg_eba6", "p_fmg_peba2", "p_fmg_peba4",
    "p_fmg_peba6", "p_fmg_pall", "p_fmg_pols", "p_fmg_all"))
  fmg_duplicate_max_gap <- NA_real_
  if (!inherits(fmg, "error")) {
    key <- sub("_ml$", "", fmg$label)
    key[key == "pols2"] <- "pols"
    hit <- match(sub("^p_fmg_", "", names(p_fmg)), key)
    p_fmg[!is.na(hit)] <- fmg$p_value[hit[!is.na(hit)]]
    if (!inherits(nested, "error")) {
      fmg_duplicate_max_gap <- max(abs(c(
        p_fmg[["p_fmg_sb"]] - nested$p_scaled,
        p_fmg[["p_fmg_mv"]] - nested$p_adjusted,
        p_fmg[["p_fmg_ss"]] - nested$p_scaled_shifted)))
    }
  }

  reject_effective <- flip$p_effective < .05
  reject_standardized <- flip$p_standardized < .05
  out <- data.frame(
    rep = rep_id, ok = TRUE, error = "", nested_error = nested_error,
    fmg_error = fmg_error,
    p_basic = flip$p_basic, p_effective = flip$p_effective,
    p_standardized = flip$p_standardized, p_chisq = flip$p_chisq,
    p_mean_scaled = flip$p_mean_scaled, p_mixture = flip$p_mixture,
    p_score_ss = p_score_ss, p_score_mv = p_score_mv,
    p_score_sf = p_score_sf, p_score_eba4 = p_score_eba4,
    p_score_peba4 = p_score_peba4, p_score_pols = p_score_pols,
    p_lr_unscaled = p_nested[["p_lr_unscaled"]],
    p_lr_scaled = p_nested[["p_lr_scaled"]],
    p_lr_mv_adjusted = p_nested[["p_lr_mv_adjusted"]],
    p_lr_scaled_shifted = p_nested[["p_lr_scaled_shifted"]],
    p_lr_mixture = p_nested[["p_lr_mixture"]],
    p_fmg_sb = p_fmg[["p_fmg_sb"]], p_fmg_mv = p_fmg[["p_fmg_mv"]],
    p_fmg_ss = p_fmg[["p_fmg_ss"]], p_fmg_sf = p_fmg[["p_fmg_sf"]],
    p_fmg_eba2 = p_fmg[["p_fmg_eba2"]],
    p_fmg_eba4 = p_fmg[["p_fmg_eba4"]],
    p_fmg_eba6 = p_fmg[["p_fmg_eba6"]],
    p_fmg_peba2 = p_fmg[["p_fmg_peba2"]],
    p_fmg_peba4 = p_fmg[["p_fmg_peba4"]],
    p_fmg_peba6 = p_fmg[["p_fmg_peba6"]],
    p_fmg_pall = p_fmg[["p_fmg_pall"]],
    p_fmg_pols = p_fmg[["p_fmg_pols"]],
    p_fmg_all = p_fmg[["p_fmg_all"]],
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
    core_total_seconds = flip$total_seconds,
    nested_seconds = nested_seconds, fmg_seconds = fmg_seconds,
    fmg_duplicate_max_gap = fmg_duplicate_max_gap)
  stopifnot(identical(p_columns, intersect(p_columns, names(out))))
  out
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  rows <- lapply(seq_len(opts$reps), function(r) tryCatch(
    one_rep(cell, r), error = function(e) empty_rep(r, conditionMessage(e))))
  out <- do.call(rbind, rows)
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
write.csv(raw, file.path(results_dir, paste0(output_prefix, "_replications.csv")),
          row.names = FALSE)

summarize_cell <- function(x) {
  ok <- x[x$ok, , drop = FALSE]
  safe_mean <- function(x) if (any(is.finite(x))) mean(x, na.rm = TRUE) else NA_real_
  safe_max <- function(x) if (any(is.finite(x))) max(x, na.rm = TRUE) else NA_real_
  out <- list(
    reps = nrow(x), reps_ok = nrow(ok), failure_rate = mean(!x$ok),
    nested_failure_rate = mean(nzchar(ok$nested_error)),
    fmg_failure_rate = mean(nzchar(ok$fmg_error)),
    changed_to_accept = mean(ok$changed_to_accept),
    changed_to_reject = mean(ok$changed_to_reject),
    mean_abs_delta_p = mean(ok$abs_delta_p),
    mean_variance_relative_shift = mean(ok$mean_variance_relative_shift),
    max_variance_relative_shift = safe_max(ok$max_variance_relative_shift),
    median_variance_condition = median(ok$max_variance_condition),
    median_fit_ms = 1000 * median(ok$fit_seconds),
    median_flip_ms = 1000 * median(ok$flip_elapsed_seconds),
    median_setup_ms = 1000 * median(ok$setup_seconds),
    median_resampling_score_ms = 1000 * median(ok$resampling_score_seconds),
    median_standardization_ms =
      1000 * median(ok$resampling_standardization_seconds),
    median_asymptotic_ms = 1000 * median(ok$asymptotic_seconds),
    median_nested_ms = 1000 * median(ok$nested_seconds),
    median_fmg_ms = 1000 * median(ok$fmg_seconds),
    max_fmg_duplicate_gap = safe_max(ok$fmg_duplicate_max_gap))
  for (name in p_columns) {
    suffix <- sub("^p_", "", name)
    out[[paste0("rejection_", suffix)]] <- safe_mean(ok[[name]] < .05)
    out[[paste0("available_", suffix)]] <- sum(is.finite(ok[[name]]))
  }
  as.data.frame(out)
}
split_raw <- split(raw, raw$cell_id)
summary <- do.call(rbind, lapply(split_raw, summarize_cell))
summary$cell_id <- as.integer(names(split_raw))
summary <- merge(grid, summary, by = "cell_id", sort = TRUE)
write.csv(summary, file.path(results_dir, paste0(output_prefix, "_summary.csv")),
          row.names = FALSE)

metadata <- data.frame(
  key = c("mode", "cells", "reps", "flips", "cores", "seed_base",
          "elapsed_seconds", "magmaan_version", "R_version"),
  value = c(opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$seed_base, proc.time()[["elapsed"]] - t0,
            as.character(packageVersion("magmaan")), R.version.string))
write.csv(metadata, file.path(results_dir, paste0(output_prefix, "_metadata.csv")),
          row.names = FALSE)
cat(sprintf("wrote %s results to %s (%.1fs)\n", output_prefix, results_dir,
            proc.time()[["elapsed"]] - t0))
