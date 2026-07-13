#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("calibrate_power.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "design.R"))
source(experiment_path("R", "engine.R"))

usage <- function() cat(
  "Usage: Rscript calibrate_power.R [options]\n\n",
  "Calibrate the centered published G=8 loading deviations to approximately\n",
  "50% effective-flip power at n_avg=100, homogeneous balanced normal data.\n\n",
  "  --smoke          2 reps, 19 flips, one bisection step.\n",
  "  --reps N         Default 200.\n",
  "  --flips N        Default 199.\n",
  "  --steps N        Maximum bisection steps; default 6.\n",
  "  --cores N        Parallel replications.\n",
  "  --target X       Target rejection probability; default .50.\n",
  "  --seed-base N    Default 20260713.\n",
  "  --output P       Calibration CSV path.\n", sep = "")

opts <- list(smoke = FALSE, reps = 200L, flips = 199L, steps = 6L,
             cores = max(1L, parallel::detectCores() - 1L), target = .5,
             seed_base = 20260713,
             output = experiment_path("results", "calibration",
                                      "power_calibration.csv"))
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
  else if (a == "--smoke") opts$smoke <- TRUE
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--steps") opts$steps <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--target") opts$target <- as.numeric(take())
  else if (a == "--seed-base") opts$seed_base <- as.numeric(take())
  else if (a == "--output") opts$output <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (opts$smoke) {
  opts$reps <- 2L
  opts$flips <- 19L
  opts$steps <- 1L
}
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$steps > 0L,
          opts$cores > 0L, opts$target > 0, opts$target < 1)
dir.create(dirname(opts$output), recursive = TRUE, showWarnings = FALSE)
history_path <- sub("\\.csv$", "_history.csv", opts$output)
code_files <- c(experiment_path("calibrate_power.R"),
                experiment_path("R", "design.R"),
                experiment_path("R", "engine.R"))
code_hash <- paste(unname(tools::md5sum(code_files)), collapse = ":")

calibration_rep <- function(p, multiplier, rep_id, sampler, specs) {
  seed <- opts$seed_base + p * 1000003 + rep_id * 1009
  sample <- frontier_draw_replication(sampler, rep(100L, frontier_groups), seed)
  fits <- tryCatch(lapply(specs, function(spec) magmaan::magmaan(
    spec, sample$data, estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none")),
    error = function(e) e)
  if (inherits(fits, "error") ||
      !all(vapply(fits, function(x) isTRUE(x$converged), logical(1)))) {
    return(NA_real_)
  }
  names(fits) <- names(specs)
  flip <- tryCatch(magmaan::score_flip_test(
    fits$H1, fits$H0, sample$blocks, n_flips = opts$flips,
    seed = seed + 700000001), error = function(e) e)
  if (inherits(flip, "error") || flip$df != 28L) NA_real_ else flip$p_effective
}

history <- if (file.exists(history_path))
  read.csv(history_path, stringsAsFactors = FALSE) else data.frame()
if (nrow(history) && !"code_hash" %in% names(history)) history$code_hash <- ""

estimate_power <- function(p, multiplier) {
  previous <- if (nrow(history)) history[
    history$p == p & abs(history$multiplier - multiplier) < 1e-12 &
      history$reps == opts$reps & history$flips == opts$flips &
      history$code_hash == code_hash, ] else data.frame()
  if (nrow(previous)) return(previous$power[[nrow(previous)]])
  pop <- frontier_population(p, "homogeneous", "power", multiplier)
  sampler <- frontier_calibrate_sampler(pop, "normal")
  specs <- frontier_model_specs(p)
  worker <- function(rep_id) tryCatch(
    calibration_rep(p, multiplier, rep_id, sampler, specs),
    error = function(e) NA_real_)
  pvalues <- if (.Platform$OS.type != "windows" && opts$cores > 1L) {
    parallel::mclapply(seq_len(opts$reps), worker,
                       mc.cores = min(opts$cores, opts$reps),
                       mc.preschedule = TRUE, mc.set.seed = FALSE)
  } else lapply(seq_len(opts$reps), worker)
  pvalues <- unlist(pvalues, use.names = FALSE)
  valid <- is.finite(pvalues)
  power <- if (any(valid)) mean(pvalues[valid] <= .05) else NA_real_
  ci <- frontier_wilson(sum(pvalues[valid] <= .05), sum(valid))
  row <- data.frame(
    p = p, multiplier = multiplier, reps = opts$reps, flips = opts$flips,
    n_valid = sum(valid), power = power, ci_lower = ci[[1L]],
    ci_upper = ci[[2L]], target = opts$target, code_hash = code_hash,
    stringsAsFactors = FALSE)
  history <<- rbind(history, row)
  write_csv(history, history_path)
  message(sprintf("p=%d multiplier=%.5g power=%.3f (%d/%d valid)",
                  p, multiplier, power, sum(valid), opts$reps))
  power
}

calibrate_one <- function(p) {
  lo <- 0
  hi <- 1
  p_lo <- estimate_power(p, lo)
  p_hi <- estimate_power(p, hi)
  while (is.finite(p_hi) && p_hi < opts$target && hi < 16) {
    lo <- hi
    p_lo <- p_hi
    hi <- hi * 2
    p_hi <- estimate_power(p, hi)
  }
  for (step in seq_len(opts$steps)) {
    mid <- (lo + hi) / 2
    p_mid <- estimate_power(p, mid)
    if (!is.finite(p_mid)) break
    if (p_mid < opts$target) {
      lo <- mid
      p_lo <- p_mid
    } else {
      hi <- mid
      p_hi <- p_mid
    }
  }
  candidates <- history[history$p == p & history$reps == opts$reps &
                          history$flips == opts$flips &
                          history$code_hash == code_hash &
                          is.finite(history$power), ]
  best <- candidates[which.min(abs(candidates$power - opts$target)), ]
  data.frame(p = p, multiplier = best$multiplier, estimated_power = best$power,
             n_valid = best$n_valid, reps = opts$reps, flips = opts$flips,
             target = opts$target, seed_base = opts$seed_base)
}

frontier_validate_design()
result <- do.call(rbind, lapply(c(5L, 20L), calibrate_one))
write_csv(result, opts$output)
write_metadata(sub("\\.csv$", "_metadata.csv", opts$output), list(
  design = "G8 n_avg100 homogeneous balanced normal",
  tested_items = "x2:x5", df = 28L, reps = opts$reps, flips = opts$flips,
  target = opts$target, seed_base = opts$seed_base,
  selection = "closest evaluated effective-flip power"), packages = "magmaan")
cat("wrote ", opts$output, "\n", sep = "")
