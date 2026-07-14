#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "..", "_support", "R", "missingness.R"))
source(file.path(script_dir, "R", "design.R"))
source(file.path(script_dir, "R", "engine.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--probe] [options]\n\n",
  "Focused FIML nested score-flip probe using the FIML-FMG paper DGP.\n\n",
  "  --smoke          Normal-null edge rates, 2 reps, 19 flips (default).\n",
  "  --probe          Full 36-cell grid, 100 reps, 199 flips.\n",
  "  --reps N         Override replications per cell.\n",
  "  --flips N        Override random flips per replication.\n",
  "  --cores N        Parallel distribution/truth workers.\n",
  "  --n-values CSV   Group-1 sample sizes (default 50,100,200).\n",
  "  --rates CSV      MCAR rates (default 0,0.15,0.30).\n",
  "  --dists CSV      normal,pl.\n",
  "  --truths CSV     null,power.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(
  mode = "smoke", reps = NULL, flips = NULL,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  n_values = c(50L, 100L, 200L), rates = c(0, 0.15, 0.30),
  distributions = c("normal", "pl"), truths = c("null", "power"),
  seed_base = 20260714L, results_dir = NULL)
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
csv_chr <- function(x) strsplit(x, ",", fixed = TRUE)[[1L]]
csv_num <- function(x) as.numeric(csv_chr(x))
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--probe") opts$mode <- "probe"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--n-values") opts$n_values <- as.integer(csv_num(take()))
  else if (a == "--rates") opts$rates <- csv_num(take())
  else if (a == "--dists") opts$distributions <- csv_chr(take())
  else if (a == "--truths") opts$truths <- csv_chr(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- if (opts$mode == "smoke") 2L else 100L
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 19L else 199L
stopifnot(opts$reps > 0L, opts$flips > 0L, opts$cores > 0L,
          all(opts$n_values > 0L), all(opts$rates >= 0 & opts$rates < 1),
          all(opts$distributions %in% c("normal", "pl")),
          all(opts$truths %in% c("null", "power")))

results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
fiml_flip_validate_design()
grid <- fiml_flip_grid(opts$n_values, opts$rates, opts$distributions,
                       opts$truths)
if (opts$mode == "smoke") {
  grid <- grid[grid$distribution == "normal" & grid$truth == "null" &
                 grid$n_group1 == min(grid$n_group1) &
                 grid$missing_rate %in% range(grid$missing_rate), , drop = FALSE]
}
grid$cell_id <- seq_len(nrow(grid))
specs <- fiml_flip_specs()

task_key <- paste(grid$distribution, grid$truth, sep = "|")
tasks <- split(seq_len(nrow(grid)), task_key)
run_task <- function(indices) {
  first <- grid[indices[[1L]], , drop = FALSE]
  distribution <- first$distribution[[1L]]
  truth <- first$truth[[1L]]
  message(sprintf("  task %s/%s (%d cells)", distribution, truth,
                  length(indices)))
  pop <- fiml_flip_population(truth)
  task_seed <- opts$seed_base + match(distribution, c("normal", "pl")) * 100003L +
    match(truth, c("null", "power")) * 10000019L
  sampler <- tryCatch(fiml_flip_make_sampler(
    pop, distribution, unique(grid$n_group1[indices]), opts$reps, task_seed),
    error = function(e) e)
  rows <- list(); cursor <- 0L
  for (index in indices) {
    cell <- as.list(grid[index, , drop = FALSE])
    for (rep_id in seq_len(opts$reps)) {
      cursor <- cursor + 1L
      if (inherits(sampler, "error")) {
        result <- fiml_flip_empty_replication(
          rep_id, paste0("generator: ", conditionMessage(sampler)))
      } else {
        complete_data <- sampler$draw(cell$n_group1, rep_id)
        result <- tryCatch(fiml_flip_one_rep(
          cell, rep_id, complete_data, specs, opts$flips, opts$seed_base),
          error = function(e) fiml_flip_empty_replication(
            rep_id, paste0("replication: ", conditionMessage(e))))
      }
      for (name in names(cell)) result[[name]] <- cell[[name]]
      result$generator_setup_seconds <- if (inherits(sampler, "error"))
        NA_real_ else sampler$setup_seconds
      result$unique_group_calibrations <- if (inherits(sampler, "error"))
        NA_integer_ else sampler$unique_group_calibrations
      rows[[cursor]] <- result
    }
  }
  do.call(rbind, rows)
}

cat(sprintf("mode=%s cells=%d tasks=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), length(tasks), opts$reps, opts$flips,
            opts$cores))
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L && length(tasks) > 1L) {
  pieces <- parallel::mclapply(
    tasks, run_task, mc.cores = min(opts$cores, length(tasks)),
    mc.preschedule = FALSE)
} else {
  pieces <- lapply(tasks, run_task)
}
raw <- do.call(rbind, pieces)
row.names(raw) <- NULL
wall_seconds <- proc.time()[["elapsed"]] - wall_begin

write.csv(raw, file.path(results_dir, "replications.csv"), row.names = FALSE)
write.csv(grid, file.path(results_dir, "design.csv"), row.names = FALSE)

method_rows <- do.call(rbind, lapply(fiml_flip_p_columns, function(column) {
  data.frame(raw[c("cell_id", "distribution", "truth", "n_group1",
                   "n_group2", "n_total", "missing_rate", "df", "rep")],
             method = sub("^p_", "", column), p_value = raw[[column]],
             stringsAsFactors = FALSE)
}))
write.csv(method_rows, file.path(results_dir, "methods.csv"), row.names = FALSE)

method_split <- split(method_rows, interaction(
  method_rows$cell_id, method_rows$method, drop = TRUE))
method_summary <- do.call(rbind, lapply(method_split, function(x) {
  valid <- is.finite(x$p_value); n <- sum(valid)
  rejected <- sum(x$p_value[valid] < .05)
  ci <- fiml_flip_wilson(rejected, n)
  data.frame(x[1L, c("cell_id", "distribution", "truth", "n_group1",
                     "n_group2", "n_total", "missing_rate", "df", "method")],
             n = n, rejected = rejected,
             rejection_rate = if (n) rejected / n else NA_real_,
             ci_lower = ci[[1L]], ci_upper = ci[[2L]])
}))
method_summary <- method_summary[order(method_summary$cell_id,
                                       method_summary$method), ]
row.names(method_summary) <- NULL
write.csv(method_summary, file.path(results_dir, "method_summary.csv"),
          row.names = FALSE)

cell_split <- split(raw, raw$cell_id)
paired_summary <- do.call(rbind, lapply(cell_split, function(x) {
  keep <- is.finite(x$p_flip_basic) & is.finite(x$p_flip_effective) &
    is.finite(x$p_flip_standardized)
  y <- x[keep, , drop = FALSE]; n <- nrow(y)
  basic <- y$p_flip_basic < .05
  effective <- y$p_flip_effective < .05
  standardized <- y$p_flip_standardized < .05
  delta <- as.numeric(standardized) - as.numeric(effective)
  data.frame(x[1L, c("cell_id", "distribution", "truth", "n_group1",
                     "n_group2", "n_total", "missing_rate", "df")],
             n = n,
             rejection_basic = if (n) mean(basic) else NA_real_,
             rejection_effective = if (n) mean(effective) else NA_real_,
             rejection_standardized = if (n) mean(standardized) else NA_real_,
             rescue_basic_to_effective = if (n) mean(basic != effective) else NA_real_,
             rejection_delta = if (n) mean(delta) else NA_real_,
             disagreement_rate = if (n) mean(effective != standardized) else NA_real_,
             changed_to_accept = if (n) mean(effective & !standardized) else NA_real_,
             changed_to_reject = if (n) mean(!effective & standardized) else NA_real_,
             mean_abs_delta_p = if (n) mean(abs(y$p_flip_standardized -
                                                   y$p_flip_effective)) else NA_real_,
             median_variance_shift = if (n) stats::median(
               y$mean_variance_relative_shift) else NA_real_,
             median_pattern_count = if (n) stats::median(y$pattern_count) else NA_real_)
}))
row.names(paired_summary) <- NULL
write.csv(paired_summary, file.path(results_dir, "paired_summary.csv"),
          row.names = FALSE)

power_rows <- method_rows[method_rows$truth == "power" &
                            is.finite(method_rows$p_value), , drop = FALSE]
if (nrow(power_rows)) {
  power_split <- split(power_rows, interaction(
    power_rows$distribution, power_rows$n_group1, power_rows$missing_rate,
    power_rows$method, drop = TRUE))
  matched_power <- do.call(rbind, lapply(power_split, function(x) {
    null <- method_rows[
      method_rows$truth == "null" &
        method_rows$distribution == x$distribution[[1L]] &
        method_rows$n_group1 == x$n_group1[[1L]] &
        method_rows$missing_rate == x$missing_rate[[1L]] &
        method_rows$method == x$method[[1L]] & is.finite(method_rows$p_value),
      "p_value"]
    empirical_p <- vapply(x$p_value, function(pv)
      (1 + sum(null <= pv)) / (length(null) + 1), numeric(1L))
    rejected <- sum(empirical_p <= .05); n <- length(empirical_p)
    ci <- fiml_flip_wilson(rejected, n)
    data.frame(x[1L, c("distribution", "n_group1", "n_group2", "n_total",
                       "missing_rate", "df", "method")],
               n_null = length(null), n_power = n, rejected = rejected,
               nominal_power = mean(x$p_value < .05),
               matched_power = if (n) rejected / n else NA_real_,
               ci_lower = ci[[1L]], ci_upper = ci[[2L]])
  }))
  row.names(matched_power) <- NULL
} else {
  matched_power <- data.frame(
    distribution = character(), n_group1 = integer(), n_group2 = integer(),
    n_total = integer(), missing_rate = numeric(), df = integer(),
    method = character(), n_null = integer(), n_power = integer(),
    rejected = integer(), nominal_power = numeric(), matched_power = numeric(),
    ci_lower = numeric(), ci_upper = numeric())
}
write.csv(matched_power, file.path(results_dir, "matched_power.csv"),
          row.names = FALSE)

timing_columns <- c(
  missing = "missing_seconds", fit = "fit_seconds", flip_total = "flip_seconds",
  flip_setup = "flip_setup_seconds", flip_score = "flip_score_seconds",
  flip_standardization = "flip_standardization_seconds",
  flip_asymptotic = "flip_asymptotic_seconds", nested_fmg = "nested_seconds",
  replication_total = "total_seconds")
timing_summary <- do.call(rbind, lapply(cell_split, function(x) {
  do.call(rbind, lapply(names(timing_columns), function(phase) {
    values <- x[[timing_columns[[phase]]]]
    values <- values[is.finite(values)]
    data.frame(x[1L, c("cell_id", "distribution", "truth", "n_group1",
                       "n_group2", "n_total", "missing_rate", "df")],
               phase = phase, n = length(values),
               median_seconds = if (length(values)) stats::median(values) else NA_real_,
               p90_seconds = if (length(values)) unname(stats::quantile(
                 values, .9, type = 8)) else NA_real_)
  }))
}))
row.names(timing_summary) <- NULL
write.csv(timing_summary, file.path(results_dir, "timing_summary.csv"),
          row.names = FALSE)

dgp_timing <- unique(raw[c("distribution", "truth",
                           "generator_setup_seconds",
                           "unique_group_calibrations")])
write.csv(dgp_timing, file.path(results_dir, "dgp_timing.csv"), row.names = FALSE)

failures <- raw[!raw$fit_ok | !raw$flip_ok | !raw$nested_ok,
  c("cell_id", "distribution", "truth", "n_group1", "missing_rate", "rep",
    "fit_ok", "flip_ok", "nested_ok", "fit_error", "flip_error",
    "nested_error")]
write.csv(failures, file.path(results_dir, "failures.csv"), row.names = FALSE)

cache_ref <- magmaan_cache_ref()
cache_ref_string <- sprintf(
  "%s@%s git=%s dirty=%s",
  cache_ref$package, cache_ref$package_version,
  cache_ref$git_head, cache_ref$git_dirty)
metadata <- data.frame(
  key = c("mode", "reps", "flips", "cores", "seed_base", "cells", "df",
          "n_values", "rates", "distributions", "truths", "missingness",
          "wall_seconds", "magmaan_version", "git_ref"),
  value = c(opts$mode, opts$reps, opts$flips, opts$cores, opts$seed_base,
            nrow(grid), 5L, paste(opts$n_values, collapse = ","),
            paste(opts$rates, collapse = ","),
            paste(opts$distributions, collapse = ","),
            paste(opts$truths, collapse = ","),
            "Savalei-Bentler MCAR; x1/x2 intact", round(wall_seconds, 3),
            as.character(utils::packageVersion("magmaan")),
            cache_ref_string), stringsAsFactors = FALSE)
write.csv(metadata, file.path(results_dir, "metadata.csv"), row.names = FALSE)

if (opts$mode == "smoke") {
  stopifnot(nrow(raw) == nrow(grid) * opts$reps,
            all(raw$fit_ok), all(raw$flip_ok), all(raw$nested_ok),
            all(is.finite(raw$p_flip_standardized)),
            all(raw$df == 5L), nrow(failures) == 0L)
  message("SMOKE PASS")
}
cat(sprintf("done: %d replications, %d failures, %.2f wall seconds\n",
            nrow(raw), nrow(failures), wall_seconds))
cat("wrote ", normalizePath(results_dir), "\n", sep = "")
