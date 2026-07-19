#!/usr/bin/env Rscript
# Focused fully recomputed permutation robust-score arm for experiment 61.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("scripts/run_score_pivot.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(script))), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "score_pivot.R",
                       script = dirname(script_path())))

usage <- function() {
  cat(
    "Usage: Rscript scripts/run_score_pivot.R [options]\n\n",
    "Profiles:\n",
    "  --smoke             1 replication, 9 permutations (default).\n",
    "  --probe             100 replications, 99 permutations.\n",
    "  --sensitivity       500 replications, 99 permutations.\n",
    "  --allocation-gate   1:3 VM gate: 500 replications, 99 permutations.\n",
    "  --allocation-reverse  3:1 mirror of the allocation gate.\n",
    "  --mixture-gate      1:3 VM analytic gate: 2000 reps, no permutations.\n",
    "  --power-gate        Local 1/sqrt(N) loading alternative, both mirrors.\n",
    "  --full              1000 replications, 199 permutations.\n\n",
    "Options:\n",
    "  --reps N            Override replications per cell.\n",
    "  --permutations B    Override label permutations per replication.\n",
    "  --cores N           Parallel replication workers.\n",
    "  --seed-base N       Base seed. Default: 20260719.\n",
    "  --q L               Restriction ranks. Default: 1,4,8.\n",
    "  --n-total L         Total sample sizes. Default: 60,120,240.\n",
    "  --regimes L         exchangeable_normal,heterogeneous_vm.\n",
    "  --allocations L     1:1, 1:3, or 3:1. Default: 1:1.\n",
    "  --local-strength X  Local loading shift multiplier. Default: 1.\n",
    "  --max-cells N       Run the first N selected cells.\n",
    "  --no-wald           Skip the fully recomputed sandwich-Wald comparator.\n",
    "  --results-dir PATH  Output root. Default: results.\n\n",
    "The permutation score and permutation Hotelling columns must agree exactly:\n",
    "the Hotelling tail is a strictly decreasing transform of the same score\n",
    "quadratic when n and q are fixed.\n",
    sep = "")
}

parse_args <- function(args) {
  out <- list(
    mode = "smoke", reps = NULL, permutations = NULL,
    cores = max(1L, parallel::detectCores() - 1L),
    seed_base = 20260719, q = NULL, n_total = NULL,
    regimes = NULL, allocations = NULL,
    local_strength = 1,
    max_cells = NULL, include_wald = TRUE,
    results_dir = experiment_path("results", script = dirname(script_path())))
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]],
                               call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--smoke") out$mode <- "smoke"
    else if (a == "--probe") out$mode <- "probe"
    else if (a == "--sensitivity") out$mode <- "sensitivity"
    else if (a == "--allocation-gate") out$mode <- "allocation"
    else if (a == "--allocation-reverse") out$mode <- "allocation-reverse"
    else if (a == "--mixture-gate") out$mode <- "mixture"
    else if (a == "--power-gate") out$mode <- "power"
    else if (a == "--full") out$mode <- "full"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--permutations") out$permutations <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--q") out$q <- as.integer(parse_csv_numeric(take()))
    else if (a == "--n-total") {
      out$n_total <- as.integer(parse_csv_numeric(take()))
    } else if (a == "--regimes") out$regimes <- parse_csv_arg(take())
    else if (a == "--allocations") out$allocations <- parse_csv_arg(take())
    else if (a == "--local-strength") {
      out$local_strength <- as.numeric(take())
    }
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else if (a == "--no-wald") out$include_wald <- FALSE
    else if (a == "--results-dir") out$results_dir <- take()
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$mode,
    smoke = c(reps = 1L, permutations = 9L),
    probe = c(reps = 100L, permutations = 99L),
    sensitivity = c(reps = 500L, permutations = 99L),
    allocation = c(reps = 500L, permutations = 99L),
    `allocation-reverse` = c(reps = 500L, permutations = 99L),
    mixture = c(reps = 2000L, permutations = 0L),
    power = c(reps = 500L, permutations = 99L),
    full = c(reps = 1000L, permutations = 199L))
  for (nm in names(defaults)) {
    if (is.null(out[[nm]])) out[[nm]] <- defaults[[nm]]
  }
  if (out$mode %in% c("allocation", "allocation-reverse")) {
    if (is.null(out$q)) out$q <- c(4L, 8L)
    if (is.null(out$n_total)) out$n_total <- c(120L, 240L, 480L, 960L)
    if (is.null(out$regimes)) out$regimes <- "heterogeneous_vm"
    if (is.null(out$allocations)) {
      out$allocations <- if (out$mode == "allocation") "1:3" else "3:1"
    }
  } else if (out$mode == "mixture") {
    if (is.null(out$q)) out$q <- c(4L, 8L)
    if (is.null(out$n_total)) {
      out$n_total <- c(480L, 960L, 1920L, 3840L)
    }
    if (is.null(out$regimes)) out$regimes <- "heterogeneous_vm"
    if (is.null(out$allocations)) out$allocations <- "1:3"
    out$include_wald <- FALSE
  } else if (out$mode == "power") {
    if (is.null(out$q)) out$q <- c(4L, 8L)
    if (is.null(out$n_total)) out$n_total <- c(120L, 240L, 480L, 960L)
    if (is.null(out$regimes)) out$regimes <- "heterogeneous_vm"
    if (is.null(out$allocations)) out$allocations <- c("1:3", "3:1")
  } else {
    if (is.null(out$q)) out$q <- c(1L, 4L, 8L)
    if (is.null(out$n_total)) out$n_total <- c(60L, 120L, 240L)
    if (is.null(out$regimes)) {
      out$regimes <- c("exchangeable_normal", "heterogeneous_vm")
    }
    if (is.null(out$allocations)) out$allocations <- "1:1"
  }
  if (any(!out$q %in% c(1L, 4L, 8L))) {
    stop("--q must be drawn from 1,4,8", call. = FALSE)
  }
  if (any(out$n_total < 20L) || any(out$n_total %% 2L != 0L)) {
    stop("--n-total values must be even and at least 20", call. = FALSE)
  }
  allowed_regimes <- c("exchangeable_normal", "heterogeneous_vm")
  if (any(!out$regimes %in% allowed_regimes)) {
    stop("unknown regime in --regimes", call. = FALSE)
  }
  allowed_allocations <- c("1:1", "1:3", "3:1")
  if (any(!out$allocations %in% allowed_allocations)) {
    stop("unknown allocation in --allocations", call. = FALSE)
  }
  if (any(out$allocations %in% c("1:3", "3:1")) &&
      any(out$n_total %% 4L != 0L)) {
    stop("unequal allocation requires total N divisible by four",
         call. = FALSE)
  }
  positive <- c("reps", "cores")
  if (any(vapply(out[positive], function(x) is.na(x) || x < 1L,
                 logical(1)))) {
    stop("reps and cores must be positive", call. = FALSE)
  }
  if (is.na(out$permutations) || out$permutations < 0L) {
    stop("permutations must be non-negative", call. = FALSE)
  }
  if (!is.finite(out$local_strength) || out$local_strength <= 0) {
    stop("local strength must be finite and positive", call. = FALSE)
  }
  out
}

score_pivot_cells <- function(cfg) {
  cells <- expand.grid(
    q = cfg$q, n_total = cfg$n_total, regime = cfg$regimes,
    allocation = cfg$allocations,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  cells <- cells[order(match(cells$allocation, cfg$allocations),
                       match(cells$regime, cfg$regimes),
                       cells$n_total, cells$q), , drop = FALSE]
  cells$cell_id <- seq_len(nrow(cells))
  cells$p <- 9L
  cells$n1 <- ifelse(
    cells$allocation == "1:1", cells$n_total %/% 2L,
    ifelse(cells$allocation == "1:3",
           cells$n_total %/% 4L, 3L * cells$n_total %/% 4L))
  cells$n2 <- cells$n_total - cells$n1
  cells$n_over_q <- cells$n_total / cells$q
  cells$truth <- if (cfg$mode == "power") "local_alternative" else "null"
  cells$local_strength <- if (cfg$mode == "power") cfg$local_strength else 0
  cells$delta <- if (cfg$mode == "power") {
    cfg$local_strength / sqrt(cells$n_total)
  } else 0
  cells <- cells[, c("cell_id", "p", "q", "n_total", "n1", "n2",
                     "allocation", "n_over_q", "regime", "truth",
                     "local_strength", "delta")]
  if (!is.null(cfg$max_cells)) cells <- head(cells, cfg$max_cells)
  if (!nrow(cells)) stop("no score-pivot cells selected", call. = FALSE)
  cells
}

decorate_score_rows <- function(rows, cell, cfg) {
  for (nm in names(cell)) rows[[nm]] <- cell[[nm]][[1L]]
  rows$permutations_requested <- cfg$permutations
  rows$include_wald <- cfg$include_wald
  rows
}

summarize_score_cell <- function(x, alpha = .05) {
  p_columns <- grep("^p_", names(x), value = TRUE)
  ans <- x[1L, c("cell_id", "p", "q", "n_total", "n1", "n2",
                  "allocation", "n_over_q", "regime", "truth",
                  "local_strength", "delta"), drop = FALSE]
  ans$reps <- nrow(x)
  ans$observed_score_ok <- sum(x$observed_score_ok)
  ans$observed_wald_ok <- if (all(is.na(x$observed_wald_ok))) {
    NA_integer_
  } else {
    sum(x$observed_wald_ok, na.rm = TRUE)
  }
  ans$score_failure_rate <- mean(!x$observed_score_ok)
  ans$wald_failure_rate <- if (all(is.na(x$observed_wald_ok))) {
    NA_real_
  } else {
    mean(!x$observed_wald_ok, na.rm = TRUE)
  }
  for (nm in p_columns) {
    usable <- x[[nm]][is.finite(x[[nm]])]
    suffix <- sub("^p_", "", nm)
    ans[[paste0("reject_", suffix)]] <- if (length(usable)) {
      mean(usable <= alpha)
    } else NA_real_
    ans[[paste0("n_", suffix)]] <- length(usable)
  }
  ans$median_valid_score_permutation_fraction <-
    if (all(x$permutations_requested == 0L)) NA_real_ else stats::median(
      x$n_permutation_score / x$permutations_requested, na.rm = TRUE)
  ans$median_valid_wald_permutation_fraction <-
    if (all(x$permutations_requested == 0L)) NA_real_ else stats::median(
      x$n_permutation_wald / x$permutations_requested, na.rm = TRUE)
  identity_error <- x$permutation_rank_identity_error[
    is.finite(x$permutation_rank_identity_error)]
  ans$max_permutation_rank_identity_error <- if (length(identity_error)) {
    max(identity_error)
  } else {
    NA_real_
  }
  ans$median_simulation_seconds <- stats::median(
    x$simulation_seconds, na.rm = TRUE)
  ans$median_observed_score_seconds <- stats::median(
    x$observed_score_seconds, na.rm = TRUE)
  ans$median_observed_wald_seconds <- stats::median(
    x$observed_wald_seconds, na.rm = TRUE)
  ans$median_permutation_score_seconds <- stats::median(
    x$permutation_score_seconds, na.rm = TRUE)
  ans$median_permutation_wald_seconds <- stats::median(
    x$permutation_wald_seconds, na.rm = TRUE)
  ans$median_score_eigen_mean <- stats::median(
    x$score_eigen_mean, na.rm = TRUE)
  ans$median_score_eigen_cv <- stats::median(
    x$score_eigen_cv, na.rm = TRUE)
  ans$median_score_eigen_ratio <- stats::median(
    x$score_eigen_ratio, na.rm = TRUE)
  ans$median_observed_seconds <- stats::median(
    x$observed_seconds, na.rm = TRUE)
  ans$median_permutation_seconds <- stats::median(
    x$permutation_seconds, na.rm = TRUE)
  ans$median_total_seconds <- stats::median(
    x$total_seconds, na.rm = TRUE)
  ans
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
cells <- score_pivot_cells(cfg)
message("Calibrating exchangeable-normal and heterogeneous-VM samplers...")
null_pop <- score_pivot_population()
null_samplers <- score_pivot_calibrate_samplers(null_pop)
path <- file.path(cfg$results_dir, paste0("score-pivot-", cfg$mode))
dir.create(path, recursive = TRUE, showWarnings = FALSE)
replication_path <- file.path(path, "replications.csv")
if (file.exists(replication_path)) unlink(replication_path)
write_csv(cells, file.path(path, "manifest.csv"))
write_csv(
  data.frame(
    schema_version = 3L, mode = cfg$mode, reps = cfg$reps,
    permutations = cfg$permutations, cores = cfg$cores,
    seed_base = cfg$seed_base, local_strength = cfg$local_strength,
    include_wald = cfg$include_wald),
  file.path(path, "run_config.csv"))

message(sprintf(
  "Score pivot: %d cells x %d reps x %d permutations; %d cores; wald=%s",
  nrow(cells), cfg$reps, cfg$permutations, cfg$cores, cfg$include_wald))
started <- Sys.time()
summaries <- vector("list", nrow(cells))
for (i in seq_len(nrow(cells))) {
  cell <- cells[i, , drop = FALSE]
  specs <- score_pivot_specs(cell$q[[1L]], cell$p[[1L]])
  if (cell$truth[[1L]] == "local_alternative") {
    pop <- score_pivot_population(
      p = cell$p[[1L]], q = cell$q[[1L]], delta = cell$delta[[1L]])
    sampler <- score_pivot_calibrate_samplers(pop)[[cell$regime[[1L]]]]
  } else {
    pop <- null_pop
    sampler <- null_samplers[[cell$regime[[1L]]]]
  }
  worker <- function(rep_id) {
    score_pivot_replication(
      rep_id, cell, specs, sampler, pop, cfg$permutations,
      cfg$seed_base, include_wald = cfg$include_wald)
  }
  if (cfg$cores > 1L && cfg$reps > 1L) {
    rows <- parallel::mclapply(
      seq_len(cfg$reps), worker,
      mc.cores = min(cfg$cores, cfg$reps),
      mc.preschedule = TRUE, mc.set.seed = FALSE)
  } else {
    rows <- lapply(seq_len(cfg$reps), worker)
  }
  rows <- decorate_score_rows(do.call(rbind, rows), cell, cfg)
  append_csv(rows, replication_path)
  summaries[[i]] <- summarize_score_cell(rows)
  total_elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  eta <- (total_elapsed / i) * (nrow(cells) - i)
  message(sprintf(
    "  [%d/%d] %s %s q=%d N=%d (%d,%d): score %d/%d, median %.2fs/rep; ETA %.1f min",
    i, nrow(cells), cell$regime, cell$allocation, cell$q, cell$n_total,
    cell$n1, cell$n2,
    sum(rows$observed_score_ok), nrow(rows),
    stats::median(rows$total_seconds, na.rm = TRUE),
    max(0, eta) / 60))
}

summary <- do.call(rbind, summaries)
write_csv(summary, file.path(path, "summary.csv"))
failures <- read.csv(replication_path, stringsAsFactors = FALSE)
failures <- failures[
  !failures$observed_score_ok |
    (cfg$include_wald & !failures$observed_wald_ok),
  c("cell_id", "rep", "regime", "q", "n_total",
    "error_score", "error_wald"), drop = FALSE]
write_csv(failures, file.path(path, "failures.csv"))
write_metadata(
  file.path(path, "metadata.csv"),
  values = list(
    experiment = "61-permutation-measurement-invariance",
    arm = "fully-recomputed-score-pivot",
    mode = cfg$mode, reps = cfg$reps,
    permutations = cfg$permutations, cores = cfg$cores,
    seed_base = cfg$seed_base, q = cfg$q, n_total = cfg$n_total,
    regimes = cfg$regimes, allocations = cfg$allocations,
    local_strength = cfg$local_strength,
    include_wald = cfg$include_wald,
    elapsed_minutes = round(
      as.numeric(difftime(Sys.time(), started, units = "mins")), 3),
    score_hotelling_permutation_identity =
      "rank-equivalent; asserted within 1e-15"),
  packages = "magmaan")
message("Wrote:\n  ", paste(
  c(replication_path, file.path(path, c(
    "summary.csv", "manifest.csv", "run_config.csv",
    "failures.csv", "metadata.csv"))),
  collapse = "\n  "))
