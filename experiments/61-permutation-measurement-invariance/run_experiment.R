#!/usr/bin/env Rscript
# Experiment 61: fully refitted studentized permutation Wald tests for
# measurement invariance. Two checkpointed studies share the same statistic:
# the two-factor reference-law design and the Chen--Chao imbalance design.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "main_study.R"))
source(experiment_path("R", "simulation.R"))
source(experiment_path("R", "inference.R"))

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Profiles:\n",
    "  --smoke             Both study smokes (default): 1 rep, 9 permutations.\n",
    "  --modal             Representative design: 100 reps, 99 permutations.\n",
    "  --full              Paper grid: 1000 reps, 499 permutations.\n\n",
    "Study and size:\n",
    "  --study S           reference, imbalance, or both. Default: both.\n",
    "  --reps N            Override replications per cell.\n",
    "  --permutations B    Override random permutations per replication.\n",
    "  --chunk-size N      Reps written per checkpoint. Default: 1/5/10.\n",
    "  --cores N           Parallel replication fits.\n",
    "  --seed-base N       Base seed. Default: 20260711.\n",
    "  --results-dir PATH  Output root. Default: results.\n\n",
    "Cell filters (comma-separated):\n",
    "  --cells L           Stable cell IDs.\n",
    "  --generators L      normal,ig1,ig2,ordinal_*.\n",
    "  --steps L           metric,scalar,strict.\n",
    "  --truth L           null,alternative.\n",
    "  --allocations L     1:1,1:3,unbalanced,matched_balanced.\n",
    "  --patterns L        none,small,large,mixed,nonuniform.\n",
    "  --max-cells N       Run the first N filtered cells.\n\n",
    "  --shard-index J     Run shard J (one-based).\n",
    "  --shard-count K     Partition stable cell IDs over K jobs.\n\n",
    "Alternative calibration overrides (reference study):\n",
    "  --calibration-file PATH  CSV with step and delta columns.\n",
    "  --delta-metric X --delta-scalar X --delta-strict X\n\n",
    "Completed chunks are reused automatically. Use a new --results-dir for a\n",
    "different configuration. Each study writes under <root>/<study>-<mode>/.\n",
    sep = "")
}

parse_args <- function(args) {
  out <- list(study = "both", mode = "smoke", reps = NULL, permutations = NULL,
              chunk_size = NULL, cores = max(1L, parallel::detectCores() - 1L),
              seed_base = 20260711, results_dir = experiment_path("results"), cells = NULL,
              generators = NULL, steps = NULL, truth = NULL, allocations = NULL,
              patterns = NULL, max_cells = NULL,
              shard_index = 1L, shard_count = 1L,
              calibration_file = NULL,
              delta_metric = NULL, delta_scalar = NULL, delta_strict = NULL)
  i <- 1L
  take <- function() { i <<- i + 1L; args[[i]] }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
    else if (a == "--smoke") out$mode <- "smoke"
    else if (a == "--modal") out$mode <- "modal"
    else if (a == "--full") out$mode <- "full"
    else if (a == "--study") out$study <- take()
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--permutations") out$permutations <- as.integer(take())
    else if (a == "--chunk-size") out$chunk_size <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--cells") out$cells <- as.integer(parse_csv_arg(take()))
    else if (a == "--generators") out$generators <- parse_csv_arg(take())
    else if (a == "--steps") out$steps <- parse_csv_arg(take())
    else if (a == "--truth") out$truth <- parse_csv_arg(take())
    else if (a == "--allocations") out$allocations <- parse_csv_arg(take())
    else if (a == "--patterns") out$patterns <- parse_csv_arg(take())
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else if (a == "--shard-index") out$shard_index <- as.integer(take())
    else if (a == "--shard-count") out$shard_count <- as.integer(take())
    else if (a == "--calibration-file") out$calibration_file <- take()
    else if (a == "--delta-metric") out$delta_metric <- as.numeric(take())
    else if (a == "--delta-scalar") out$delta_scalar <- as.numeric(take())
    else if (a == "--delta-strict") out$delta_strict <- as.numeric(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (!out$study %in% c("reference", "imbalance", "both"))
    stop("--study must be reference, imbalance, or both", call. = FALSE)
  defaults <- switch(out$mode,
    smoke = c(reps = 1L, permutations = 9L, chunk_size = 1L),
    modal = c(reps = 100L, permutations = 99L, chunk_size = 5L),
    full = c(reps = 1000L, permutations = 499L, chunk_size = 10L))
  for (nm in names(defaults)) if (is.null(out[[nm]])) out[[nm]] <- defaults[[nm]]
  numeric_positive <- c("reps", "permutations", "chunk_size", "cores")
  if (any(vapply(out[numeric_positive], function(x) is.na(x) || x < 1L, logical(1))))
    stop("replications, permutations, chunk size, and cores must be positive",
         call. = FALSE)
  if (is.na(out$shard_count) || is.na(out$shard_index) || out$shard_count < 1L ||
      out$shard_index < 1L || out$shard_index > out$shard_count)
    stop("shard index must be in 1..shard count", call. = FALSE)
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))

filter_cells <- function(cells, cfg) {
  if (!is.null(cfg$cells)) cells <- cells[cells$cell_id %in% cfg$cells, ]
  if (!is.null(cfg$generators)) cells <- cells[cells$generator %in% cfg$generators, ]
  if (!is.null(cfg$steps)) cells <- cells[cells$step %in% cfg$steps, ]
  if (!is.null(cfg$truth)) cells <- cells[cells$truth %in% cfg$truth, ]
  if (!is.null(cfg$allocations)) cells <- cells[cells$allocation %in% cfg$allocations, ]
  if (!is.null(cfg$patterns)) cells <- cells[cells$pattern %in% cfg$patterns, ]
  if (!is.null(cfg$max_cells)) cells <- head(cells, cfg$max_cells)
  if (!nrow(cells)) stop("cell filters selected no rows", call. = FALSE)
  cells
}

apply_delta_overrides <- function(cells, cfg) {
  if (!is.null(cfg$calibration_file) && any(cells$study == "reference")) {
    calibration <- read.csv(cfg$calibration_file, stringsAsFactors = FALSE)
    if (!all(c("step", "delta") %in% names(calibration)))
      stop("--calibration-file needs step and delta columns", call. = FALSE)
    for (i in seq_len(nrow(calibration))) {
      cells$delta[cells$step == calibration$step[[i]] &
                    cells$truth == "alternative"] <- calibration$delta[[i]]
    }
  }
  for (step in c("metric", "scalar", "strict")) {
    value <- cfg[[paste0("delta_", step)]]
    if (!is.null(value)) cells$delta[cells$step == step & cells$truth == "alternative"] <- value
  }
  cells
}

cell_signature <- function(cells) {
  fields <- c("cell_id", "study", "p", "n1", "n2", "generator", "step", "truth",
              "null_kind", "pattern", "affected_fraction", "delta")
  values <- cells[, fields, drop = FALSE]
  values[] <- lapply(values, function(x) {
    x <- as.character(x)
    x[is.na(x)] <- ""
    x
  })
  apply(values, 1L, paste, collapse = "|")
}

write_or_validate_manifest <- function(cells, path, cfg, study) {
  manifest_path <- file.path(path, "manifest.csv")
  config_path <- file.path(path, "run_config.csv")
  config <- data.frame(
    study = study, mode = cfg$mode, reps = cfg$reps,
    permutations = cfg$permutations, chunk_size = cfg$chunk_size,
    seed_base = cfg$seed_base, stringsAsFactors = FALSE)
  if (file.exists(manifest_path) || file.exists(config_path)) {
    if (!file.exists(manifest_path) || !file.exists(config_path))
      stop("incomplete existing run metadata in ", path, call. = FALSE)
    old_cells <- read.csv(manifest_path, stringsAsFactors = FALSE,
                          check.names = FALSE)
    old_config <- read.csv(config_path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    if (!identical(unname(cell_signature(old_cells)),
                   unname(cell_signature(cells))) ||
        !identical(unname(as.character(old_config[1L, names(config)])),
                   unname(as.character(config[1L, names(config)])))) {
      stop("existing output configuration differs; choose a new --results-dir: ",
           path, call. = FALSE)
    }
  } else {
    write_csv(cells, manifest_path)
    write_csv(config, config_path)
  }
}

replication_seed <- function(cfg, study, cell_id, rep_id) {
  study_offset <- if (study == "reference") 0 else 500000000
  cfg$seed_base + study_offset + cell_id * 1000003 + rep_id * 1009
}

decorate_rows <- function(rows, cell) {
  fields <- names(cell)
  for (nm in fields) rows[[nm]] <- cell[[nm]][[1L]]
  rows
}

sampler_cache <- new.env(parent = emptyenv())

run_cell <- function(cell, cfg, raw_dir) {
  cell_dir <- file.path(raw_dir, sprintf("cell_%04d", cell$cell_id[[1L]]))
  dir.create(cell_dir, recursive = TRUE, showWarnings = FALSE)
  existing <- list.files(cell_dir, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$",
                         full.names = TRUE)
  completed <- integer()
  if (length(existing)) {
    completed <- unique(unlist(lapply(existing, function(f)
      read.csv(f, stringsAsFactors = FALSE)$rep), use.names = FALSE))
  }
  remaining <- setdiff(seq_len(cfg$reps), completed)
  if (!length(remaining)) return(invisible(length(completed)))

  pop <- population_for_cell(cell)
  setup_started <- proc.time()[["elapsed"]]
  cache_key <- sampler_cache_key(pop, cell)
  if (exists(cache_key, envir = sampler_cache, inherits = FALSE)) {
    sampler <- get(cache_key, envir = sampler_cache, inherits = FALSE)
  } else {
    sampler <- tryCatch(calibrate_cell_sampler(pop, cell), error = function(e) e)
    if (!inherits(sampler, "error")) assign(cache_key, sampler, envir = sampler_cache)
  }
  setup_seconds <- proc.time()[["elapsed"]] - setup_started
  chunks <- split(remaining, ceiling(seq_along(remaining) / cfg$chunk_size))
  for (rep_ids in chunks) {
    if (inherits(sampler, "error")) {
      rows <- do.call(rbind, lapply(rep_ids, function(rep_id)
        empty_replication(rep_id, paste0("simulation calibration: ",
                                        conditionMessage(sampler)))))
    } else {
      simulations <- lapply(rep_ids, function(rep_id) {
        seed <- replication_seed(cfg, cell$study[[1L]], cell$cell_id[[1L]], rep_id)
        started <- proc.time()[["elapsed"]]
        value <- tryCatch(draw_cell_replication(sampler, c(cell$n1, cell$n2), seed),
                          error = function(e) e)
        list(value = value, seconds = proc.time()[["elapsed"]] - started,
             seed = seed)
      })
      worker <- function(j) {
        rep_id <- rep_ids[[j]]
        sim <- simulations[[j]]
        if (inherits(sim$value, "error")) {
          out <- empty_replication(rep_id,
                                   paste0("simulation draw: ", conditionMessage(sim$value)))
          out$simulation_seconds <- sim$seconds
          return(out)
        }
        run_replication(rep_id, sim$value, pop, cell$step[[1L]], cfg$permutations,
                        sim$seed + 700000001, sim$seconds)
      }
      if (cfg$cores > 1L && length(rep_ids) > 1L) {
        rows <- parallel::mclapply(seq_along(rep_ids), worker,
                                   mc.cores = min(cfg$cores, length(rep_ids)),
                                   mc.preschedule = TRUE, mc.set.seed = FALSE)
      } else rows <- lapply(seq_along(rep_ids), worker)
      rows <- do.call(rbind, rows)
    }
    rows$permutations_requested <- cfg$permutations
    rows$setup_seconds <- setup_seconds
    rows <- decorate_rows(rows, cell)
    chunk_path <- file.path(cell_dir, sprintf("chunk_%05d_%05d.csv",
                                              min(rep_ids), max(rep_ids)))
    write_csv(rows, chunk_path)
    message(sprintf("    cell %d: checkpoint reps %d-%d; observed %d/%d; median %.2fs/rep",
                    cell$cell_id[[1L]], min(rep_ids), max(rep_ids),
                    sum(rows$observed_ok), nrow(rows),
                    stats::median(rows$total_seconds, na.rm = TRUE)))
  }
  invisible(cfg$reps)
}

summarize_cell <- function(files, alpha = .05) {
  x <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE,
                            check.names = FALSE))
  x <- x[order(x$rep), ]
  p_columns <- grep("^p_", names(x), value = TRUE)
  design_columns <- c("cell_id", "study", "p", "n_total", "n1", "n2",
                      "allocation", "ratio", "generator", "data_type",
                      "categories", "thresholds", "step", "truth", "null_kind",
                      "pattern", "affected_fraction", "delta")
  ans <- x[1L, design_columns, drop = FALSE]
  ans$reps <- nrow(x)
  ans$observed_ok <- sum(x$observed_ok)
  ans$failure_rate <- mean(!x$observed_ok)
  for (nm in p_columns) {
    usable <- x[[nm]][is.finite(x[[nm]])]
    suffix <- sub("^p_", "", nm)
    ans[[paste0("reject_", suffix)]] <- if (length(usable)) mean(usable <= alpha) else NA_real_
    ans[[paste0("n_", suffix)]] <- length(usable)
  }
  ans$median_valid_permutation_fraction <-
    stats::median(x$n_perm_studentized / x$permutations_requested, na.rm = TRUE)
  ans$mean_setup_seconds <- mean(x$setup_seconds, na.rm = TRUE)
  ans$median_simulation_seconds <- stats::median(x$simulation_seconds, na.rm = TRUE)
  ans$median_observed_seconds <- stats::median(x$observed_seconds, na.rm = TRUE)
  ans$median_permutation_seconds <- stats::median(x$permutation_seconds, na.rm = TRUE)
  ans$median_total_seconds <- stats::median(x$total_seconds, na.rm = TRUE)
  ans
}

run_study <- function(study, cfg) {
  manifest_cells <- if (study == "reference") reference_cells(cfg$mode) else imbalance_cells(cfg$mode)
  manifest_cells <- apply_delta_overrides(manifest_cells, cfg)
  manifest_cells <- filter_cells(manifest_cells, cfg)
  cells <- manifest_cells[(manifest_cells$cell_id - 1L) %% cfg$shard_count ==
                            cfg$shard_index - 1L, , drop = FALSE]
  if (!nrow(cells)) stop("selected shard has no cells", call. = FALSE)
  path <- file.path(cfg$results_dir, paste0(study, "-", cfg$mode))
  raw_dir <- file.path(path, "raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  write_or_validate_manifest(manifest_cells, path, cfg, study)
  message(sprintf("%s: %d/%d cells (shard %d/%d) x %d reps x %d permutations; %d cores; checkpoints %s",
                  study, nrow(cells), nrow(manifest_cells), cfg$shard_index,
                  cfg$shard_count, cfg$reps, cfg$permutations, cfg$cores, raw_dir))
  started <- Sys.time()
  for (i in seq_len(nrow(cells))) {
    cell <- cells[i, , drop = FALSE]
    message(sprintf("  [%d/%d] cell %d %s/%s/%s n=(%d,%d)",
                    i, nrow(cells), cell$cell_id, cell$generator, cell$step,
                    paste(cell$truth, cell$null_kind, sep = ":"), cell$n1, cell$n2))
    tryCatch(run_cell(cell, cfg, raw_dir), error = function(e)
      message("    CELL FAILED before checkpoint: ", conditionMessage(e)))
  }
  cell_dirs <- list.dirs(raw_dir, recursive = FALSE, full.names = TRUE)
  summaries <- lapply(cell_dirs, function(d) {
    files <- list.files(d, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$", full.names = TRUE)
    if (!length(files)) return(NULL)
    summarize_cell(files)
  })
  summaries <- Filter(Negate(is.null), summaries)
  if (!length(summaries)) stop("no checkpointed cells completed for ", study,
                               call. = FALSE)
  summary <- do.call(rbind, summaries)
  summary <- summary[order(summary$cell_id), ]
  write_csv(summary, file.path(path, "summary.csv"))
  failures <- do.call(rbind, lapply(cell_dirs, function(d) {
    files <- list.files(d, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$", full.names = TRUE)
    if (!length(files)) return(NULL)
    x <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
    x[!x$observed_ok, c("cell_id", "rep", "generator", "step", "truth", "error"),
      drop = FALSE]
  }))
  if (!is.null(failures) && nrow(failures)) write_csv(failures, file.path(path, "failures.csv"))
  write_metadata(
    file.path(path, "metadata.csv"),
    values = list(experiment = "61-permutation-measurement-invariance",
                  study = study, mode = cfg$mode, reps = cfg$reps,
                  permutations = cfg$permutations, chunk_size = cfg$chunk_size,
                  cores = cfg$cores, seed_base = cfg$seed_base,
                  shard_index = cfg$shard_index, shard_count = cfg$shard_count,
                  cells_in_manifest = nrow(manifest_cells),
                  shard_cells_requested = nrow(cells),
                  cells_completed_available = nrow(summary),
                  invocation_elapsed_minutes = round(as.numeric(difftime(Sys.time(), started,
                                                                          units = "mins")), 3),
                  recorded_cell_work_minutes = round(sum(
                    summary$mean_setup_seconds + summary$median_total_seconds * summary$reps,
                    na.rm = TRUE) / 60, 3),
                  alternative_calibration = if (study == "reference")
                    "pilot defaults unless overridden or supplied by calibration file" else
                    "Chen--Chao metric/scalar values plus strict proportional extension"),
    packages = "magmaan")
  message(sprintf("%s complete: %d/%d cells summarized in %s (%.2f min)",
                  study, nrow(summary), nrow(manifest_cells), path,
                  as.numeric(difftime(Sys.time(), started, units = "mins"))))
  invisible(path)
}

studies <- if (cfg$study == "both") c("reference", "imbalance") else cfg$study
paths <- lapply(studies, run_study, cfg = cfg)
message("Wrote:\n  ", paste(unlist(paths), collapse = "\n  "))
