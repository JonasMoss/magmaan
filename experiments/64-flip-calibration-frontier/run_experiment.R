#!/usr/bin/env Rscript

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
source(experiment_path("R", "design.R"))
source(experiment_path("R", "engine.R"))
source(experiment_path("R", "summaries.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--screen|--focus] [options]\n\n",
  "Fixed-G=8, fixed-df=28 score-flip calibration frontier.\n\n",
  "Profiles:\n",
  "  --smoke          8 null cells, 2 reps, 19 flips (default).\n",
  "  --screen         128 null cells, 1000 reps, 199 flips.\n",
  "  --focus          32 null/power cells, 2000 reps, 999 flips.\n\n",
  "Run controls:\n",
  "  --reps N --flips N --chunk-size N --cores N --seed-base N\n",
  "  --results-dir P              Output directory.\n",
  "  --power-calibration-file P   CSV produced by calibrate_power.R.\n",
  "  --allow-uncalibrated-power   Permit multiplier=1 focus diagnostics.\n",
  "  --shard-index J --shard-count K\n\n",
  "Cell filters (comma-separated):\n",
  "  --cells L --p L --n-avg L --information L --allocation L\n",
  "  --distributions L --truth L --max-cells N\n\n",
  "Completed chunks are reused only when the manifest and run configuration\n",
  "match exactly. Rejection uses the exact Monte Carlo convention p <= .05;\n",
  "the p < .05 sensitivity is retained separately.\n", sep = "")

parse_args <- function(args) {
  out <- list(
    profile = "smoke", reps = NULL, flips = NULL, chunk_size = NULL,
    cores = max(1L, parallel::detectCores() - 1L), seed_base = 20260713,
    results_dir = NULL, power_calibration_file = NULL,
    allow_uncalibrated_power = FALSE, shard_index = 1L, shard_count = 1L,
    cells = NULL, p = NULL, n_avg = NULL, information = NULL,
    allocation = NULL, distributions = NULL, truth = NULL, max_cells = NULL)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]], call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
    else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--screen") out$profile <- "screen"
    else if (a == "--focus") out$profile <- "focus"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--flips") out$flips <- as.integer(take())
    else if (a == "--chunk-size") out$chunk_size <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--power-calibration-file")
      out$power_calibration_file <- take()
    else if (a == "--allow-uncalibrated-power")
      out$allow_uncalibrated_power <- TRUE
    else if (a == "--shard-index") out$shard_index <- as.integer(take())
    else if (a == "--shard-count") out$shard_count <- as.integer(take())
    else if (a == "--cells") out$cells <- as.integer(parse_csv_arg(take()))
    else if (a == "--p") out$p <- as.integer(parse_csv_arg(take()))
    else if (a == "--n-avg") out$n_avg <- as.integer(parse_csv_arg(take()))
    else if (a == "--information") out$information <- parse_csv_arg(take())
    else if (a == "--allocation") out$allocation <- parse_csv_arg(take())
    else if (a == "--distributions") out$distributions <- parse_csv_arg(take())
    else if (a == "--truth") out$truth <- parse_csv_arg(take())
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(out$profile,
    smoke = c(reps = 2L, flips = 19L, chunk_size = 1L),
    screen = c(reps = 1000L, flips = 199L, chunk_size = 25L),
    focus = c(reps = 2000L, flips = 999L, chunk_size = 25L))
  for (nm in names(defaults)) if (is.null(out[[nm]])) out[[nm]] <- defaults[[nm]]
  numeric_positive <- c("reps", "flips", "chunk_size", "cores", "shard_count",
                        "shard_index")
  if (any(vapply(out[numeric_positive], function(x)
    length(x) != 1L || is.na(x) || x < 1, logical(1)))) {
    stop("run sizes, cores, and shard coordinates must be positive", call. = FALSE)
  }
  if (out$shard_index > out$shard_count)
    stop("--shard-index must be in 1..--shard-count", call. = FALSE)
  if (out$profile == "focus" && is.null(out$power_calibration_file) &&
      !out$allow_uncalibrated_power) {
    stop("--focus requires --power-calibration-file; use ",
         "--allow-uncalibrated-power only for diagnostics", call. = FALSE)
  }
  out
}

filter_design <- function(grid, cfg) {
  for (nm in c("cells", "p", "n_avg", "information", "allocation", "truth")) {
    field <- if (nm == "cells") "cell_id" else nm
    if (!is.null(cfg[[nm]])) grid <- grid[grid[[field]] %in% cfg[[nm]], ]
  }
  if (!is.null(cfg$distributions))
    grid <- grid[grid$distribution %in% cfg$distributions, ]
  # Shard on the filtered row position so cells spread evenly across shards
  # regardless of how cell_id correlates with cost. Deterministic for a given
  # (profile, filters, shard_count); seeds key off pair_id, not shard, so the
  # partition never changes results.
  grid <- grid[(seq_len(nrow(grid)) - 1L) %% cfg$shard_count + 1L == cfg$shard_index, ]
  if (!is.null(cfg$max_cells)) grid <- head(grid, cfg$max_cells)
  if (!nrow(grid)) stop("cell filters selected no rows", call. = FALSE)
  grid
}

manifest_signature <- function(x) {
  x[] <- lapply(x, as.character)
  unname(apply(x, 1L, paste, collapse = "|"))
}

write_or_validate_manifest <- function(grid, cfg, path, calibration) {
  manifest_path <- file.path(path, "manifest.csv")
  config_path <- file.path(path, "run_config.csv")
  code_files <- c(experiment_path("run_experiment.R"),
                  experiment_path("R", "design.R"),
                  experiment_path("R", "engine.R"),
                  experiment_path("R", "summaries.R"))
  code_hash <- paste(unname(tools::md5sum(code_files)), collapse = ":")
  config <- data.frame(
    schema_version = 1L, profile = cfg$profile, reps = cfg$reps,
    flips = cfg$flips, chunk_size = cfg$chunk_size, seed_base = cfg$seed_base,
    shard_index = cfg$shard_index, shard_count = cfg$shard_count,
    calibration_5 = unname(calibration[["5"]]),
    calibration_20 = unname(calibration[["20"]]),
    magmaan_version = as.character(utils::packageVersion("magmaan")),
    experiment_code_hash = code_hash, stringsAsFactors = FALSE)
  if (file.exists(manifest_path) || file.exists(config_path)) {
    if (!file.exists(manifest_path) || !file.exists(config_path))
      stop("incomplete existing run metadata in ", path, call. = FALSE)
    old_grid <- read.csv(manifest_path, stringsAsFactors = FALSE,
                         check.names = FALSE)
    old_config <- read.csv(config_path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    if (!identical(manifest_signature(old_grid), manifest_signature(grid)) ||
        !identical(unname(as.character(old_config[1L, names(config)])),
                   unname(as.character(config[1L, ])))) {
      stop("existing output configuration differs; choose a new --results-dir: ",
           path, call. = FALSE)
    }
  } else {
    write_csv(grid, manifest_path)
    write_csv(config, config_path)
  }
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
results <- cfg$results_dir %||% experiment_path("results", cfg$profile)
dir.create(results, recursive = TRUE, showWarnings = FALSE)
frontier_validate_design()
calibration <- frontier_read_power_calibration(cfg$power_calibration_file)
grid <- filter_design(frontier_design(cfg$profile, calibration), cfg)
write_or_validate_manifest(grid, cfg, results, calibration)
write_csv(grid, file.path(results, "design.csv"))
write_csv(frontier_method_table(), file.path(results, "methods.csv"))

sampler_cache <- new.env(parent = emptyenv())
spec_cache <- new.env(parent = emptyenv())
get_specs <- function(p) {
  key <- as.character(p)
  if (!exists(key, envir = spec_cache, inherits = FALSE))
    assign(key, frontier_model_specs(p), envir = spec_cache)
  get(key, envir = spec_cache, inherits = FALSE)
}

replication_seed <- function(pair_id, rep_id) {
  # pair_id excludes truth, so matched null/power cells use common random draws.
  cfg$seed_base + pair_id * 1000003 + rep_id * 1009
}

run_cell <- function(cell) {
  raw_dir <- file.path(results, "raw", sprintf("cell_%04d", cell$cell_id))
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  files <- list.files(raw_dir, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$",
                      full.names = TRUE)
  completed <- if (length(files)) unique(unlist(lapply(files, function(f)
    read.csv(f, stringsAsFactors = FALSE)$rep), use.names = FALSE)) else integer()
  remaining <- setdiff(seq_len(cfg$reps), completed)
  if (!length(remaining)) return(invisible(cfg$reps))

  pop <- frontier_population(cell$p, cell$information, cell$truth,
                             cell$power_multiplier)
  cache_key <- cache_digest(list(pop$Sigma, cell$distribution))
  cache_hit <- exists(cache_key, envir = sampler_cache, inherits = FALSE)
  if (cache_hit) {
    sampler <- get(cache_key, envir = sampler_cache, inherits = FALSE)
  } else {
    sampler <- tryCatch(frontier_calibrate_sampler(pop, cell$distribution),
                        error = function(e) e)
    if (!inherits(sampler, "error"))
      assign(cache_key, sampler, envir = sampler_cache)
  }
  cell_setup_seconds <- if (inherits(sampler, "error")) NA_real_ else
    if (cache_hit) 0 else sampler$setup_seconds
  sizes <- frontier_group_sizes(cell$n_avg, cell$allocation)
  specs <- get_specs(cell$p)
  chunks <- split(remaining, ceiling(seq_along(remaining) / cfg$chunk_size))
  for (rep_ids in chunks) {
    worker <- function(rep_id) {
      if (inherits(sampler, "error")) {
        out <- .frontier_empty_replication(
          rep_id, paste0("simulation calibration: ", conditionMessage(sampler)))
      } else {
        seed <- replication_seed(cell$pair_id, rep_id)
        out <- tryCatch(frontier_one_rep(
          cell, rep_id, sampler, specs, sizes, cfg$flips,
          dgp_seed = seed, flip_seed = seed + 700000001),
          error = function(e) .frontier_empty_replication(
            rep_id, paste0("replication: ", conditionMessage(e))))
      }
      out$generator_setup_seconds <- cell_setup_seconds
      out$generator_calibration_seconds <- if (inherits(sampler, "error"))
        NA_real_ else sampler$setup_seconds
      out$unique_group_calibrations <- if (inherits(sampler, "error")) NA_integer_
                                       else sampler$unique_group_calibrations
      out$generator_cache_hit <- cache_hit
      for (nm in names(cell)) out[[nm]] <- cell[[nm]]
      out
    }
    rows <- if (.Platform$OS.type != "windows" && cfg$cores > 1L &&
                length(rep_ids) > 1L) {
      parallel::mclapply(rep_ids, worker,
                         mc.cores = min(cfg$cores, length(rep_ids)),
                         mc.preschedule = TRUE, mc.set.seed = FALSE)
    } else lapply(rep_ids, worker)
    rows <- do.call(rbind, rows)
    path <- file.path(raw_dir, sprintf("chunk_%05d_%05d.csv",
                                       min(rep_ids), max(rep_ids)))
    tmp <- tempfile("checkpoint-", tmpdir = raw_dir, fileext = ".csv")
    write_csv(rows, tmp)
    if (!file.rename(tmp, path)) {
      unlink(tmp)
      stop("could not atomically publish checkpoint ", path, call. = FALSE)
    }
    med <- suppressWarnings(stats::median(rows$total_seconds, na.rm = TRUE))
    message(sprintf("  cell %d checkpoint %d-%d: fit %d/%d, median %.2fs/rep",
                    cell$cell_id, min(rep_ids), max(rep_ids),
                    sum(rows$fit_ok), nrow(rows), med))
  }
  invisible(cfg$reps)
}

cat(sprintf("profile=%s cells=%d reps=%d flips=%d chunk=%d cores=%d shard=%d/%d\n",
            cfg$profile, nrow(grid), cfg$reps, cfg$flips, cfg$chunk_size,
            cfg$cores, cfg$shard_index, cfg$shard_count))
wall_begin <- proc.time()[["elapsed"]]
for (k in seq_len(nrow(grid))) {
  message(sprintf("cell %d/%d (stable id %d)", k, nrow(grid), grid$cell_id[[k]]))
  run_cell(as.list(grid[k, , drop = FALSE]))
}

files <- list.files(file.path(results, "raw"),
                    pattern = "^chunk_[0-9]+_[0-9]+\\.csv$",
                    recursive = TRUE, full.names = TRUE)
raw <- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE,
                            check.names = FALSE))
raw <- raw[raw$cell_id %in% grid$cell_id & raw$rep <= cfg$reps, ]
raw <- raw[order(raw$cell_id, raw$rep), ]
if (anyDuplicated(raw[c("cell_id", "rep")]))
  stop("duplicate checkpoint rows detected", call. = FALSE)
expected <- nrow(grid) * cfg$reps
if (nrow(raw) != expected)
  stop("checkpoint set has ", nrow(raw), " rows; expected ", expected,
       call. = FALSE)

pvalues <- raw[c(frontier_design_columns, "rep", frontier_p_columns)]
write_csv(pvalues, file.path(results, "pvalues.csv"))
method_rows <- frontier_method_rows(raw)
method_summary <- frontier_summarize_methods(method_rows)
write_csv(method_summary, file.path(results, "method_summary.csv"))
write_csv(frontier_availability_summary(method_rows),
          file.path(results, "availability_summary.csv"))
write_csv(frontier_summarize_paired(raw), file.path(results, "paired_summary.csv"))
write_csv(frontier_summarize_matched_power(method_rows),
          file.path(results, "matched_power.csv"))
write_csv(frontier_summarize_timing(raw), file.path(results, "timing_summary.csv"))
write_csv(frontier_calibration_summary(method_summary),
          file.path(results, "calibration_summary.csv"))
failures <- raw[!raw$fit_ok | !raw$flip_ok | !raw$nested_ok,
                c(frontier_design_columns, "rep", "fit_ok", "flip_ok",
                  "nested_ok", "fit_error", "flip_error", "nested_error")]
write_csv(failures, file.path(results, "failures.csv"))
elapsed <- proc.time()[["elapsed"]] - wall_begin
write_metadata(file.path(results, "metadata.csv"), list(
  profile = cfg$profile, cells = nrow(grid), reps = cfg$reps,
  flips = cfg$flips, chunk_size = cfg$chunk_size, cores = cfg$cores,
  seed_base = cfg$seed_base, shard_index = cfg$shard_index,
  shard_count = cfg$shard_count, elapsed_seconds = elapsed,
  alpha_convention = "p <= 0.05", strict_sensitivity = "p < 0.05",
  target_skewness = 3, target_excess_kurtosis = 21,
  paper_doi = "10.3758/s13428-026-02968-4",
  paper_osf = "https://osf.io/h2y3n/"), packages = "magmaan")
cat(sprintf("wrote %s in %.1fs (%d pipeline failures)\n",
            results, elapsed, nrow(failures)))
