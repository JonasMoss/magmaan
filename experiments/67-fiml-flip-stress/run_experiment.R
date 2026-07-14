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
source(support_path("R", "missingness.R"))
source(experiment_path("R", "design.R"))
source(experiment_path("R", "engine.R"))
source(experiment_path("R", "summaries.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--screen|--confirm] [options]\n\n",
  "Direct-FIML nested score-flip null-calibration stress test.\n\n",
  "Profiles:\n",
  "  --smoke          8 cells, 2 reps, 19 flips (default).\n",
  "  --screen         240 cells, 500 reps, 199 flips.\n",
  "  --confirm        11 difficult cells, 2000 reps, 999 flips.\n\n",
  "Run controls:\n",
  "  --reps N --flips N --chunk-size N --cores N --seed-base N\n",
  "  --results-dir P --shard-index J --shard-count K\n\n",
  "Cell filters (comma-separated):\n",
  "  --distributions normal,vm,ig,pl\n",
  "  --n-values 50,100,200,400\n",
  "  --missingness complete,mcar15,mcar30,mar30,strong_mar30\n",
  "  --ranks 1,3,5 --max-base-cells N\n\n",
  "One configural fit is reused across every selected rank for a base cell.\n",
  "Chunks resume only when the design, run configuration, package version,\n",
  "and experiment code hash match. Rejection uses p <= .05 and also records\n",
  "the p < .05 sensitivity.\n", sep = "")

parse_args <- function(args) {
  out <- list(
    profile = "smoke", reps = NULL, flips = NULL, chunk_size = NULL,
    cores = max(1L, parallel::detectCores() - 1L), seed_base = NULL,
    results_dir = NULL, shard_index = 1L, shard_count = 1L,
    distributions = NULL, n_values = NULL, missingness = NULL, ranks = NULL,
    max_base_cells = NULL)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]],
                               call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
    else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--screen") out$profile <- "screen"
    else if (a == "--confirm") out$profile <- "confirm"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--flips") out$flips <- as.integer(take())
    else if (a == "--chunk-size") out$chunk_size <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--shard-index") out$shard_index <- as.integer(take())
    else if (a == "--shard-count") out$shard_count <- as.integer(take())
    else if (a == "--distributions") out$distributions <- parse_csv_arg(take())
    else if (a == "--n-values")
      out$n_values <- as.integer(parse_csv_numeric(take()))
    else if (a == "--missingness") out$missingness <- parse_csv_arg(take())
    else if (a == "--ranks") out$ranks <- as.integer(parse_csv_numeric(take()))
    else if (a == "--max-base-cells") out$max_base_cells <- as.integer(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(out$profile,
    smoke = c(reps = 2L, flips = 19L, chunk_size = 1L),
    screen = c(reps = 500L, flips = 199L, chunk_size = 25L),
    confirm = c(reps = 2000L, flips = 999L, chunk_size = 25L))
  for (name in names(defaults)) if (is.null(out[[name]]))
    out[[name]] <- defaults[[name]]
  if (is.null(out$seed_base)) out$seed_base <- if (out$profile == "confirm")
    20261714 else 20260714
  positive <- c("reps", "flips", "chunk_size", "cores", "shard_index",
                "shard_count")
  if (any(vapply(out[positive], function(x)
    length(x) != 1L || is.na(x) || x < 1, logical(1L)))) {
    stop("run sizes, cores, and shard coordinates must be positive",
         call. = FALSE)
  }
  if (out$shard_index > out$shard_count)
    stop("--shard-index must be in 1..--shard-count", call. = FALSE)
  out
}

filter_design <- function(grid, cfg) {
  filters <- list(distribution = cfg$distributions, n_group1 = cfg$n_values,
                  missingness = cfg$missingness, rank = cfg$ranks)
  for (field in names(filters)) if (!is.null(filters[[field]]))
    grid <- grid[grid[[field]] %in% filters[[field]], ]
  if (!nrow(grid)) stop("cell filters selected no rows", call. = FALSE)
  bases <- stress_base_design(grid)
  shard_keep <- (seq_len(nrow(bases)) - 1L) %% cfg$shard_count + 1L ==
    cfg$shard_index
  bases <- bases[shard_keep, ]
  if (!is.null(cfg$max_base_cells)) bases <- head(bases, cfg$max_base_cells)
  grid <- grid[grid$base_id %in% bases$base_id, ]
  if (!nrow(grid)) stop("sharding selected no rows", call. = FALSE)
  row.names(grid) <- NULL
  grid
}

manifest_signature <- function(x) {
  x[] <- lapply(x, as.character)
  unname(apply(x, 1L, paste, collapse = "|"))
}

write_or_validate_manifest <- function(grid, cfg, path) {
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

completed_replications <- function(raw_dir, cell_ids) {
  files <- list.files(raw_dir, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$",
                      full.names = TRUE)
  if (!length(files)) return(integer())
  chunks <- lapply(files, read.csv, stringsAsFactors = FALSE,
                   check.names = FALSE)
  old <- do.call(rbind, chunks)
  pieces <- split(old$cell_id, old$rep)
  as.integer(names(Filter(function(ids)
    identical(sort(unique(ids)), sort(cell_ids)), pieces)))
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
results <- cfg$results_dir %||% experiment_path("results", cfg$profile)
dir.create(results, recursive = TRUE, showWarnings = FALSE)
stress_validate_design()
grid <- filter_design(stress_design(cfg$profile), cfg)
bases <- stress_base_design(grid)
write_or_validate_manifest(grid, cfg, results)
write_csv(grid, file.path(results, "design.csv"))
write_csv(stress_method_table(), file.path(results, "methods.csv"))

specs <- stress_model_specs(sort(unique(grid$rank)))
population <- stress_population()
sampler_cache <- new.env(parent = emptyenv())
remaining_at_start <- 0L
for (k in seq_len(nrow(bases))) {
  base <- bases[k, ]
  cells <- grid[grid$base_id == base$base_id, ]
  raw_dir <- file.path(results, "raw", sprintf("base_%04d", base$base_id))
  remaining_at_start <- remaining_at_start + length(setdiff(
    seq_len(cfg$reps), completed_replications(raw_dir, cells$cell_id)))
}

progress_begin <- proc.time()[["elapsed"]]
completed_now <- 0L
cat(sprintf(
  "profile=%s cells=%d bases=%d reps=%d flips=%d chunk=%d cores=%d shard=%d/%d\n",
  cfg$profile, nrow(grid), nrow(bases), cfg$reps, cfg$flips, cfg$chunk_size,
  cfg$cores, cfg$shard_index, cfg$shard_count))
cat(sprintf("new base-replication jobs=%d; H1 fits saved by rank reuse=%d\n",
            remaining_at_start,
            as.integer(round(remaining_at_start *
                             (mean(table(grid$base_id)) - 1)))))

run_base <- function(base, position) {
  cells <- grid[grid$base_id == base$base_id, ]
  raw_dir <- file.path(results, "raw", sprintf("base_%04d", base$base_id))
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  completed <- completed_replications(raw_dir, cells$cell_id)
  remaining <- setdiff(seq_len(cfg$reps), completed)
  if (!length(remaining)) {
    message(sprintf("base %d/%d (stable %d): already complete",
                    position, nrow(bases), base$base_id))
    return(invisible(0L))
  }

  sampler_key <- base$distribution
  cache_hit <- exists(sampler_key, envir = sampler_cache, inherits = FALSE)
  if (cache_hit) {
    sampler <- get(sampler_key, envir = sampler_cache, inherits = FALSE)
  } else {
    sampler <- tryCatch(stress_calibrate_sampler(population, base$distribution),
                        error = function(e) e)
    assign(sampler_key, sampler, envir = sampler_cache)
  }
  setup_seconds <- if (inherits(sampler, "error")) NA_real_ else
    if (cache_hit) 0 else sampler$setup_seconds
  distribution_index <- match(base$distribution, stress_distributions)
  missing_index <- match(base$missingness, stress_missingness)
  chunks <- split(remaining, ceiling(seq_along(remaining) / cfg$chunk_size))
  message(sprintf("base %d/%d (stable %d): %s n1=%d %s; %d ranks",
                  position, nrow(bases), base$base_id, base$distribution,
                  base$n_group1, base$missingness, nrow(cells)))
  for (rep_ids in chunks) {
    worker <- function(rep_id) {
      if (inherits(sampler, "error")) {
        rows <- lapply(seq_len(nrow(cells)), function(k)
          stress_empty_replication(as.list(cells[k, , drop = FALSE]), rep_id,
            paste0("simulation calibration: ", conditionMessage(sampler))))
      } else {
        draw_seed <- cfg$seed_base + distribution_index * 100000003 +
          base$n_group1 * 10007 + rep_id * 1009
        missing_seed <- cfg$seed_base + distribution_index * 10000019 +
          base$n_group1 * 100003 + missing_index * 1000003 + rep_id * 1013
        flip_seed <- cfg$seed_base + 700000001 + rep_id * 1019
        rows <- tryCatch(stress_one_rep(
          as.list(base), cells, rep_id, sampler, specs, cfg$flips,
          draw_seed, missing_seed, flip_seed), error = function(e)
            lapply(seq_len(nrow(cells)), function(k)
              stress_empty_replication(
                as.list(cells[k, , drop = FALSE]), rep_id,
                paste0("replication: ", conditionMessage(e)))))
      }
      lapply(rows, function(out) {
        out$generator_setup_seconds <- setup_seconds
        out$unique_group_calibrations <- if (inherits(sampler, "error"))
          NA_integer_ else sampler$unique_group_calibrations
        out
      })
    }
    pieces <- if (.Platform$OS.type != "windows" && cfg$cores > 1L &&
                  length(rep_ids) > 1L) {
      parallel::mclapply(
        rep_ids, worker, mc.cores = min(cfg$cores, length(rep_ids)),
        mc.preschedule = TRUE, mc.set.seed = FALSE)
    } else lapply(rep_ids, worker)
    rows <- do.call(rbind, unlist(pieces, recursive = FALSE))
    checkpoint <- file.path(raw_dir, sprintf(
      "chunk_%05d_%05d.csv", min(rep_ids), max(rep_ids)))
    temporary <- tempfile("checkpoint-", tmpdir = raw_dir, fileext = ".csv")
    write_csv(rows, temporary)
    if (!file.rename(temporary, checkpoint)) {
      unlink(temporary)
      stop("could not atomically publish checkpoint ", checkpoint,
           call. = FALSE)
    }
    completed_now <<- completed_now + length(rep_ids)
    elapsed <- proc.time()[["elapsed"]] - progress_begin
    eta <- if (completed_now > 0L) elapsed / completed_now *
      (remaining_at_start - completed_now) else NA_real_
    base_times <- rows$replication_seconds[!duplicated(rows$rep)]
    message(sprintf(
      "  checkpoint %d-%d: fit %d/%d rows; median %.2fs/base-rep; elapsed %.1fs; ETA %.1fs",
      min(rep_ids), max(rep_ids), sum(rows$fit_ok), nrow(rows),
      stats::median(base_times, na.rm = TRUE), elapsed, eta))
  }
  invisible(length(remaining))
}

for (k in seq_len(nrow(bases))) run_base(as.list(bases[k, , drop = FALSE]), k)

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

pvalues <- raw[c(stress_design_columns, "rep", stress_p_columns)]
write_csv(pvalues, file.path(results, "pvalues.csv"))
method_rows <- stress_method_rows(raw)
method_summary <- stress_summarize_methods(method_rows)
paired_summary <- stress_summarize_paired(raw)
write_csv(method_summary, file.path(results, "method_summary.csv"))
write_csv(stress_calibration_summary(method_summary),
          file.path(results, "calibration_summary.csv"))
write_csv(stress_availability_summary(method_rows),
          file.path(results, "availability_summary.csv"))
write_csv(paired_summary, file.path(results, "paired_summary.csv"))
write_csv(stress_standardization_summary(paired_summary),
          file.path(results, "standardization_summary.csv"))
write_csv(stress_summarize_timing(raw),
          file.path(results, "timing_summary.csv"))
write_csv(stress_summarize_base_timing(raw),
          file.path(results, "base_timing_summary.csv"))
write_csv(stress_confirmation_candidates(method_summary, paired_summary),
          file.path(results, "confirmation_candidates.csv"))
failures <- raw[!raw$fit_ok | !raw$flip_ok | !raw$nested_ok,
  c(stress_design_columns, "rep", "fit_ok", "flip_ok", "nested_ok",
    "fit_error", "flip_error", "nested_error")]
write_csv(failures, file.path(results, "failures.csv"))
elapsed <- proc.time()[["elapsed"]] - progress_begin
write_metadata(file.path(results, "metadata.csv"), list(
  profile = cfg$profile, cells = nrow(grid), base_cells = nrow(bases),
  reps = cfg$reps, flips = cfg$flips, chunk_size = cfg$chunk_size,
  cores = cfg$cores, seed_base = cfg$seed_base,
  shard_index = cfg$shard_index, shard_count = cfg$shard_count,
  elapsed_seconds = elapsed, new_base_replications = remaining_at_start,
  h1_fit_reuse = "one configural fit per base replication across ranks",
  alpha_convention = "p <= 0.05", strict_sensitivity = "p < 0.05",
  target_skewness = 3, target_excess_kurtosis = 21,
  missing_rate_denominator = "eligible variables x3:x6",
  paper_mar = "calibrated Savalei-Bentler 2005 rules; x1/x2 observed",
  strong_mar = "item-specific logistic selection on observed x1/x2; calibrated to 30%"),
  packages = "magmaan")

if (cfg$profile == "smoke") {
  stopifnot(nrow(raw) == expected, all(raw$fit_ok), all(raw$flip_ok),
            all(raw$nested_ok), all(is.finite(raw$p_flip_standardized)),
            all(raw$rank == raw$df), all(raw$h1_fit_count == 1L),
            nrow(failures) == 0L)
  message("SMOKE PASS")
}
cat(sprintf("wrote %s in %.1fs (%d pipeline failures)\n",
            normalizePath(results), elapsed, nrow(failures)))
