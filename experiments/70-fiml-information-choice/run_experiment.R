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
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
source(experiment_path("R", "design.R"))
source(experiment_path("R", "engine.R"))
source(experiment_path("R", "summaries.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--overnight] [options]\n\n",
  "Focused FIML information-choice simulation. One magmaan fit is paired with\n",
  "expected Fisher, observed-H1, and full observed-Hessian breads; each gets\n",
  "both a model-based and same-meat empirical sandwich covariance.\n\n",
  "Profiles:\n",
  "  --smoke       4 sentinel cells x 2 reps (default)\n",
  "  --pilot       12 spread cells x 50 reps\n",
  "  --overnight   all 48 cells x 200000 reps\n\n",
  "Options:\n",
  "  --reps N --chunk-size N --cores N --seed-base N\n",
  "  --results-dir PATH --cells L --max-cells N\n",
  "  --lavaan-parity --lavaan-max-cells N\n\n",
  "Chunks are deterministic, atomically checkpointed, and resumed only when\n",
  "the manifest, run settings, package version, and code hashes agree.\n",
  sep = "")

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    reps = NULL,
    chunk_size = NULL,
    cores = max(1L, min(8L, parallel::detectCores() - 1L)),
    seed_base = 20260717,
    results_dir = NULL,
    cells = NULL,
    max_cells = NULL,
    lavaan_parity = FALSE,
    lavaan_max_cells = 3L
  )
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--pilot") out$profile <- "pilot"
    else if (a == "--overnight") out$profile <- "overnight"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--chunk-size") out$chunk_size <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--cells")
      out$cells <- as.integer(parse_csv_arg(take()))
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else if (a == "--lavaan-parity") out$lavaan_parity <- TRUE
    else if (a == "--lavaan-max-cells")
      out$lavaan_max_cells <- as.integer(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$profile,
    smoke = c(reps = 2L, chunk_size = 1L),
    pilot = c(reps = 50L, chunk_size = 10L),
    overnight = c(reps = 200000L, chunk_size = 2000L)
  )
  for (name in names(defaults)) {
    if (is.null(out[[name]])) out[[name]] <- defaults[[name]]
  }
  positive <- c("reps", "chunk_size", "cores", "lavaan_max_cells")
  if (any(vapply(out[positive], function(x)
    length(x) != 1L || is.na(x) || x < 1L, logical(1)))) {
    stop("replications, chunks, cores, and sentinel counts must be positive",
         call. = FALSE)
  }
  if (!is.null(out$max_cells) &&
      (length(out$max_cells) != 1L ||
       is.na(out$max_cells) || out$max_cells < 1L)) {
    stop("--max-cells must be positive", call. = FALSE)
  }
  if (length(out$seed_base) != 1L || !is.finite(out$seed_base) ||
      out$seed_base < 1) {
    stop("--seed-base must be one positive finite number", call. = FALSE)
  }
  out
}

atomic_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile("checkpoint-", tmpdir = dirname(path), fileext = ".csv")
  write_csv(x, temporary)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop("could not atomically publish ", path, call. = FALSE)
  }
}

task_path <- function(checkpoint_dir, task) {
  file.path(checkpoint_dir, sprintf(
    "cell_%03d_rep_%06d_%06d.csv",
    task$cell_id, task$rep_start, task$rep_end))
}

manifest_signature <- function(x) {
  x[] <- lapply(x, as.character)
  unname(apply(x, 1L, paste, collapse = "|"))
}

write_or_validate_manifest <- function(grid, cfg, results) {
  code_files <- c(
    experiment_path("run_experiment.R"),
    experiment_path("R", "design.R"),
    experiment_path("R", "engine.R"),
    experiment_path("R", "summaries.R")
  )
  config <- data.frame(
    schema_version = 1L,
    profile = cfg$profile,
    reps = cfg$reps,
    chunk_size = cfg$chunk_size,
    seed_base = cfg$seed_base,
    lavaan_parity = cfg$lavaan_parity,
    lavaan_max_cells = cfg$lavaan_max_cells,
    magmaan_version = as.character(packageVersion("magmaan")),
    code_hash = paste(unname(tools::md5sum(code_files)), collapse = ":"),
    stringsAsFactors = FALSE
  )
  manifest_path <- file.path(results, "manifest.csv")
  config_path <- file.path(results, "run_config.csv")
  if (file.exists(manifest_path) || file.exists(config_path)) {
    if (!file.exists(manifest_path) || !file.exists(config_path)) {
      stop("existing run metadata is incomplete in ", results, call. = FALSE)
    }
    old_grid <- read.csv(manifest_path, stringsAsFactors = FALSE)
    old_config <- read.csv(config_path, stringsAsFactors = FALSE)
    same_grid <- identical(
      manifest_signature(old_grid), manifest_signature(grid))
    same_config <- identical(
      unname(as.character(old_config[1L, names(config)])),
      unname(as.character(config[1L, ])))
    if (!same_grid || !same_config) {
      stop("existing output has different settings or code; choose a new ",
           "--results-dir", call. = FALSE)
    }
  } else {
    write_csv(grid, manifest_path)
    write_csv(config, config_path)
  }
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
validate_fiml_information_design()
grid <- fiml_information_design(cfg$profile)
if (!is.null(cfg$cells)) {
  grid <- grid[grid$cell_id %in% cfg$cells, , drop = FALSE]
}
if (!is.null(cfg$max_cells)) grid <- head(grid, cfg$max_cells)
if (!nrow(grid)) stop("cell filters selected no cells", call. = FALSE)

results <- cfg$results_dir %||%
  experiment_path("results", cfg$profile)
dir.create(results, recursive = TRUE, showWarnings = FALSE)
checkpoint_dir <- file.path(results, "checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
write_or_validate_manifest(grid, cfg, results)
write_csv(grid, file.path(results, "design.csv"))
write_csv(fiml_information_methods, file.path(results, "methods.csv"))

if (cfg$lavaan_parity) {
  require_pkg("lavaan")
  cat("Running lavaan sentinel before the simulation...\n")
  parity <- fiml_lavaan_parity(
    grid, cfg$seed_base, max_cells = cfg$lavaan_max_cells)
  write_csv(parity, file.path(results, "lavaan_parity.csv"))
  if (!all(parity$pass)) {
    stop("lavaan sentinel failed; inspect lavaan_parity.csv", call. = FALSE)
  }
  cat(sprintf(
    "lavaan sentinel passed %d checks; max |delta SE| = %.4g\n",
    nrow(parity), max(parity$abs_difference)))
} else {
  write_csv(data.frame(), file.path(results, "lavaan_parity.csv"))
}

tasks <- do.call(rbind, lapply(grid$cell_id, function(cell_id) {
  starts <- seq.int(1L, cfg$reps, by = cfg$chunk_size)
  data.frame(
    cell_id = cell_id,
    rep_start = starts,
    rep_end = pmin(starts + cfg$chunk_size - 1L, cfg$reps))
}))
tasks$path <- vapply(seq_len(nrow(tasks)), function(i)
  task_path(checkpoint_dir, tasks[i, ]), character(1))
pending <- tasks[!file.exists(tasks$path), , drop = FALSE]
pending_replications <- if (nrow(pending)) {
  sum(pending$rep_end - pending$rep_start + 1L)
} else {
  0L
}

cat(sprintf(
  "%s profile: %d cells x %d reps = %d fits; %d/%d chunks pending on %d cores\n",
  cfg$profile, nrow(grid), cfg$reps, nrow(grid) * cfg$reps,
  nrow(pending), nrow(tasks), cfg$cores))

run_started <- proc.time()[["elapsed"]]
done_initial <- nrow(tasks) - nrow(pending)
if (nrow(pending)) {
  waves <- split(seq_len(nrow(pending)),
                 ceiling(seq_len(nrow(pending)) / cfg$cores))
  completed_now <- 0L
  for (wave in waves) {
    batch <- split(pending[wave, , drop = FALSE], seq_along(wave))
    runner <- function(task)
      fiml_run_chunk(task, grid, cfg$seed_base)
    outputs <- if (cfg$cores > 1L && length(batch) > 1L) {
      parallel::mclapply(
        batch, runner, mc.cores = min(cfg$cores, length(batch)),
        mc.preschedule = FALSE, mc.set.seed = FALSE)
    } else {
      lapply(batch, runner)
    }
    for (j in seq_along(batch)) {
      atomic_csv(outputs[[j]], batch[[j]]$path)
    }
    completed_now <- completed_now + length(batch)
    elapsed <- proc.time()[["elapsed"]] - run_started
    rate <- completed_now / max(elapsed, 1e-9)
    remaining <- nrow(pending) - completed_now
    eta_minutes <- remaining / max(rate, 1e-9) / 60
    cat(sprintf(
      "[%s] chunks %d/%d complete (%.1f%%); ETA %.1f min\n",
      format(Sys.time(), "%H:%M:%S"),
      done_initial + completed_now, nrow(tasks),
      100 * (done_initial + completed_now) / nrow(tasks), eta_minutes))
  }
}
wall_seconds <- proc.time()[["elapsed"]] - run_started

checkpoint_files <- tasks$path
if (!all(file.exists(checkpoint_files))) {
  stop("not all expected checkpoints exist after the run", call. = FALSE)
}
summary_parts <- list(
  cell_method = list(), timing = list(), failures = list(),
  warnings = list(), information_failures = list())
keep_combined_raw <- nrow(grid) * cfg$reps <= 100000L
combined_raw <- if (keep_combined_raw) list() else NULL
fit_failures <- 0L
comparison_failures <- 0L
warning_count <- 0L
worker_seconds <- 0
worker_replications <- 0L

for (i in seq_len(nrow(grid))) {
  cell_id <- grid$cell_id[[i]]
  paths <- checkpoint_files[tasks$cell_id == cell_id]
  raw_cell <- do.call(rbind, lapply(paths, function(path)
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)))
  raw_cell$error[is.na(raw_cell$error)] <- ""
  raw_cell$warning[is.na(raw_cell$warning)] <- ""
  if (keep_combined_raw) combined_raw[[i]] <- raw_cell

  cell_summary <- fiml_information_summaries(
    raw_cell, grid[grid$cell_id == cell_id, , drop = FALSE])
  for (name in names(summary_parts)) {
    summary_parts[[name]][[i]] <- cell_summary[[name]]
  }

  replication <- raw_cell[
    !duplicated(raw_cell[c("cell_id", "rep")]), , drop = FALSE]
  fit_failures <- fit_failures + sum(!replication$fit_ok)
  comparison_failures <- comparison_failures +
    sum(replication$fit_ok & !replication$comparison_ok)
  warning_count <- warning_count +
    sum(nzchar(replication$warning))
  seconds <- replication$fit_seconds + replication$inference_seconds
  worker_seconds <- worker_seconds + sum(seconds, na.rm = TRUE)
  worker_replications <- worker_replications + sum(is.finite(seconds))
  rm(raw_cell, replication, cell_summary)
  if (i %% 8L == 0L) gc(verbose = FALSE)
}

summaries <- lapply(summary_parts, function(parts)
  do.call(rbind, parts))
if (keep_combined_raw) {
  atomic_csv(
    do.call(rbind, combined_raw),
    file.path(results, "replications.csv"))
}
atomic_csv(summaries$cell_method, file.path(results, "cell_method_summary.csv"))
atomic_csv(summaries$timing, file.path(results, "timing_summary.csv"))
atomic_csv(summaries$failures, file.path(results, "failures.csv"))
atomic_csv(summaries$warnings, file.path(results, "warnings.csv"))
atomic_csv(
  summaries$information_failures,
  file.path(results, "information_failures.csv"))

mean_rep_seconds <- worker_seconds / worker_replications
target_replications <- 48 * 200000
observed_projected_hours <- if (pending_replications > 0L) {
  if (pending_replications >= 10000L) {
    wall_seconds / pending_replications * target_replications / 3600
  } else {
    NA_real_
  }
} else {
  NA_real_
}
projection <- data.frame(
  assumed_cores = cfg$cores,
  mean_worker_seconds_per_rep = mean_rep_seconds,
  overnight_cells = 48L,
  overnight_reps_per_cell = 200000L,
  ideal_hours = mean_rep_seconds * target_replications / cfg$cores / 3600,
  observed_throughput_projection_hours = observed_projected_hours,
  projected_hours_with_15pct_overhead =
    1.15 * if (is.finite(observed_projected_hours))
      observed_projected_hours else
      mean_rep_seconds * target_replications / cfg$cores / 3600
)
atomic_csv(projection, file.path(results, "runtime_projection.csv"))

write_metadata(
  file.path(results, "metadata.csv"),
  values = list(
    profile = cfg$profile,
    cells = nrow(grid),
    reps_per_cell = cfg$reps,
    total_replications = nrow(grid) * cfg$reps,
    chunk_size = cfg$chunk_size,
    cores = cfg$cores,
    seed_base = cfg$seed_base,
    wall_seconds_this_invocation = wall_seconds,
    raw_output = if (keep_combined_raw)
      "checkpoints plus combined replications.csv" else
      "per-cell chunk checkpoints; summaries aggregated one cell at a time",
    fit_failures = fit_failures,
    comparison_failures = comparison_failures,
    warnings = warning_count,
    lavaan_parity = cfg$lavaan_parity,
    scope = paste(
      "continuous two-factor CFA; normal/t5; correct/omitted cross-loading;",
      "complete/MCAR30/MAR30; N=150/300/600/1200")
  ),
  packages = c("magmaan", if (cfg$lavaan_parity) "lavaan"))

cat(sprintf(
  "Completed %d replications: %d fit failures, %d information failures.\n",
  nrow(grid) * cfg$reps, fit_failures, comparison_failures))
cat(sprintf(
  "Measured %.3f worker-seconds/rep; 48 x 200000 projects to %.2f h on %d cores ",
  mean_rep_seconds, projection$projected_hours_with_15pct_overhead, cfg$cores))
cat("(15% overhead allowance).\n")
cat("Results: ", normalizePath(results), "\n", sep = "")
