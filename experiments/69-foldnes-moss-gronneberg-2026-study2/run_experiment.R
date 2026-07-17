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
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Native magmaan replication of Foldnes, Moss and Gronneberg (2026),\n",
  "Study 2 weak-invariance Type-I calibration. The primary run uses the\n",
  "Satorra-2000/delta restriction map used by semTests; method 2001 is out of\n",
  "scope. All 38 Study 2 method variants include biased/unbiased Gamma and\n",
  "ML/RLS bases.\n\n",
  "Profiles:\n",
  "  --smoke       4 structural cells x 1 rep (default).\n",
  "  --pilot       21 cells x 200 reps: all distributions at low/mid/high corners.\n",
  "  --full        all 189 cells x 2000 reps (the published design).\n\n",
  "Run controls:\n",
  "  --reps N --chunk-size N --cores N --seed-base N\n",
  "  --results-dir P --shard-index J --shard-count K\n\n",
  "Cell filters (comma-separated):\n",
  "  --cells L --p L --groups L --n-group L --distributions L --max-cells N\n\n",
  "Sanity oracle:\n",
  "  --semtests-parity       Compare rep 1 in a spread of selected cells.\n",
  "  --semtests-max-cells N  Maximum sentinel cells (default 3).\n\n",
  "Chunks are atomic and resumable when the manifest and code hash match.\n",
  "Use a distinct --results-dir when changing any run setting.\n", sep = "")

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    reps = NULL,
    chunk_size = NULL,
    cores = max(1L, parallel::detectCores() - 1L),
    seed_base = 20260717,
    results_dir = NULL,
    shard_index = 1L,
    shard_count = 1L,
    cells = NULL,
    p = NULL,
    groups = NULL,
    n_group = NULL,
    distributions = NULL,
    max_cells = NULL,
    semtests_parity = FALSE,
    semtests_max_cells = 3L
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
    else if (a == "--full") out$profile <- "full"
    else if (a == "--reps") out$reps <- as.integer(take())
    else if (a == "--chunk-size") out$chunk_size <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--seed-base") out$seed_base <- as.numeric(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else if (a == "--shard-index") out$shard_index <- as.integer(take())
    else if (a == "--shard-count") out$shard_count <- as.integer(take())
    else if (a == "--cells") out$cells <- as.integer(parse_csv_arg(take()))
    else if (a == "--p") out$p <- as.integer(parse_csv_arg(take()))
    else if (a == "--groups") out$groups <- as.integer(parse_csv_arg(take()))
    else if (a == "--n-group") out$n_group <- as.integer(parse_csv_arg(take()))
    else if (a == "--distributions")
      out$distributions <- parse_csv_arg(take())
    else if (a == "--max-cells") out$max_cells <- as.integer(take())
    else if (a == "--semtests-parity") out$semtests_parity <- TRUE
    else if (a == "--semtests-max-cells")
      out$semtests_max_cells <- as.integer(take())
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$profile,
    smoke = c(reps = 1L, chunk_size = 1L),
    pilot = c(reps = 200L, chunk_size = 20L),
    full = c(reps = 2000L, chunk_size = 25L)
  )
  for (name in names(defaults)) {
    if (is.null(out[[name]])) out[[name]] <- defaults[[name]]
  }
  positive <- c("reps", "chunk_size", "cores", "shard_index", "shard_count",
                "semtests_max_cells")
  if (any(vapply(out[positive], function(x) {
    length(x) != 1L || is.na(x) || x < 1
  }, logical(1)))) {
    stop("run sizes, cores, and shard coordinates must be positive",
         call. = FALSE)
  }
  if (out$shard_index > out$shard_count) {
    stop("--shard-index must be in 1..--shard-count", call. = FALSE)
  }
  bad_distribution <- setdiff(out$distributions %||% character(),
                              study2_distributions)
  if (length(bad_distribution)) {
    stop("unknown distributions: ", paste(bad_distribution, collapse = ", "),
         call. = FALSE)
  }
  out
}

filter_design <- function(grid, cfg) {
  filters <- c("cells", "p", "groups", "n_group")
  for (name in filters) {
    field <- if (name == "cells") "cell_id" else name
    if (!is.null(cfg[[name]])) {
      grid <- grid[grid[[field]] %in% cfg[[name]], , drop = FALSE]
    }
  }
  if (!is.null(cfg$distributions)) {
    grid <- grid[
      grid$distribution %in% cfg$distributions, , drop = FALSE]
  }
  if (!nrow(grid)) stop("cell filters selected no rows", call. = FALSE)
  grid <- grid[
    (seq_len(nrow(grid)) - 1L) %% cfg$shard_count + 1L ==
      cfg$shard_index, , drop = FALSE]
  if (!is.null(cfg$max_cells)) grid <- head(grid, cfg$max_cells)
  if (!nrow(grid)) stop("shard/filter combination selected no rows",
                        call. = FALSE)
  row.names(grid) <- NULL
  grid
}

manifest_signature <- function(x) {
  x[] <- lapply(x, as.character)
  unname(apply(x, 1L, paste, collapse = "|"))
}

write_or_validate_manifest <- function(grid, cfg, results) {
  manifest_path <- file.path(results, "manifest.csv")
  config_path <- file.path(results, "run_config.csv")
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
    shard_index = cfg$shard_index,
    shard_count = cfg$shard_count,
    semtests_parity = cfg$semtests_parity,
    semtests_max_cells = cfg$semtests_max_cells,
    magmaan_version = as.character(packageVersion("magmaan")),
    code_hash = paste(unname(tools::md5sum(code_files)), collapse = ":"),
    stringsAsFactors = FALSE
  )
  if (file.exists(manifest_path) || file.exists(config_path)) {
    if (!file.exists(manifest_path) || !file.exists(config_path)) {
      stop("incomplete existing run metadata in ", results, call. = FALSE)
    }
    old_grid <- read.csv(manifest_path, stringsAsFactors = FALSE,
                         check.names = FALSE)
    old_config <- read.csv(config_path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    if (!identical(manifest_signature(old_grid), manifest_signature(grid)) ||
        !identical(
          unname(as.character(old_config[1L, names(config)])),
          unname(as.character(config[1L, ])))) {
      stop("existing output configuration differs; choose a new --results-dir: ",
           results, call. = FALSE)
    }
  } else {
    write_csv(grid, manifest_path)
    write_csv(config, config_path)
  }
}

completed_replications <- function(raw_dir) {
  files <- list.files(
    raw_dir, pattern = "^chunk_[0-9]+_[0-9]+\\.csv$", full.names = TRUE)
  if (!length(files)) return(integer())
  unique(unlist(lapply(files, function(path) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)$rep
  }), use.names = FALSE))
}

atomic_checkpoint <- function(rows, path) {
  temporary <- tempfile("checkpoint-", tmpdir = dirname(path), fileext = ".csv")
  write_csv(rows, temporary)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop("could not atomically publish checkpoint ", path, call. = FALSE)
  }
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
results <- cfg$results_dir %||% experiment_path("results", cfg$profile)
dir.create(results, recursive = TRUE, showWarnings = FALSE)
study2_validate_design()
grid <- filter_design(study2_design(cfg$profile), cfg)
write_or_validate_manifest(grid, cfg, results)
write_csv(grid, file.path(results, "design.csv"))
write_csv(study2_method_table(), file.path(results, "methods.csv"))

if (cfg$semtests_parity) {
  require_pkg("lavaan")
  require_pkg("semTests")
}
parity_cells <- integer()
if (cfg$semtests_parity) {
  positions <- unique(as.integer(round(seq(
    1L, nrow(grid), length.out = min(cfg$semtests_max_cells, nrow(grid))))))
  parity_cells <- grid$cell_id[positions]
}

sampler_cache <- new.env(parent = emptyenv())
spec_cache <- new.env(parent = emptyenv())
generator_calibrations <- list()
cache_ref <- magmaan_cache_ref()

get_specs <- function(p, groups) {
  key <- paste(p, groups, sep = "_")
  if (!exists(key, envir = spec_cache, inherits = FALSE)) {
    assign(key, study2_model_specs(p, groups), envir = spec_cache)
  }
  get(key, envir = spec_cache, inherits = FALSE)
}

get_sampler <- function(p, distribution) {
  key <- paste(p, distribution, sep = "_")
  if (exists(key, envir = sampler_cache, inherits = FALSE)) {
    out <- get(key, envir = sampler_cache, inherits = FALSE)
    out$memory_cache_hit <- TRUE
    return(out)
  }
  population <- study2_population(p)
  moments <- study2_distribution_moments(distribution, p)
  disk_key <- calibration_cache_key(
    population = population$Sigma,
    generator = distribution,
    options = moments,
    ref = cache_ref
  )
  disk_hit <- calibration_cache_exists(
    disk_key, cache_dir = file.path(results, "cache"))
  if (disk_hit) {
    sampler <- calibration_cache_read(
      disk_key, cache_dir = file.path(results, "cache"))
  } else {
    sampler <- study2_calibrate_sampler(population, distribution)
    calibration_cache_write(
      disk_key, sampler,
      cache_dir = file.path(results, "cache"),
      metadata = list(p = p, distribution = distribution)
    )
  }
  sampler$disk_cache_hit <- disk_hit
  sampler$memory_cache_hit <- FALSE
  assign(key, sampler, envir = sampler_cache)
  sampler
}

record_sampler_diagnostics <- function(cell, sampler) {
  key <- paste(cell$p, cell$distribution, sep = "_")
  if (!is.null(generator_calibrations[[key]])) return(invisible(NULL))
  generator_calibrations[[key]] <<- if (inherits(sampler, "error")) {
    moments <- study2_distribution_moments(cell$distribution, cell$p)
    data.frame(
      p = cell$p,
      distribution = cell$distribution,
      generator = sub("[12]$", "", cell$distribution),
      target_skewness = moments$skew[[1L]],
      target_excess_kurtosis = moments$exkurt[[1L]],
      setup_seconds = NA_real_,
      intermediate_min_eigen = NA_real_,
      calibration_max_abs_correlation_error = NA_real_,
      calibration_ok = FALSE,
      disk_cache_hit = FALSE,
      error = conditionMessage(sampler),
      stringsAsFactors = FALSE
    )
  } else {
    out <- study2_sampler_diagnostics(sampler)
    out$error <- ""
    out
  }
  invisible(NULL)
}

replication_seed <- function(cell_id, rep_id) {
  study2_seed(cfg$seed_base + cell_id * 1000003 + rep_id * 1009)
}

parity_dir <- file.path(results, "parity")
if (cfg$semtests_parity) dir.create(parity_dir, recursive = TRUE,
                                    showWarnings = FALSE)

total_jobs <- nrow(grid) * cfg$reps
completed_at_start <- 0L
for (k in seq_len(nrow(grid))) {
  raw_dir <- file.path(results, "raw", sprintf("cell_%04d", grid$cell_id[[k]]))
  completed_at_start <- completed_at_start +
    length(completed_replications(raw_dir))
}
new_jobs <- total_jobs - completed_at_start
completed_now <- 0L
wall_begin <- proc.time()[["elapsed"]]

cat(sprintf(
  "profile=%s cells=%d reps=%d methods=38 chunk=%d cores=%d shard=%d/%d\n",
  cfg$profile, nrow(grid), cfg$reps, cfg$chunk_size, cfg$cores,
  cfg$shard_index, cfg$shard_count))
cat(sprintf(
  "new replications=%d; semTests sentinel cells=%s\n",
  new_jobs,
  if (length(parity_cells)) paste(parity_cells, collapse = ",") else "none"))

for (k in seq_len(nrow(grid))) {
  cell <- as.list(grid[k, , drop = FALSE])
  raw_dir <- file.path(results, "raw", sprintf("cell_%04d", cell$cell_id))
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  completed <- completed_replications(raw_dir)
  remaining <- setdiff(seq_len(cfg$reps), completed)
  parity_path <- file.path(
    parity_dir, sprintf("cell_%04d.csv", cell$cell_id))
  needs_parity <- cell$cell_id %in% parity_cells && !file.exists(parity_path)
  if (!length(remaining) && !needs_parity) {
    sampler <- tryCatch(get_sampler(cell$p, cell$distribution),
                        error = function(e) e)
    record_sampler_diagnostics(cell, sampler)
    message(sprintf("cell %d/%d (stable %d): already complete",
                    k, nrow(grid), cell$cell_id))
    next
  }

  message(sprintf(
    "cell %d/%d (stable %d): p=%d G=%d n/G=%d %s df=%d",
    k, nrow(grid), cell$cell_id, cell$p, cell$groups, cell$n_group,
    cell$distribution, cell$df))
  sampler <- tryCatch(get_sampler(cell$p, cell$distribution),
                      error = function(e) e)
  record_sampler_diagnostics(cell, sampler)
  specs <- get_specs(cell$p, cell$groups)
  plan <- study2_test_plan(cell$df)

  if (needs_parity) {
    details <- if (inherits(sampler, "error")) NULL else tryCatch(
      study2_one_rep(
        cell, 1L, sampler, specs, replication_seed(cell$cell_id, 1L),
        return_details = TRUE),
      error = function(e) e
    )
    parity <- if (is.null(details) || inherits(details, "error") ||
                  is.null(details$nested)) {
      error <- if (inherits(sampler, "error")) {
        paste0("simulation calibration: ", conditionMessage(sampler))
      } else if (inherits(details, "error")) {
        conditionMessage(details)
      } else {
        details$row$fit_error %||% details$row$nested_error %||%
          "magmaan sentinel replication failed"
      }
      data.frame(
        cell_id = cell$cell_id, p = cell$p, groups = cell$groups,
        n_group = cell$n_group, df = cell$df,
        distribution = cell$distribution, rep = 1L,
        method_id = plan$method_id, input = plan$input,
        magmaan = NA_real_, semtests = NA_real_, abs_diff = NA_real_,
        tolerance = 5e-5, pass = FALSE, warnings = "", error = error,
        stringsAsFactors = FALSE
      )
    } else {
      study2_semtests_parity(cell, details)
    }
    write_csv(parity, parity_path)
    if (!inherits(details, "error") && !is.null(details$row) &&
        1L %in% remaining) {
      details$row$generator_setup_seconds <- if (inherits(sampler, "error")) {
        NA_real_
      } else sampler$setup_seconds
      details$row$generator_disk_cache_hit <- if (inherits(sampler, "error")) {
        NA
      } else sampler$disk_cache_hit
      details$row$generator_memory_cache_hit <- if (inherits(sampler, "error")) {
        NA
      } else sampler$memory_cache_hit
      checkpoint <- file.path(raw_dir, "chunk_00001_00001.csv")
      atomic_checkpoint(details$row, checkpoint)
      remaining <- setdiff(remaining, 1L)
      completed_now <- completed_now + 1L
    }
    comparable <- is.finite(parity$abs_diff)
    message(sprintf(
      "  semTests sentinel: %d/%d comparable pass; max |delta p| %.3g",
      sum(parity$pass[comparable]), sum(comparable),
      if (any(comparable)) max(parity$abs_diff[comparable]) else NA_real_))
  }

  chunks <- split(remaining, ceiling(seq_along(remaining) / cfg$chunk_size))
  for (rep_ids in chunks) {
    if (!length(rep_ids)) next
    worker <- function(rep_id) {
      if (inherits(sampler, "error")) {
        row <- study2_empty_replication(
          cell, rep_id, plan,
          paste0("simulation calibration: ", conditionMessage(sampler)))
      } else {
        row <- tryCatch(
          study2_one_rep(
            cell, rep_id, sampler, specs,
            replication_seed(cell$cell_id, rep_id)),
          error = function(e) study2_empty_replication(
            cell, rep_id, plan,
            paste0("replication: ", conditionMessage(e)))
        )
      }
      row$generator_setup_seconds <- if (inherits(sampler, "error")) {
        NA_real_
      } else sampler$setup_seconds
      row$generator_disk_cache_hit <- if (inherits(sampler, "error")) {
        NA
      } else sampler$disk_cache_hit
      row$generator_memory_cache_hit <- if (inherits(sampler, "error")) {
        NA
      } else sampler$memory_cache_hit
      row
    }
    rows <- if (.Platform$OS.type != "windows" && cfg$cores > 1L &&
                length(rep_ids) > 1L) {
      parallel::mclapply(
        rep_ids, worker,
        mc.cores = min(cfg$cores, length(rep_ids)),
        mc.preschedule = TRUE,
        mc.set.seed = FALSE
      )
    } else lapply(rep_ids, worker)
    rows <- do.call(rbind, rows)
    checkpoint <- file.path(
      raw_dir,
      sprintf("chunk_%05d_%05d.csv", min(rep_ids), max(rep_ids)))
    atomic_checkpoint(rows, checkpoint)
    completed_now <- completed_now + length(rep_ids)
    elapsed <- proc.time()[["elapsed"]] - wall_begin
    eta <- if (completed_now > 0L) {
      elapsed / completed_now * (new_jobs - completed_now)
    } else NA_real_
    message(sprintf(
      "  checkpoint %d-%d: nested %d/%d; median %.3fs/rep; elapsed %.1fs; ETA %.1fs",
      min(rep_ids), max(rep_ids), sum(rows$nested_ok), nrow(rows),
      stats::median(rows$replication_seconds, na.rm = TRUE), elapsed, eta))
  }
}

files <- list.files(
  file.path(results, "raw"),
  pattern = "^chunk_[0-9]+_[0-9]+\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)
if (!length(files)) stop("no replication checkpoints were written",
                         call. = FALSE)
raw <- do.call(rbind, lapply(
  files, read.csv, stringsAsFactors = FALSE, check.names = FALSE))
raw <- raw[
  raw$cell_id %in% grid$cell_id & raw$rep <= cfg$reps, , drop = FALSE]
raw <- raw[order(raw$cell_id, raw$rep), , drop = FALSE]
for (name in c("fit_error", "nested_error", "fit_warnings",
               "nested_warnings")) {
  raw[[name]][is.na(raw[[name]])] <- ""
}
if (anyDuplicated(raw[c("cell_id", "rep")])) {
  stop("duplicate checkpoint rows detected", call. = FALSE)
}
if (nrow(raw) != total_jobs) {
  stop("checkpoint set has ", nrow(raw), " rows; expected ", total_jobs,
       call. = FALSE)
}

method_rows <- study2_method_rows(raw)
cell_summary <- study2_cell_summary(method_rows)
calibration_summary <- study2_calibration_summary(cell_summary)
write_csv(
  raw[c("cell_id", "p", "groups", "n_group", "n_total", "df",
        "distribution", "rep",
        paste0("p_", study2_method_table()$method_id))],
  file.path(results, "pvalues.csv"))
write_csv(cell_summary, file.path(results, "cell_summary.csv"))
write_csv(calibration_summary, file.path(results, "calibration_summary.csv"))
write_csv(study2_generator_summary(raw),
          file.path(results, "generator_summary.csv"))
generator_calibration <- do.call(rbind, generator_calibrations)
row.names(generator_calibration) <- NULL
write_csv(generator_calibration,
          file.path(results, "generator_calibration.csv"))
write_csv(study2_timing_summary(raw), file.path(results, "timing_summary.csv"))

failures <- raw[
  !raw$fit_ok | !raw$nested_ok,
  c("cell_id", "p", "groups", "n_group", "df", "distribution", "rep",
    "fit_ok", "nested_ok", "fit_error", "nested_error",
    "fit_warnings", "nested_warnings"),
  drop = FALSE
]
warnings <- raw[
  nzchar(raw$fit_warnings) | nzchar(raw$nested_warnings),
  c("cell_id", "p", "groups", "n_group", "df", "distribution", "rep",
    "fit_warnings", "nested_warnings"),
  drop = FALSE
]
write_csv(failures, file.path(results, "failures.csv"))
write_csv(warnings, file.path(results, "warnings.csv"))

parity_files <- if (dir.exists(parity_dir)) {
  list.files(parity_dir, pattern = "^cell_[0-9]+\\.csv$", full.names = TRUE)
} else character()
parity <- if (length(parity_files)) {
  do.call(rbind, lapply(
    parity_files, read.csv, stringsAsFactors = FALSE, check.names = FALSE))
} else {
  data.frame(
    cell_id = integer(), p = integer(), groups = integer(),
    n_group = integer(), df = integer(), distribution = character(),
    rep = integer(), method_id = character(), input = character(),
    magmaan = numeric(), semtests = numeric(), abs_diff = numeric(),
    tolerance = numeric(), pass = logical(), warnings = character(),
    error = character(), stringsAsFactors = FALSE
  )
}
write_csv(parity, file.path(results, "semtests_parity.csv"))

elapsed <- proc.time()[["elapsed"]] - wall_begin
write_metadata(file.path(results, "metadata.csv"), list(
  profile = cfg$profile,
  cells = nrow(grid),
  reps = cfg$reps,
  methods = 38L,
  chunk_size = cfg$chunk_size,
  cores = cfg$cores,
  seed_base = cfg$seed_base,
  shard_index = cfg$shard_index,
  shard_count = cfg$shard_count,
  elapsed_seconds = elapsed,
  new_replications = new_jobs,
  restriction_method = "Satorra 2000",
  A_method = "delta",
  alpha_convention = "p <= 0.05",
  target_skewness_moderate = 2,
  target_excess_kurtosis_moderate = 7,
  target_skewness_severe = 3,
  target_excess_kurtosis_severe = 21,
  paper_doi = "10.3758/s13428-026-02968-4",
  paper_osf = "https://osf.io/h2y3n/"
), packages = c("magmaan", "lavaan", "semTests"))

if (cfg$profile == "smoke") {
  stopifnot(
    nrow(raw) == total_jobs,
    all(raw$fit_ok),
    all(raw$nested_ok),
    nrow(failures) == 0L,
    all(generator_calibration$calibration_ok),
    all(is.finite(raw[[paste0("p_", "pebad_ug_ml")]]))
  )
  if (cfg$semtests_parity) {
    comparable <- is.finite(parity$abs_diff)
    stopifnot(any(comparable), all(parity$pass[comparable]))
  }
  message("SMOKE PASS")
}
cat(sprintf(
  "wrote %s in %.1fs (%d failures, %d warning rows)\n",
  normalizePath(results), elapsed, nrow(failures), nrow(warnings)))
