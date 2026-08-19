#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaan", "install the current R package first")
suppressPackageStartupMessages(library(magmaan))

source(experiment_path("R", "design.R"))
source(experiment_path("R", "objectives.R"))
source(experiment_path("R", "fitters.R"))
source(experiment_path("R", "summaries.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot] [options]\n\n",
  "Exercise ordinary and covariance-honest point estimators across the\n",
  "continuous pilot geometries or all-family smoke anchors.\n\n",
  "Profile:\n",
  "  --smoke             two deterministic replications per planned smoke cell\n\n",
  "  --pilot             100 replications per continuous one-axis pilot cell\n\n",
  "Options:\n",
  "  --reps N            override profile replications\n",
  "  --families CSV      subset the registered estimator families\n",
  "  --geometries CSV    subset geometry IDs listed by --dry-run-plan\n",
  "  --seed-base N       deterministic seed base (default 20260819)\n",
  "  --max-iter N        optimizer iteration cap (default 5000)\n",
  "  --results-dir PATH  output directory (default results/<profile>)\n",
  "  --resume            retain completed task checkpoints\n",
  "  --progress-every N  progress interval in tasks (default 1)\n",
  "  --dry-run-plan      print the task grid without fitting\n",
  sep = ""
)

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    reps = NULL,
    families = NULL,
    geometries = NULL,
    seed_base = 20260819L,
    max_iter = 5000L,
    results_dir = NULL,
    resume = FALSE,
    progress_every = 1L,
    dry_run_plan = FALSE
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
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--smoke") {
      out$profile <- "smoke"
    } else if (arg == "--pilot") {
      out$profile <- "pilot"
    } else if (arg == "--reps") out$reps <- as.integer(take())
    else if (arg == "--families") out$families <- parse_csv_arg(take())
    else if (arg == "--geometries") out$geometries <- parse_csv_arg(take())
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--max-iter") out$max_iter <- as.integer(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else if (arg == "--resume") out$resume <- TRUE
    else if (arg == "--progress-every") out$progress_every <- as.integer(take())
    else if (arg == "--dry-run-plan") out$dry_run_plan <- TRUE
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  if (is.null(out$reps)) {
    out$reps <- if (identical(out$profile, "pilot")) 100L else 2L
  }
  ints <- c(out$reps, out$seed_base, out$max_iter, out$progress_every)
  if (anyNA(ints) || any(ints < 1L)) {
    stop("reps, seed-base, max-iter, and progress-every must be positive",
         call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
design <- task_grid(args$profile, args$reps, args$families, args$geometries)
design$seed <- mapply(
  function(geometry, rep) simulation_seed(args$seed_base, geometry, rep),
  design$geometry, design$rep
)
if (args$dry_run_plan) {
  print(design, row.names = FALSE)
  quit(save = "no", status = 0L)
}

out_dir <- args$results_dir %||%
  experiment_path("results", args$profile)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
paths <- list(
  design = file.path(out_dir, "design.csv"),
  raw = file.path(out_dir, "raw.csv"),
  checkpoint = file.path(out_dir, "checkpoint.csv"),
  pairs = file.path(out_dir, "pairs.csv"),
  summary = file.path(out_dir, "summary.csv"),
  cells = file.path(out_dir, "cells.csv"),
  pair_summary = file.path(out_dir, "pair_summary.csv"),
  invariants = file.path(out_dir, "invariants.csv"),
  metadata = file.path(out_dir, "metadata.csv")
)

if (!args$resume) {
  unlink(unlist(paths), force = TRUE)
}
write_csv(design, paths$design)
completed <- if (args$resume && file.exists(paths$checkpoint)) {
  unique(utils::read.csv(paths$checkpoint, stringsAsFactors = FALSE)$task_id)
} else character()
if (args$resume && file.exists(paths$raw)) {
  existing <- utils::read.csv(
    paths$raw, stringsAsFactors = FALSE, check.names = FALSE
  )
  # A task is committed only after all of its fit rows have been appended and
  # its checkpoint row has been written. Drop any interrupted task fragment so
  # resume can rerun it without duplicating rows.
  existing <- existing[existing$task_id %in% completed, , drop = FALSE]
  if (nrow(existing)) write_csv(existing, paths$raw) else unlink(paths$raw)
}

control <- list(max_iter = args$max_iter, gtol = 1e-8)
pending <- design[!design$task_id %in% completed, , drop = FALSE]
message(
  "PSD estimator ", args$profile, ": ", nrow(design), " task(s), ", nrow(pending),
  " pending, ", args$reps, " replication(s)"
)
started <- proc.time()[["elapsed"]]
for (i in seq_len(nrow(pending))) {
  task <- pending[i, , drop = FALSE]
  if (i == 1L || i %% args$progress_every == 0L || i == nrow(pending)) {
    elapsed <- proc.time()[["elapsed"]] - started
    rate <- if (i > 1L && elapsed > 0) (i - 1L) / elapsed else NA_real_
    eta <- if (is.finite(rate) && rate > 0) (nrow(pending) - i + 1L) / rate else NA_real_
    message(
      sprintf(
        "[%d/%d] %s | elapsed %.1fs%s",
        i, nrow(pending), task$task_id, elapsed,
        if (is.finite(eta)) sprintf(" | ETA %.1fs", eta) else ""
      )
    )
  }
  rows <- run_task(task, args$seed_base, control)
  append_csv(rows, paths$raw)
  append_csv(
    data.frame(
      task_id = task$task_id,
      completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      rows = nrow(rows),
      stringsAsFactors = FALSE
    ),
    paths$checkpoint
  )
}

if (!file.exists(paths$raw)) {
  stop("no fit rows were produced", call. = FALSE)
}
raw <- utils::read.csv(paths$raw, stringsAsFactors = FALSE, check.names = FALSE)
raw <- raw[raw$task_id %in% design$task_id, , drop = FALSE]
pairs <- pair_results(raw)
summary <- family_summary(raw)
cells <- pilot_cell_summary(raw)
pair_summary <- pilot_pair_summary(pairs)
invariants <- invariant_summary(raw, pairs)
write_csv(pairs, paths$pairs)
write_csv(summary, paths$summary)
write_csv(cells, paths$cells)
write_csv(pair_summary, paths$pair_summary)
write_csv(invariants, paths$invariants)
elapsed_seconds <- proc.time()[["elapsed"]] - started
if (args$resume && !nrow(pending) && file.exists(paths$metadata)) {
  previous_metadata <- utils::read.csv(
    paths$metadata, stringsAsFactors = FALSE
  )
  previous_elapsed <- suppressWarnings(as.numeric(
    previous_metadata$value[previous_metadata$key == "elapsed_seconds"]
  ))
  if (length(previous_elapsed) == 1L && is.finite(previous_elapsed)) {
    elapsed_seconds <- previous_elapsed
  }
}
write_metadata(
  paths$metadata,
  values = list(
    experiment = "78-psd-estimator-stress",
    profile = args$profile,
    evidence = identical(args$profile, "pilot") && args$reps >= 100L,
    reps = args$reps,
    seed_base = args$seed_base,
    max_iter = args$max_iter,
    families = unique(design$family),
    geometries = unique(design$geometry),
    tasks = nrow(design),
    fit_rows = nrow(raw),
    elapsed_seconds = elapsed_seconds,
    git_head = git_scalar(c("rev-parse", "HEAD")),
    git_dirty = git_dirty()
  ),
  packages = c("magmaan")
)

message("Wrote:")
for (path in unlist(paths)) message("  ", path)
violations <- sum(invariants$violations)
if (violations > 0L) {
  print(invariants, row.names = FALSE)
  stop(args$profile, " validation found ", violations,
       " hard invariant violation(s)",
       call. = FALSE)
}
message("All hard ", args$profile, " invariants passed.")
