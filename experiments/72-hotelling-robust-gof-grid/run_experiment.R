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
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "engine.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--probe|--full] [options]\n\n",
  "Does ordinary Hotelling calibrate a centered pivotal SEM GOF score across\n",
  "residual rank and sample-to-rank ratio?\n\n",
  "Profiles:\n",
  "  --smoke       Full 32-cell grid, 2 reps/cell (default).\n",
  "  --probe       Full grid, 500 reps/cell.\n",
  "  --full        Full grid, 2,000 reps/cell.\n\n",
  "Design defaults:\n",
  "  p = 4,5,6,8 (q = 2,5,9,20); n = 30,60,120,240;\n",
  "  normal and severe Vale-Maurelli (skew 3, excess kurtosis 21).\n\n",
  "Options:\n",
  "  --reps N        Override replications per cell.\n",
  "  --p CSV         Override indicator counts.\n",
  "  --n CSV         Override sample sizes.\n",
  "  --max-cells N   Run the first N design cells.\n",
  "  --cores N       Parallel cell workers (default at most 4).\n",
  "  --seed-base N   Deterministic seed base.\n",
  "  --results-dir P Output directory.\n",
  "  --help          Show this help.\n", sep = "")

parse_args <- function(args) {
  out <- list(
    mode = "smoke", reps = NULL, p = c(4L, 5L, 6L, 8L),
    n = c(30L, 60L, 120L, 240L), max_cells = NULL,
    cores = min(4L, max(1L, parallel::detectCores() - 2L)),
    seed_base = 20260720L, results_dir = NULL)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
    args[[i]]
  }
  while (i <= length(args)) {
    argument <- args[[i]]
    if (argument %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (argument == "--smoke") {
      out$mode <- "smoke"
    } else if (argument == "--probe") {
      out$mode <- "probe"
    } else if (argument == "--full") {
      out$mode <- "full"
    } else if (argument == "--reps") {
      out$reps <- as.integer(take())
    } else if (argument == "--p") {
      out$p <- as.integer(parse_csv_numeric(take()))
    } else if (argument == "--n") {
      out$n <- as.integer(parse_csv_numeric(take()))
    } else if (argument == "--max-cells") {
      out$max_cells <- as.integer(take())
    } else if (argument == "--cores") {
      out$cores <- as.integer(take())
    } else if (argument == "--seed-base") {
      out$seed_base <- as.integer(take())
    } else if (argument == "--results-dir") {
      out$results_dir <- take()
    } else {
      stop("unknown argument: ", argument, call. = FALSE)
    }
    i <- i + 1L
  }
  if (is.null(out$reps)) {
    out$reps <- switch(out$mode, probe = 500L, full = 2000L, smoke = 2L)
  }
  if (out$reps < 1L || any(!out$p %in% 4:8) ||
      any(out$n < 10L) || out$cores < 1L ||
      out$seed_base < 0L ||
      (!is.null(out$max_cells) && out$max_cells < 1L)) {
    stop("invalid numeric option", call. = FALSE)
  }
  out
}

opt <- parse_args(commandArgs(trailingOnly = TRUE))
script_dir <- experiment_dir()
results <- opt$results_dir %||% file.path(script_dir, "results", opt$mode)
dir.create(results, recursive = TRUE, showWarnings = FALSE)

populations <- stats::setNames(
  lapply(opt$p, hrg_population), as.character(opt$p))
specs <- stats::setNames(
  lapply(opt$p, hrg_model_spec), as.character(opt$p))
samplers <- list()
for (p in opt$p) {
  for (distribution in c("normal", "vm")) {
    key <- paste(p, distribution, sep = "_")
    cat(sprintf("calibrating p=%d distribution=%s\n", p, distribution))
    samplers[[key]] <- hrg_calibrate_sampler(
      populations[[as.character(p)]], distribution)
  }
}

design <- expand.grid(
  p = opt$p, n = opt$n, distribution = c("normal", "vm"),
  stringsAsFactors = FALSE)
design$q <- vapply(
  design$p,
  function(p) populations[[as.character(p)]]$q,
  integer(1L))
design$n_over_q <- design$n / design$q
design$cell_id <- seq_len(nrow(design))
design <- design[c(
  "cell_id", "p", "q", "n", "n_over_q", "distribution")]
design <- design[design$n > design$q + 1L, , drop = FALSE]
if (!is.null(opt$max_cells)) {
  design <- design[
    seq_len(min(opt$max_cells, nrow(design))), , drop = FALSE]
}

cat(sprintf(
  "mode=%s cells=%d reps=%d fits=%d cores=%d\n",
  opt$mode, nrow(design), opt$reps, nrow(design) * opt$reps, opt$cores))
wall_begin <- proc.time()[["elapsed"]]

run_cell <- function(k) {
  cell <- design[k, , drop = FALSE]
  key <- paste(cell$p, cell$distribution, sep = "_")
  rows <- vector("list", opt$reps)
  progress_every <- max(1L, opt$reps %/% 10L)
  for (rep_id in seq_len(opt$reps)) {
    rows[[rep_id]] <- hrg_one_rep(
      cell, rep_id, specs[[as.character(cell$p)]],
      samplers[[key]], opt$seed_base)
    if (rep_id %% progress_every == 0L || rep_id == opt$reps) {
      cat(sprintf(
        "[%d/%d] p=%d q=%d n=%d %-6s progress=%d/%d\n",
        k, nrow(design), cell$p, cell$q, cell$n,
        cell$distribution, rep_id, opt$reps))
    }
  }
  out <- do.call(rbind, rows)
  cat(sprintf(
    "[%d/%d] p=%d q=%d n=%d %-6s fit=%d/%d score=%d/%d fmg=%d/%d\n",
    k, nrow(design), cell$p, cell$q, cell$n, cell$distribution,
    sum(out$fit_ok), nrow(out), sum(out$score_ok), nrow(out),
    sum(out$fmg_ok), nrow(out)))
  out
}

if (.Platform$OS.type != "windows" &&
    opt$cores > 1L && nrow(design) > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(design)), run_cell,
    mc.cores = min(opt$cores, nrow(design)),
    mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(design)), run_cell)
}
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep), , drop = FALSE]
row.names(raw) <- NULL

write_csv(design, file.path(results, "design.csv"))
write_csv(raw, file.path(results, "replications.csv"))
write_csv(
  hrg_summarize_methods(raw),
  file.path(results, "method_summary.csv"))
write_csv(
  hrg_summarize_diagnostics(raw),
  file.path(results, "diagnostics_summary.csv"))
failures <- raw[
  !raw$fit_ok | !raw$score_ok | !raw$fmg_ok,
  c(
    "cell_id", "p", "q", "n", "distribution", "rep",
    "fit_ok", "score_ok", "fmg_ok",
    "fit_error", "score_error", "fmg_error"),
  drop = FALSE]
write_csv(failures, file.path(results, "failures.csv"))
write_metadata(
  file.path(results, "metadata.csv"),
  list(
    profile = opt$mode,
    cells = nrow(design),
    reps_per_cell = opt$reps,
    indicator_counts = paste(opt$p, collapse = ","),
    sample_sizes = paste(opt$n, collapse = ","),
    distributions = "normal,vale_maurelli",
    vm_skewness = 3,
    vm_excess_kurtosis = 21,
    residual_ranks = paste(sort(unique(design$q)), collapse = ","),
    alpha = 0.05,
    seed_base = opt$seed_base,
    cores = opt$cores,
    elapsed_seconds = round(proc.time()[["elapsed"]] - wall_begin, 3),
    note = paste(
      "correctly specified single-group covariance-only one-factor models;",
      "Hotelling is a finite-sample reference for the centered pivotal score")),
  packages = "magmaan")

cat(sprintf(
  "wrote %s (%.1f seconds, %d failure rows)\n",
  results, proc.time()[["elapsed"]] - wall_begin, nrow(failures)))
