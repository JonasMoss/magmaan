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
  "Usage: Rscript run_experiment.R [--smoke|--pilot] [options]\n\n",
  "Projected Satterthwaite-Hotelling calibration for pivotal SEM GOF scores.\n\n",
  "Profiles:\n",
  "  --smoke          5 reps/cell, 1 x reference n=5,000 (default).\n",
  "  --pilot          500 reps/cell, 5 x reference n=100,000.\n\n",
  "Options:\n",
  "  --reps N         Override replications per cell.\n",
  "  --reference-n N  Override score-shape reference size.\n",
  "  --reference-reps N  Override independent reference repetitions.\n",
  "  --n CSV          Sample sizes (default 30,50,100,200).\n",
  "  --cores N        Parallel workers (default at most 4).\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

parse_args <- function(args) {
  out <- list(
    mode = "smoke", reps = NULL, reference_n = NULL, reference_reps = NULL,
    n = c(30L, 50L, 100L, 200L),
    cores = min(4L, max(1L, parallel::detectCores() - 2L)),
    seed_base = 20260719L, results_dir = NULL)
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
    } else if (a == "--smoke") {
      out$mode <- "smoke"
    } else if (a == "--pilot") {
      out$mode <- "pilot"
    } else if (a == "--reps") {
      out$reps <- as.integer(take())
    } else if (a == "--reference-n") {
      out$reference_n <- as.integer(take())
    } else if (a == "--reference-reps") {
      out$reference_reps <- as.integer(take())
    } else if (a == "--n") {
      out$n <- as.integer(parse_csv_numeric(take()))
    } else if (a == "--cores") {
      out$cores <- as.integer(take())
    } else if (a == "--seed-base") {
      out$seed_base <- as.integer(take())
    } else if (a == "--results-dir") {
      out$results_dir <- take()
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (is.null(out$reps)) out$reps <- if (out$mode == "pilot") 500L else 5L
  if (is.null(out$reference_n)) {
    out$reference_n <- if (out$mode == "pilot") 100000L else 5000L
  }
  if (is.null(out$reference_reps)) {
    out$reference_reps <- if (out$mode == "pilot") 5L else 1L
  }
  if (out$reps < 1L || out$reference_n < 1000L ||
      out$reference_reps < 1L ||
      any(!is.finite(out$n)) || any(out$n < 10L) ||
      out$cores < 1L || out$seed_base < 0L) {
    stop("invalid numeric option", call. = FALSE)
  }
  out
}

opt <- parse_args(commandArgs(trailingOnly = TRUE))
script_dir <- experiment_dir()
results <- opt$results_dir %||% file.path(script_dir, "results", opt$mode)
dir.create(results, recursive = TRUE, showWarnings = FALSE)

pop <- pss_population()
spec <- pss_model_spec(pop$p)
samplers <- pss_calibrate_samplers(pop, skew = 2, exkurt = 7)
design <- expand.grid(
  distribution = c("normal", "vm"), n = opt$n,
  stringsAsFactors = FALSE)
design$q <- pop$df
design$cell_id <- seq_len(nrow(design))
design <- design[c("cell_id", "distribution", "n", "q")]

cat(sprintf(
  "mode=%s cells=%d reps=%d reference=%d x %d cores=%d\n",
  opt$mode, nrow(design), opt$reps, opt$reference_reps,
  opt$reference_n, opt$cores))
wall_begin <- proc.time()[["elapsed"]]
reference_shapes <- do.call(rbind, lapply(seq_along(samplers), function(k) {
  distribution <- names(samplers)[[k]]
  do.call(rbind, lapply(seq_len(opt$reference_reps), function(reference_rep) {
    cat(sprintf("reference shape: %s %d/%d (%d cases)\n",
                distribution, reference_rep, opt$reference_reps,
                opt$reference_n))
    pss_reference_shape(
      spec, samplers[[k]], distribution, opt$reference_n, reference_rep,
      opt$seed_base + 9000000L + k * 10000L + reference_rep)
  }))
}))
write_csv(reference_shapes, file.path(results, "reference_shapes.csv"))
reference_summary <- pss_summarize_reference(reference_shapes)
write_csv(reference_summary, file.path(results, "reference_summary.csv"))

run_cell <- function(k) {
  cell <- design[k, , drop = FALSE]
  rows <- lapply(seq_len(opt$reps), function(rep_id) {
    pss_one_rep(
      cell, rep_id, spec, samplers[[cell$distribution]],
      reference_summary, opt$seed_base)
  })
  out <- do.call(rbind, rows)
  cat(sprintf(
    "[%d/%d] %-6s n=%3d fit=%d/%d score=%d/%d\n",
    k, nrow(design), cell$distribution, cell$n,
    sum(out$fit_ok), nrow(out), sum(out$score_ok), nrow(out)))
  out
}

if (.Platform$OS.type != "windows" && opt$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(design)), run_cell,
    mc.cores = min(opt$cores, nrow(design)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(design)), run_cell)
}
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep), , drop = FALSE]
row.names(raw) <- NULL

write_csv(design, file.path(results, "design.csv"))
write_csv(raw, file.path(results, "replications.csv"))
write_csv(pss_summarize_methods(raw), file.path(results, "method_summary.csv"))
write_csv(
  pss_summarize_diagnostics(raw),
  file.path(results, "diagnostics_summary.csv"))
failures <- raw[
  !raw$fit_ok | !raw$score_ok | !raw$fmg_ok,
  intersect(
    c("cell_id", "distribution", "n", "rep", "fit_ok", "score_ok", "fmg_ok",
      "fit_error", "score_error", "fmg_error"),
    names(raw)),
  drop = FALSE]
write_csv(failures, file.path(results, "failures.csv"))
write_metadata(
  file.path(results, "metadata.csv"),
  list(
    profile = opt$mode,
    cells = nrow(design),
    reps_per_cell = opt$reps,
    reference_n = opt$reference_n,
    reference_reps = opt$reference_reps,
    sample_sizes = paste(opt$n, collapse = ","),
    distributions = "normal,vale_maurelli",
    vm_skewness = 2,
    vm_excess_kurtosis = 7,
    q = pop$df,
    alpha = 0.05,
    seed_base = opt$seed_base,
    cores = opt$cores,
    elapsed_seconds = round(proc.time()[["elapsed"]] - wall_begin, 3),
    note = paste(
      "true five-indicator one-factor model;",
      "oracle means DGP-specific fixed score shape, not a feasible method")),
  packages = "magmaan")

cat(sprintf(
  "wrote %s (%.1f seconds, %d failure rows)\n",
  results, proc.time()[["elapsed"]] - wall_begin, nrow(failures)))
