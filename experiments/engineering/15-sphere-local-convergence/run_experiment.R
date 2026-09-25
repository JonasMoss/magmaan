#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "..", "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaanlab", "install the current R package first (just r-dev)")
suppressPackageStartupMessages(library(magmaanlab))

source(experiment_path("R", "designs.R"))
source(experiment_path("R", "fit.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--full] [options]\n\n",
  "Local convergence of ML and PSD-ML fits, ordinary vs sphere route, under the\n",
  "marker and std.lv identifications. Success is a certified local optimum\n",
  "(or, on the sphere route, a local optimum reported as lying outside the\n",
  "identification). Every failure is refitted with PORT and SLSQP and\n",
  "compared with the PSD-ML minimum to label it an optimizer failure or a\n",
  "missing estimate. Global optimality is not assessed.\n\n",
  "Profiles:\n",
  "  --smoke   10 replications, N = 10 and 50 (default)\n",
  "  --full    200 replications, N = 10, 20, 50, 100\n\n",
  "Options:\n",
  "  --reps 200   --ns 10,20,50,100   --designs ernst,weak_marker,high_r2\n",
  "  --cores 6    --seed-base 88000000   --results-dir PATH\n",
  "Writes results/<profile>/{fits,metadata}.csv.\n",
  sep = "")

parse_args <- function(args) {
  out <- list(profile = "smoke", reps = NULL, ns = NULL, designs = names(designs_all()),
              cores = 6L, seed_base = 88000000L, results_dir = NULL)
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]], call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--smoke") out$profile <- "smoke"
    else if (arg == "--full") out$profile <- "full"
    else if (arg == "--reps") out$reps <- as.integer(take())
    else if (arg == "--ns") out$ns <- as.integer(parse_csv_numeric(take()))
    else if (arg == "--designs") out$designs <- parse_csv_arg(take())
    else if (arg == "--cores") out$cores <- as.integer(take())
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  smoke <- identical(out$profile, "smoke")
  out$reps <- out$reps %||% if (smoke) 10L else 200L
  out$ns <- out$ns %||% if (smoke) c(10L, 50L) else c(10L, 20L, 50L, 100L)
  if (any(!out$designs %in% names(designs_all()))) stop("unknown design", call. = FALSE)
  if (is.null(out$results_dir)) out$results_dir <- file.path(results_dir(), out$profile)
  out
}

`%||%` <- function(x, y) if (is.null(x)) y else x

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
  path <- function(name) file.path(opts$results_dir, paste0(name, ".csv"))
  grid <- expand.grid(rep = seq_len(opts$reps), n = opts$ns, design = opts$designs,
                      stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  grid$seed <- opts$seed_base + 1000000L * match(grid$design, names(designs_all())) +
    1000L * grid$n + grid$rep
  cat(sprintf("magmaan %s | profile %s | %d draws x 8 fits | %d cores\n",
              as.character(utils::packageVersion("magmaanlab")), opts$profile,
              nrow(grid), opts$cores))
  t0 <- proc.time()[["elapsed"]]
  cells <- split(seq_len(nrow(grid)), paste(grid$design, grid$n))
  rows <- list()
  for (k in seq_along(cells)) {
    idx <- cells[[k]]
    chunk <- parallel::mclapply(idx, function(i)
      run_draw(grid$design[i], grid$n[i], grid$rep[i], grid$seed[i]),
      mc.cores = opts$cores, mc.preschedule = TRUE)
    bad <- !vapply(chunk, is.data.frame, logical(1))
    if (any(bad)) stop("worker failure: ", as.character(chunk[[which(bad)[1L]]]), call. = FALSE)
    rows <- c(rows, chunk)
    cat(sprintf("  [%2d/%d] %-18s %6.1fs\n", k, length(cells), names(cells)[k],
                proc.time()[["elapsed"]] - t0))
  }
  write_rows(rows, path("fits"))
  ref <- magmaan_cache_ref()
  write_metadata(path("metadata"), values = list(
    profile = opts$profile, reps = opts$reps, ns = opts$ns, designs = opts$designs,
    seed_base = opts$seed_base, cores = opts$cores,
    magmaan_git_head = ref$git_head, magmaan_git_dirty = ref$git_dirty),
    packages = "magmaanlab")
  cat(sprintf("done in %.1fs. Wrote:\n  %s\n  %s\n", proc.time()[["elapsed"]] - t0,
              path("fits"), path("metadata")))
}

main()
