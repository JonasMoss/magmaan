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
require_pkg("lavaan")
require_pkg("MASS")
suppressPackageStartupMessages({
  library(magmaanlab)
  library(lavaan)
})

source(experiment_path("R", "cases.R"))
source(experiment_path("R", "compare.R"))
source(experiment_path("R", "population.R"))
source(experiment_path("R", "invariance.R"))
source(experiment_path("R", "coverage.R"))

sections_all <- c("recovery", "population", "invariance", "coverage")

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--full] [options]\n\n",
  "Sanity checks for the sphere chart (frontier_fit_sphere). Deterministic:\n",
  "fixed data sets and exact population moments, no replications.\n\n",
  "Sections:\n",
  "  recovery    ~38 models (identification conventions, measurement\n",
  "              invariance, equality and constraint syntax, structure) x\n",
  "              ML/ULS/GLS/WLS/FIML/PSD: sphere vs ordinary magmaan vs lavaan,\n",
  "              plus SEs, robust SEs, standardized, fit measures, MIs, :=.\n",
  "  population  exact population moments, including populations outside the\n",
  "              marker, std.lv or effect-coding chart; recovery of the\n",
  "              population in every chart that contains it.\n",
  "  invariance  one data set under four identifications: same sphere\n",
  "              solution, and translation between identifications.\n",
  "  coverage    inputs the sphere route does not support must error.\n\n",
  "Profiles:\n",
  "  --smoke   7 recovery cases, 3 populations, 1 invariance case (default)\n",
  "  --full    everything\n\n",
  "Options:\n",
  "  --sections recovery,population   --cases hs_marker,pd_labels\n",
  "  --results-dir PATH\n",
  "Writes results/<profile>/{recovery,population,population_reidentify,\n",
  "invariance,invariance_reidentify,coverage,metadata}.csv.\n",
  sep = "")

parse_args <- function(args) {
  out <- list(profile = "smoke", sections = sections_all, cases = NULL,
              results_dir = NULL)
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
    else if (arg == "--sections") out$sections <- parse_csv_arg(take())
    else if (arg == "--cases") out$cases <- parse_csv_arg(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  if (any(!out$sections %in% sections_all)) stop("unknown section", call. = FALSE)
  if (is.null(out$results_dir)) out$results_dir <- file.path(results_dir(), out$profile)
  out
}

progress <- function(t0, i, n, what) {
  cat(sprintf("  [%3d/%d] %-42s %6.1fs\n", i, n, what,
              proc.time()[["elapsed"]] - t0))
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
  path <- function(name) file.path(opts$results_dir, paste0(name, ".csv"))
  data <- exp_datasets()
  smoke <- identical(opts$profile, "smoke")
  written <- character()
  t0 <- proc.time()[["elapsed"]]
  cat(sprintf("magmaan %s, lavaan %s | profile %s\n",
              as.character(utils::packageVersion("magmaanlab")),
              as.character(utils::packageVersion("lavaan")), opts$profile))

  if ("recovery" %in% opts$sections) {
    cases <- all_cases()
    ids <- opts$cases %||% if (smoke) smoke_case_ids else names(cases)
    if (any(!ids %in% names(cases))) stop("unknown case", call. = FALSE)
    jobs <- do.call(rbind, lapply(ids, function(id)
      data.frame(case = id, estimator = cases[[id]]$estimators, stringsAsFactors = FALSE)))
    cat(sprintf("recovery: %d cases, %d fits per route\n", length(ids), nrow(jobs)))
    rows <- vector("list", nrow(jobs))
    for (j in seq_len(nrow(jobs))) {
      rows[[j]] <- run_recovery_case(cases[[jobs$case[j]]], jobs$estimator[j], data)
      progress(t0, j, nrow(jobs), paste(jobs$case[j], jobs$estimator[j]))
    }
    write_rows(rows, path("recovery"))
    written <- c(written, path("recovery"))
  }

  if ("population" %in% opts$sections) {
    pops <- all_populations()
    if (smoke) pops <- pops[c("regular", "marker_pole", "stdlv_pole")]
    cat(sprintf("population: %d populations x %d charts\n", length(pops), length(charts)))
    fits <- list()
    reid <- list()
    for (k in seq_along(pops)) {
      r <- run_population(names(pops)[k], pops[[k]])
      fits[[k]] <- r$fits
      reid[[k]] <- r$reidentify
      progress(t0, k, length(pops), names(pops)[k])
    }
    write_rows(fits, path("population"))
    write_rows(reid, path("population_reidentify"))
    written <- c(written, path("population"), path("population_reidentify"))
  }

  if ("invariance" %in% opts$sections) {
    ics <- invariance_cases()
    if (smoke) ics <- ics["ernst_sem"]
    cat(sprintf("invariance: %d cases x %d charts\n", length(ics), length(charts)))
    fits <- list()
    reid <- list()
    for (k in seq_along(ics)) {
      r <- run_invariance_case(names(ics)[k], ics[[k]], data)
      fits[[k]] <- r$fits
      reid[[k]] <- r$reidentify
      progress(t0, k, length(ics), names(ics)[k])
    }
    write_rows(fits, path("invariance"))
    write_rows(reid, path("invariance_reidentify"))
    written <- c(written, path("invariance"), path("invariance_reidentify"))
  }

  if ("coverage" %in% opts$sections) {
    cat("coverage\n")
    write_csv(run_coverage(data), path("coverage"))
    written <- c(written, path("coverage"))
  }

  ref <- magmaan_cache_ref()
  write_metadata(path("metadata"), values = list(
    profile = opts$profile, sections = opts$sections,
    cases = opts$cases %||% "", magmaan_git_head = ref$git_head,
    magmaan_git_dirty = ref$git_dirty),
    packages = c("magmaanlab", "lavaan"))
  written <- c(written, path("metadata"))
  cat(sprintf("done in %.1fs. Wrote:\n", proc.time()[["elapsed"]] - t0))
  cat(paste0("  ", written, "\n"), sep = "")
}

`%||%` <- function(x, y) if (is.null(x)) y else x

main()
