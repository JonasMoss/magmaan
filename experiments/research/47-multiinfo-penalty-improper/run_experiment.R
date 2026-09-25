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

source(experiment_path("R", "design.R"))
source(experiment_path("R", "fitters.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Compare unbounded ML, variance-bounded ML, PSD-constrained ML, the\n",
  "multi-information penalized ML (lambda = eta - 1 = 0.1 ... 2), and the\n",
  "latent-determinacy barrier (same weights) on paired small-sample data\n",
  "from near-improper one-factor, two-factor, and recursive path populations.\n\n",
  "Profiles:\n",
  "  --smoke   N = 50, 200; 20 replications (default)\n",
  "  --pilot   N = 50, 100, 200, 400; 200 replications\n",
  "  --full    N = 50, 100, 200, 400; 1000 replications\n\n",
  "Options:\n",
  "  --reps N --n-values 50,100 --designs f1_p3,f2_r97 --methods ml,pen_l025\n",
  "  --seed-base N --cores N --results-dir PATH\n\n",
  "Designs: f1_p3, f1_p5, f2_r90, f2_r97, path_r2_90\n",
  "Methods: ml, ml_bounded, psd, pen_l010, pen_l025, pen_l050, pen_l100, pen_l200,\n",
  "         det_l010, det_l025, det_l050, det_l100, det_l200\n",
  "Writes results/<profile>/{fits,params,metadata}.csv.\n",
  sep = "")

parse_int_csv <- function(x) {
  out <- as.integer(parse_csv_arg(x))
  if (!length(out) || anyNA(out)) stop("invalid integer list: ", x, call. = FALSE)
  out
}

parse_args <- function(args) {
  out <- list(profile = "smoke", reps = NULL, n_values = NULL,
              designs = NULL, methods = NULL, seed_base = 20260922L,
              cores = max(1L, parallel::detectCores() - 2L),
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
    else if (arg == "--pilot") out$profile <- "pilot"
    else if (arg == "--full") out$profile <- "full"
    else if (arg == "--reps") out$reps <- as.integer(take())
    else if (arg == "--n-values") out$n_values <- parse_int_csv(take())
    else if (arg == "--designs") out$designs <- parse_csv_arg(take())
    else if (arg == "--methods") out$methods <- parse_csv_arg(take())
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--cores") out$cores <- as.integer(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$profile,
    smoke = list(reps = 20L, n_values = c(50L, 200L)),
    pilot = list(reps = 200L, n_values = c(50L, 100L, 200L, 400L)),
    full = list(reps = 1000L, n_values = c(50L, 100L, 200L, 400L)))
  if (is.null(out$reps)) out$reps <- defaults$reps
  if (is.null(out$n_values)) out$n_values <- defaults$n_values
  all_d <- names(all_designs())
  all_m <- names(method_table())
  if (is.null(out$designs)) out$designs <- all_d
  if (is.null(out$methods)) out$methods <- all_m
  if (any(!out$designs %in% all_d)) stop("unknown design", call. = FALSE)
  if (any(!out$methods %in% all_m)) stop("unknown method", call. = FALSE)
  if (anyNA(c(out$reps, out$seed_base, out$cores)) || out$reps < 1L ||
      out$cores < 1L || any(out$n_values < 10L)) {
    stop("reps and cores must be positive and N at least 10", call. = FALSE)
  }
  if (is.null(out$results_dir)) {
    out$results_dir <- file.path(results_dir(), out$profile)
  }
  out
}

run_rep <- function(design, spec, n, rep, opts, methods, truth) {
  seed <- simulation_seed(opts$seed_base, design$id, n, rep)
  df <- simulate_data(design, n, seed)
  dat <- df_to_data(df, spec)
  fits <- vector("list", length(opts$methods))
  params <- vector("list", length(opts$methods))
  for (k in seq_along(opts$methods)) {
    m <- opts$methods[[k]]
    rec <- fit_record(design, m, methods[[m]]$fit, spec, dat, truth)
    fits[[k]] <- rec$fit
    params[[k]] <- rec$params
  }
  fits <- do.call(rbind, fits)
  params <- do.call(rbind, params)
  tag <- data.frame(design = design$key, n = n, rep = rep, seed = seed,
                    stringsAsFactors = FALSE)
  list(fits = cbind(tag[rep(1L, nrow(fits)), ], fits),
       params = if (is.null(params)) NULL else cbind(tag[rep(1L, nrow(params)), ], params))
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  designs <- all_designs()[opts$designs]
  methods <- method_table()
  dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    fits = file.path(opts$results_dir, "fits.csv"),
    params = file.path(opts$results_dir, "params.csv"),
    design = file.path(opts$results_dir, "design.csv"),
    metadata = file.path(opts$results_dir, "metadata.csv"))
  unlink(c(paths$fits, paths$params))

  design_rows <- do.call(rbind, lapply(designs, function(d) {
    data.frame(design = d$key, id = d$id, label = d$label,
               p = ncol(d$Sigma), key_quantity = d$key_quantity,
               key_truth = d$key_truth, stringsAsFactors = FALSE)
  }))
  write_csv(design_rows, paths$design)
  truths <- lapply(designs, population_truth)

  cells <- expand.grid(n = opts$n_values, design = names(designs),
                       stringsAsFactors = FALSE)
  total <- nrow(cells) * opts$reps
  cat(sprintf("magmaan %s | %d cells x %d reps x %d methods | %d cores\n",
              as.character(utils::packageVersion("magmaanlab")), nrow(cells),
              opts$reps, length(opts$methods), opts$cores))
  t_start <- proc.time()[["elapsed"]]
  done <- 0L
  for (ci in seq_len(nrow(cells))) {
    d <- designs[[cells$design[[ci]]]]
    n <- cells$n[[ci]]
    spec <- model_spec(d$syntax)
    chunks <- split(seq_len(opts$reps),
                    cut(seq_len(opts$reps), min(opts$cores, opts$reps), labels = FALSE))
    out <- parallel::mclapply(chunks, function(reps) {
      lapply(reps, function(r) run_rep(d, spec, n, r, opts, methods,
                                       truths[[d$key]]))
    }, mc.cores = opts$cores, mc.preschedule = TRUE)
    bad <- vapply(out, inherits, logical(1), "try-error")
    if (any(bad)) stop("worker failed: ", as.character(out[bad][[1L]]), call. = FALSE)
    reps_out <- unlist(out, recursive = FALSE)
    append_csv(do.call(rbind, lapply(reps_out, `[[`, "fits")), paths$fits)
    append_csv(do.call(rbind, lapply(reps_out, `[[`, "params")), paths$params)
    done <- done + opts$reps
    elapsed <- proc.time()[["elapsed"]] - t_start
    eta <- elapsed / done * (total - done)
    cat(sprintf("[%2d/%d] %-11s N=%-4d  %6.1fs elapsed, ETA %6.1fs\n",
                ci, nrow(cells), d$key, n, elapsed, eta))
  }

  write_metadata(paths$metadata, values = list(
    profile = opts$profile, reps = opts$reps,
    n_values = paste(opts$n_values, collapse = ","),
    designs = paste(opts$designs, collapse = ","),
    methods = paste(opts$methods, collapse = ","),
    seed_base = opts$seed_base, cores = opts$cores,
    boundary_tol = boundary_tol,
    magmaan_git_head = magmaan_cache_ref()$git_head,
    magmaan_git_dirty = magmaan_cache_ref()$git_dirty),
    packages = "magmaanlab")
  cat("Wrote:\n", paste0("  ", unlist(paths), "\n"), sep = "")
}

main()
