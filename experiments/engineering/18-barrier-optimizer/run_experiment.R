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
require_pkg("jsonlite")
suppressPackageStartupMessages(library(magmaanlab))

source(experiment_path("R", "sim_designs.R"))
source(experiment_path("R", "other_designs.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--full] [options]\n\n",
  "Fit the latent-determinacy barrier (lambda log det Q) with several\n",
  "optimizers on the same data and record every return: the research/47\n",
  "populations with their seeds (marker identification), the Ernst et al.\n",
  "design and a collinear latent-predictor design in marker and std.lv\n",
  "identification, and four textbook-corpus cases (when the corpus is mounted).\n\n",
  "Profiles:\n",
  "  --smoke   20 replications, N = 50 and 400 (default)\n",
  "  --full    research/47: 500 replications, N = 50, 100, 200, 400;\n",
  "            Ernst: 150 replications, N = 10 and 30; fold: 300, N = 100\n\n",
  "Options:\n",
  "  --optimizers a,b   default nlopt-lbfgs,port,nlopt-slsqp,nlopt-lbfgs-slsqp-fallback\n",
  "  --lambdas 0.1,0.25 --seed-base N --cores N --results-dir PATH\n",
  "Writes results/<profile>/{fits,metadata}.csv.\n",
  sep = "")

parse_args <- function(args) {
  out <- list(profile = "smoke", seed_base = 20260922L,
              cores = max(1L, parallel::detectCores() - 2L), results_dir = NULL,
              optimizers = c("nlopt-lbfgs", "port", "nlopt-slsqp",
                             "nlopt-lbfgs-slsqp-fallback"),
              lambdas = c(0.1, 0.25))
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) stop("missing value after ", args[[i - 1L]], call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--smoke") out$profile <- "smoke"
    else if (a == "--full") out$profile <- "full"
    else if (a == "--optimizers") out$optimizers <- parse_csv_arg(take())
    else if (a == "--lambdas") out$lambdas <- as.numeric(parse_csv_arg(take()))
    else if (a == "--seed-base") out$seed_base <- as.integer(take())
    else if (a == "--cores") out$cores <- as.integer(take())
    else if (a == "--results-dir") out$results_dir <- take()
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out$grid <- switch(
    out$profile,
    smoke = list(sim_reps = 20L, sim_n = c(50L, 400L), ernst_reps = 20L,
                 ernst_n = c(10L, 30L), fold_reps = 20L, fold_n = 100L),
    full = list(sim_reps = 500L, sim_n = c(50L, 100L, 200L, 400L), ernst_reps = 150L,
                ernst_n = c(10L, 30L), fold_reps = 300L, fold_n = 100L))
  if (is.null(out$results_dir)) out$results_dir <- file.path(results_dir(), out$profile)
  out
}

error_kind <- function(msg) {
  k <- regmatches(msg, regexpr("\\[[A-Za-z]+\\]", msg))
  if (length(k)) gsub("[][]", "", k) else "Error"
}
error_f <- function(msg) {
  v <- regmatches(msg, regexpr("f=[-0-9.eE+]+", msg))
  if (length(v)) as.numeric(sub("f=", "", v)) else NA_real_
}

fit_one <- function(spec, dat, lambda, optimizer) {
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    suppressWarnings(frontier_fit_ml_multiinfo(spec, dat, weight = lambda,
                                               target = "determinacy",
                                               optimizer = optimizer)),
    error = function(e) e)
  sec <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    msg <- conditionMessage(fit)
    return(data.frame(lambda = lambda, optimizer = optimizer, status = error_kind(msg),
                      converged = FALSE, verdict = NA_character_, pen_fmin = NA_real_,
                      fail_f = error_f(msg), fmin = NA_real_, grad_norm = NA_real_,
                      f_evals = NA_real_, admissible = NA, seconds = sec,
                      stringsAsFactors = FALSE))
  }
  data.frame(lambda = lambda, optimizer = optimizer, status = "ok",
             converged = isTRUE(fit$converged),
             verdict = fit$verdict$status %||% NA_character_,
             pen_fmin = fit$penalty$penalized_fmin, fail_f = NA_real_,
             fmin = fit$fmin, grad_norm = fit$grad_norm %||% NA_real_,
             f_evals = fit$f_evals %||% NA_real_,
             admissible = isTRUE(fit$diagnostics$admissibility$admissible),
             seconds = sec, stringsAsFactors = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

fit_all <- function(spec, dat, opts) {
  rows <- list()
  for (l in opts$lambdas) for (o in opts$optimizers) {
    rows[[length(rows) + 1L]] <- fit_one(spec, dat, l, o)
  }
  do.call(rbind, rows)
}

gaussian_sample <- function(Sigma, n, seed) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * ncol(Sigma)), n) %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  as.data.frame(X)
}

# Cells: one population, identification, and N; replications run in parallel.
build_cells <- function(opts) {
  g <- opts$grid
  sims <- all_designs()
  cells <- list()
  for (d in sims) for (n in g$sim_n) {
    cells[[length(cells) + 1L]] <- list(study = "research47", key = d$key, chart = "marker",
                                        n = n, reps = g$sim_reps, design = d)
  }
  for (n in g$ernst_n) for (ch in c("marker", "stdlv")) {
    cells[[length(cells) + 1L]] <- list(study = "ernst", key = "ernst", chart = ch, n = n,
                                        reps = g$ernst_reps, design = ernst_design())
  }
  for (n in g$fold_n) for (ch in c("marker", "stdlv")) {
    cells[[length(cells) + 1L]] <- list(study = "fold", key = "fold", chart = ch, n = n,
                                        reps = g$fold_reps, design = fold_design())
  }
  cells
}

run_cell <- function(cell, opts) {
  spec <- model_spec(cell$design$syntax, std_lv = identical(cell$chart, "stdlv"))
  chunks <- split(seq_len(cell$reps),
                  cut(seq_len(cell$reps), min(opts$cores, cell$reps), labels = FALSE))
  out <- parallel::mclapply(chunks, function(reps) {
    do.call(rbind, lapply(reps, function(r) {
      if (cell$study == "research47") {
        seed <- simulation_seed(opts$seed_base, cell$design$id, cell$n, r)
        df <- simulate_data(cell$design, cell$n, seed)
      } else {
        seed <- opts$seed_base + (if (cell$study == "ernst") 1e8 else 2e8) + cell$n * 1e4 + r
        df <- gaussian_sample(cell$design$Sigma, cell$n, seed)
      }
      rec <- fit_all(spec, df_to_data(df, spec), opts)
      cbind(data.frame(study = cell$study, design = cell$key, chart = cell$chart,
                       n = cell$n, rep = r, stringsAsFactors = FALSE), rec)
    }))
  }, mc.cores = opts$cores, mc.preschedule = TRUE)
  bad <- vapply(out, inherits, logical(1), "try-error")
  if (any(bad)) stop("worker failed: ", as.character(out[bad][[1L]]), call. = FALSE)
  do.call(rbind, out)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
  path_fits <- file.path(opts$results_dir, "fits.csv")
  path_meta <- file.path(opts$results_dir, "metadata.csv")
  unlink(path_fits)
  cells <- build_cells(opts)
  cat(sprintf("magmaanlab %s | %d cells | optimizers %s | lambda %s | %d cores\n",
              as.character(utils::packageVersion("magmaanlab")), length(cells),
              paste(opts$optimizers, collapse = ","), paste(opts$lambdas, collapse = ","),
              opts$cores))
  t0 <- proc.time()[["elapsed"]]
  for (ci in seq_along(cells)) {
    append_csv(run_cell(cells[[ci]], opts), path_fits)
    c <- cells[[ci]]
    el <- proc.time()[["elapsed"]] - t0
    cat(sprintf("[%2d/%d] %-10s %-13s %-6s N=%-4d %7.1fs elapsed, ETA %7.1fs\n", ci,
                length(cells), c$study, c$key, c$chart, c$n, el,
                el / ci * (length(cells) - ci)))
  }
  if (corpus_available()) {
    for (id in corpus_case_ids) {
      cs <- load_corpus_case(id)
      rec <- fit_all(cs$spec, cs$data, opts)
      append_csv(cbind(data.frame(study = "corpus", design = id, chart = "model",
                                  n = sum(unlist(cs$data$nobs)), rep = 1L,
                                  stringsAsFactors = FALSE), rec), path_fits)
    }
    cat("corpus cases done\n")
  } else {
    cat("textbook corpus not mounted; corpus cases skipped\n")
  }
  write_metadata(path_meta, values = list(
    profile = opts$profile, optimizers = paste(opts$optimizers, collapse = ","),
    lambdas = paste(opts$lambdas, collapse = ","), seed_base = opts$seed_base,
    cores = opts$cores, sim_reps = opts$grid$sim_reps,
    sim_n = paste(opts$grid$sim_n, collapse = ","), ernst_reps = opts$grid$ernst_reps,
    ernst_n = paste(opts$grid$ernst_n, collapse = ","), fold_reps = opts$grid$fold_reps,
    corpus = corpus_available(),
    magmaan_git_head = magmaan_cache_ref()$git_head,
    magmaan_git_dirty = magmaan_cache_ref()$git_dirty),
    packages = c("magmaanlab", "jsonlite"))
  cat("Wrote:\n  ", path_fits, "\n  ", path_meta, "\n", sep = "")
}

main()
