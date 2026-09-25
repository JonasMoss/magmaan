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
source(experiment_path("R", "structure.R"))
source(experiment_path("R", "posterior.R"))
source(experiment_path("R", "estimators.R"))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Per simulated dataset: ML, PSD-ML, the multi-information barrier at\n",
  "lambda = 0.1, 0.25, 0.5, 1, a truncated-normal approximation around ML, and\n",
  "the posterior mean and median under a flat prior (standardized-latent\n",
  "coordinates) and under the Jeffreys prior, both restricted to admissible\n",
  "(positive semidefinite) solutions. Posteriors use 4 random-walk Metropolis\n",
  "chains; Jeffreys is applied by reweighting thinned draws.\n\n",
  "Profiles:\n",
  "  --smoke   N = 50; 3 replications (default; about a minute)\n",
  "  --pilot   N = 50, 100; 40 replications\n",
  "  --full    N = 50, 100, 200; 300 replications (3-4 hours on 10 cores; use --resume to continue)\n\n",
  "Options:\n",
  "  --reps N --n-values 50,100 --designs f1_p3,f2_r97\n",
  "  --seed-base N --cores N --results-dir PATH\n",
  "  --resume  keep existing results and skip (design, N) cells already written\n\n",
  "Designs: f1_p3, f1_p5, f2_r90, f2_r97, path_r2_90\n",
  "Writes results/<profile>/{fits,params,posterior,design,metadata}.csv.\n",
  sep = "")

parse_int_csv <- function(x) {
  out <- as.integer(parse_csv_arg(x))
  if (!length(out) || anyNA(out)) stop("invalid integer list: ", x, call. = FALSE)
  out
}

parse_args <- function(args) {
  out <- list(profile = "smoke", reps = NULL, n_values = NULL, designs = NULL,
              seed_base = 20260922L, cores = max(1L, parallel::detectCores() - 2L),
              results_dir = NULL, resume = FALSE)
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
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--cores") out$cores <- as.integer(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else if (arg == "--resume") out$resume <- TRUE
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$profile,
    smoke = list(reps = 3L, n_values = 50L),
    pilot = list(reps = 40L, n_values = c(50L, 100L)),
    full = list(reps = 300L, n_values = c(50L, 100L, 200L)))
  if (is.null(out$reps)) out$reps <- defaults$reps
  if (is.null(out$n_values)) out$n_values <- defaults$n_values
  all_d <- names(all_designs())
  if (is.null(out$designs)) out$designs <- all_d
  if (any(!out$designs %in% all_d)) stop("unknown design", call. = FALSE)
  if (anyNA(c(out$reps, out$seed_base, out$cores)) || out$reps < 1L ||
      out$cores < 1L || any(out$n_values < 20L)) {
    stop("reps and cores must be positive and N at least 20", call. = FALSE)
  }
  if (is.null(out$results_dir)) out$results_dir <- file.path(results_dir(), out$profile)
  out
}

inv_rows <- function(method, inv, truth, key, lower = NULL, upper = NULL) {
  if (is.null(inv)) return(NULL)
  q <- names(truth)
  data.frame(method = method, quantity = q, key = q %in% key,
             est = unname(inv[q]),
             lower = if (is.null(lower)) NA_real_ else unname(lower[q]),
             upper = if (is.null(upper)) NA_real_ else unname(upper[q]),
             truth = unname(truth), stringsAsFactors = FALSE)
}

sampler_start <- function(st, point) {
  for (m in c("pen_l025", "pen_l050", "psd")) {
    f <- point[[m]]$fit
    if (is.null(f)) next
    pt <- f$partable
    phi <- tryCatch(marker_to_phi(st, marker_builder(pt, st)(pt$est[free_order(pt)])),
                    error = function(e) NULL)
    if (!is.null(phi) && all(is.finite(phi)) && build_std(st, phi)$ok) {
      return(list(phi = phi, source = m))
    }
  }
  NULL
}

run_rep <- function(design, spec, n, rep, opts, truth) {
  st <- design$structure
  seed <- simulation_seed(opts$seed_base, design$id, n, rep)
  df <- simulate_data(design, n, seed)
  dat <- df_to_data(df, spec)
  S <- stats::cov(as.matrix(df)) * (n - 1) / n

  point <- lapply(point_methods(), run_point, spec = spec, dat = dat, st = st)
  fit_rows <- do.call(rbind, lapply(names(point), function(m) {
    r <- point[[m]]
    data.frame(method = m, status = r$status, converged = r$converged,
               classification = r$class, seconds = r$seconds, stringsAsFactors = FALSE)
  }))
  params <- lapply(names(point), function(m) inv_rows(m, point[[m]]$inv, truth, design$key_params))

  set.seed(seed + 400000000L)
  t0 <- proc.time()[["elapsed"]]
  tn <- tryCatch(trunc_normal(point$ml$fit, st), error = function(e) NULL)
  fit_rows <- rbind(fit_rows, data.frame(
    method = "trunc_normal", status = if (is.null(tn)) "unavailable" else "ok",
    converged = NA, classification = if (is.null(tn)) NA_character_ else "interior",
    seconds = proc.time()[["elapsed"]] - t0, stringsAsFactors = FALSE))
  params[[length(params) + 1L]] <- inv_rows("trunc_normal", tn$inv, truth, design$key_params)

  start <- sampler_start(st, point)
  post <- if (is.null(start)) NULL else tryCatch(
    posterior_fit(st, S, n, start$phi, design$key_params, seed + 500000000L),
    error = function(e) e)
  post_ok <- !is.null(post) && !inherits(post, "error")
  diag_row <- if (post_ok) post$diagnostics else data.frame(
    accept = NA_real_, rhat_key = NA_real_, rhat_phi = NA_real_, ess_key = NA_real_,
    draws = NA_integer_, jeffreys_ess = NA_real_, seconds_mcmc = NA_real_,
    seconds_jeffreys = NA_real_)
  diag_row$start <- if (is.null(start)) NA_character_ else start$source
  diag_row$error <- if (inherits(post, "error")) conditionMessage(post) else ""
  diag_row$trunc_admissible_share <- if (is.null(tn)) NA_real_ else tn$admissible_share
  for (m in posterior_methods) {
    fit_rows <- rbind(fit_rows, data.frame(
      method = m, status = if (post_ok) "ok" else "error", converged = NA,
      classification = if (post_ok) "interior" else NA_character_,
      seconds = if (post_ok) post$diagnostics$seconds_mcmc + post$diagnostics$seconds_jeffreys else NA_real_,
      stringsAsFactors = FALSE))
  }
  if (post_ok) {
    for (prior in c("flat", "jeffreys")) {
      s <- post[[prior]]
      tag <- if (prior == "flat") "flat" else "jeff"
      params[[length(params) + 1L]] <- inv_rows(paste0(tag, "_mean"), s["mean", ], truth,
                                                design$key_params, s["lower", ], s["upper", ])
      params[[length(params) + 1L]] <- inv_rows(paste0(tag, "_median"), s["median", ], truth,
                                                design$key_params, s["lower", ], s["upper", ])
    }
  }
  params <- do.call(rbind, params)
  tag <- data.frame(design = design$key, n = n, rep = rep, seed = seed,
                    stringsAsFactors = FALSE)
  list(fits = cbind(tag[rep(1L, nrow(fit_rows)), ], fit_rows),
       params = cbind(tag[rep(1L, nrow(params)), ], params),
       posterior = cbind(tag, diag_row))
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  designs <- all_designs()[opts$designs]
  dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    fits = file.path(opts$results_dir, "fits.csv"),
    params = file.path(opts$results_dir, "params.csv"),
    posterior = file.path(opts$results_dir, "posterior.csv"),
    design = file.path(opts$results_dir, "design.csv"),
    metadata = file.path(opts$results_dir, "metadata.csv"))
  done_cells <- if (opts$resume && file.exists(paths$posterior)) {
    unique(read.csv(paths$posterior, stringsAsFactors = FALSE)[, c("design", "n")])
  } else {
    unlink(c(paths$fits, paths$params, paths$posterior))
    data.frame(design = character(0), n = integer(0))
  }

  write_csv(do.call(rbind, lapply(designs, function(d) {
    data.frame(design = d$key, id = d$id, label = d$label, p = ncol(d$Sigma),
               key_params = paste(d$key_params, collapse = ";"),
               key_truth = mean(population_invariants(d)[d$key_params]),
               stringsAsFactors = FALSE)
  })), paths$design)
  truths <- lapply(designs, population_invariants)

  cells <- expand.grid(n = opts$n_values, design = names(designs), stringsAsFactors = FALSE)
  cells <- cells[!paste(cells$design, cells$n) %in% paste(done_cells$design, done_cells$n), ,
                 drop = FALSE]
  if (!nrow(cells)) {
    cat("Nothing to do: every requested cell is already in", opts$results_dir, "\n")
    return(invisible())
  }
  total <- nrow(cells) * opts$reps
  cat(sprintf("magmaan %s | %d cells x %d reps | %d cores\n",
              as.character(utils::packageVersion("magmaanlab")), nrow(cells), opts$reps, opts$cores))
  t_start <- proc.time()[["elapsed"]]
  done <- 0L
  for (ci in seq_len(nrow(cells))) {
    d <- designs[[cells$design[[ci]]]]
    n <- cells$n[[ci]]
    spec <- model_spec(d$syntax)
    chunks <- split(seq_len(opts$reps),
                    cut(seq_len(opts$reps), min(opts$cores, opts$reps), labels = FALSE))
    out <- parallel::mclapply(chunks, function(reps) {
      lapply(reps, function(r) run_rep(d, spec, n, r, opts, truths[[d$key]]))
    }, mc.cores = opts$cores, mc.preschedule = TRUE)
    bad <- vapply(out, inherits, logical(1), "try-error")
    if (any(bad)) stop("worker failed: ", as.character(out[bad][[1L]]), call. = FALSE)
    reps_out <- unlist(out, recursive = FALSE)
    append_csv(do.call(rbind, lapply(reps_out, `[[`, "fits")), paths$fits)
    append_csv(do.call(rbind, lapply(reps_out, `[[`, "params")), paths$params)
    append_csv(do.call(rbind, lapply(reps_out, `[[`, "posterior")), paths$posterior)
    done <- done + opts$reps
    elapsed <- proc.time()[["elapsed"]] - t_start
    cat(sprintf("[%2d/%d] %-11s N=%-4d  %7.1fs elapsed, ETA %7.1fs\n",
                ci, nrow(cells), d$key, n, elapsed, elapsed / done * (total - done)))
  }

  opts_s <- sampler_defaults()
  write_metadata(paths$metadata, values = list(
    profile = opts$profile, reps = opts$reps,
    n_values = paste(opts$n_values, collapse = ","),
    designs = paste(opts$designs, collapse = ","),
    seed_base = opts$seed_base, cores = opts$cores, resumed = opts$resume,
    chains = opts_s$chains, burn = opts_s$burn, keep = opts_s$keep, thin = opts_s$thin,
    boundary_tol = boundary_tol,
    magmaan_git_head = magmaan_cache_ref()$git_head,
    magmaan_git_dirty = magmaan_cache_ref()$git_dirty),
    packages = "magmaanlab")
  cat("Wrote:\n", paste0("  ", unlist(paths), "\n"), sep = "")
}

main()
