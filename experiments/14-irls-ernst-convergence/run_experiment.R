#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "ernst_helpers.R"))

parse_int_csv <- function(x) {
  out <- as.integer(parse_csv_arg(x))
  if (!length(out) || anyNA(out)) {
    stop("integer CSV argument is empty or contains non-integers", call. = FALSE)
  }
  out
}

parse_args <- function(args) {
  out <- list(
    reps = 200L,
    seed_base = 20260528L,
    n_values = 10L:100L,
    beta = 0.25,
    kbins = 12L,
    max_iter = 5000L,
    progress_every = 1L
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps N] [--seed-base N] ",
        "[--n-values 10,20,30] [--n-min N] [--n-max N] [--beta X] ",
        "[--kbins N] [--max-iter N] [--progress-every N] [--smoke]\n",
        "\n",
        "Defaults: --reps 200 --seed-base 20260528 --n-min 10 --n-max 100 ",
        "--beta 0.25 --kbins 12 --max-iter 5000 --progress-every 1\n",
        sep = ""
      )
      quit(save = "no", status = 0L)
    }
    if (grepl("=", arg, fixed = TRUE)) {
      parts <- strsplit(arg, "=", fixed = TRUE)[[1L]]
      arg <- parts[[1L]]
      value <- paste(parts[-1L], collapse = "=")
    }
    need_value <- function() {
      if (!is.null(value)) return(value)
      i <<- i + 1L
      if (i > length(args)) stop(arg, " requires a value", call. = FALSE)
      args[[i]]
    }
    if (arg == "--reps") out$reps <- as.integer(need_value())
    else if (arg == "--seed-base") out$seed_base <- as.integer(need_value())
    else if (arg == "--n-values") out$n_values <- parse_int_csv(need_value())
    else if (arg == "--n-min") {
      n_min <- as.integer(need_value())
      out$n_values <- n_min:max(out$n_values)
    } else if (arg == "--n-max") {
      n_max <- as.integer(need_value())
      out$n_values <- min(out$n_values):n_max
    } else if (arg == "--beta") out$beta <- as.numeric(need_value())
    else if (arg == "--kbins") out$kbins <- as.integer(need_value())
    else if (arg == "--max-iter") out$max_iter <- as.integer(need_value())
    else if (arg == "--progress-every") {
      out$progress_every <- as.integer(need_value())
    } else if (arg == "--smoke") {
      out$reps <- 1L
      out$n_values <- c(10L, 15L, 25L)
      out$progress_every <- 1L
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  int_fields <- c("reps", "seed_base", "kbins", "max_iter", "progress_every")
  if (anyNA(unlist(out[int_fields]))) {
    stop("integer arguments must be finite", call. = FALSE)
  }
  if (out$reps < 1L || out$kbins < 1L || out$max_iter < 1L ||
      out$progress_every < 1L) {
    stop("reps, kbins, max_iter, and progress_every must be positive",
         call. = FALSE)
  }
  out$n_values <- sort(unique(as.integer(out$n_values)))
  if (!length(out$n_values) || anyNA(out$n_values) || any(out$n_values < 7L)) {
    stop("n values must be integers >= 7", call. = FALSE)
  }
  if (!is.finite(out$beta)) stop("beta must be finite", call. = FALSE)
  out
}

fit_once <- function(method, spec, dat, control) {
  core <- magmaan::magmaan_core
  switch(method,
    `ml-lbfgs` = core$fit_ml(spec, dat, optimizer = "nlopt-lbfgs",
                             control = control),
    `ml-port` = core$fit_ml(spec, dat, optimizer = "port",
                            control = control),
    `ml-irls` = core$fit_ml_irls(spec, dat, optimizer = "port-nls",
                                 control = control),
    `ml-irls-snlls` = core$fit_ml_irls_snlls(spec, dat,
                                             optimizer = "port-nls",
                                             control = control),
    stop("unknown method: ", method, call. = FALSE)
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  require_pkg("magmaan", "Install the local R package first, e.g. `just r-install`")
  ensure_results_dir()

  raw_dir <- experiment_path("results", "raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

  control <- list(max_iter = args$max_iter, ftol = 1e-12, gtol = 1e-8)
  methods <- c("ml-lbfgs", "ml-port", "ml-irls", "ml-irls-snlls")
  grid <- ernst_design_grid(args$n_values, reps = args$reps, beta = args$beta)
  spec <- magmaan::model_spec(ernst_model())

  message("Ernst IRLS convergence: N cells = ", nrow(grid),
          ", reps = ", args$reps,
          ", fits = ", nrow(grid) * args$reps * length(methods),
          ", methods = ", paste(methods, collapse = ", "))

  started <- Sys.time()
  rows <- list()
  for (i in seq_len(nrow(grid))) {
    condition <- as.list(grid[i, , drop = FALSE])
    for (rep in seq_len(args$reps)) {
      rep_seed <- args$seed_base + 1000003L * as.integer(condition$cell) + rep
      x <- ernst_data(n = as.integer(condition$n), beta = args$beta,
                      seed = rep_seed)
      kappa_s <- kappa(stats::cov(x), exact = TRUE)
      dat <- magmaan::df_to_data(x, spec, scaling = "n-1")
      for (method in methods) {
        t0 <- proc.time()[["elapsed"]]
        fit <- tryCatch(fit_once(method, spec, dat, control), error = identity)
        wall_usec <- (proc.time()[["elapsed"]] - t0) * 1e6
        ok <- !inherits(fit, "error")
        diag <- if (ok) solution_diagnostics(fit) else
          list(improper = NA, n_neg_var = NA_integer_, min_var = NA_real_)
        audit <- if (ok) audit_fields(fit) else audit_fields(NULL)
        rows[[length(rows) + 1L]] <- data.frame(
          cell = as.integer(condition$cell),
          n = as.integer(condition$n),
          rep = rep,
          seed = rep_seed,
          beta = args$beta,
          kappa_s = kappa_s,
          method = method,
          iter = if (ok) as.integer(fit$iterations) else NA_integer_,
          f_evals = if (ok && !is.null(fit$f_evals)) as.integer(fit$f_evals)
            else NA_integer_,
          g_evals = if (ok && !is.null(fit$g_evals)) as.integer(fit$g_evals)
            else NA_integer_,
          fmin = if (ok) fit$fmin else NA_real_,
          wall_usec = wall_usec,
          improper = diag$improper,
          n_neg_var = as.integer(diag$n_neg_var),
          min_var = diag$min_var,
          optimizer_status = if (ok && !is.null(fit$optimizer_status))
            as.character(fit$optimizer_status) else NA_character_,
          audit_stationary = audit$audit_stationary,
          audit_grad_inf = audit$audit_grad_inf,
          audit_grad_scaled = audit$audit_grad_scaled,
          audit_grad_rhs = audit$audit_grad_rhs,
          audit_advisory = audit$audit_advisory,
          audit_f_consistent = audit$audit_f_consistent,
          diag_sigma_pd = audit$diag_sigma_pd,
          diag_lin_eq_ok = audit$diag_lin_eq_ok,
          diag_nl_eq_ok = audit$diag_nl_eq_ok,
          diag_snlls_fallback = audit$diag_snlls_fallback,
          error = if (ok) NA_character_ else conditionMessage(fit),
          stringsAsFactors = FALSE
        )
      }
    }
    if (i %% args$progress_every == 0L || i == nrow(grid)) {
      elapsed_now <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      done_fits <- i * args$reps * length(methods)
      total_fits <- nrow(grid) * args$reps * length(methods)
      rate <- if (elapsed_now > 0) done_fits / elapsed_now else NA_real_
      eta <- if (is.finite(rate) && rate > 0) (total_fits - done_fits) / rate
        else NA_real_
      message(sprintf(
        "  %d/%d N cells (%d/%d fits), elapsed %.1fs, ETA %.1fs",
        i, nrow(grid), done_fits, total_fits, elapsed_now, eta
      ))
    }
  }

  raw <- do.call(rbind, rows)
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  message("Finished ", nrow(raw), " fits in ", round(elapsed, 1), "s.")

  finite_kappa <- raw$kappa_s[is.finite(raw$kappa_s)]
  kbrks <- unique(stats::quantile(finite_kappa,
                                  probs = seq(0, 1,
                                              length.out = args$kbins + 1L)))
  raw$kbin <- cut(raw$kappa_s, breaks = kbrks, include.lowest = TRUE)

  summary_n <- summarize_by(raw, c("n", "method"))
  summary_n <- summary_n[order(summary_n$method, summary_n$n), ]
  summary_k <- summarize_by(raw, c("kbin", "method"))
  summary_k <- summary_k[order(summary_k$method, summary_k$kappa_med), ]

  ds_key <- paste(raw$cell, raw$rep, sep = "\r")
  dataset <- do.call(rbind, lapply(split(raw, ds_key), function(x) {
    ok <- x[is.na(x$error) | !nzchar(x$error), , drop = FALSE]
    imp <- ok$improper[!is.na(ok$improper)]
    fmin_rel_range <- {
      v <- ok$fmin[is.finite(ok$fmin)]
      if (length(v) < 2L) NA_real_ else
        (max(v) - min(v)) / max(abs(v), 1e-12)
    }
    data.frame(
      cell = x$cell[1L],
      n = x$n[1L],
      rep = x$rep[1L],
      seed = x$seed[1L],
      beta = x$beta[1L],
      kappa_s = x$kappa_s[1L],
      kbin = x$kbin[1L],
      n_success = nrow(ok),
      improper = if (length(imp)) imp[1L] else NA,
      improper_agree = length(unique(imp)) <= 1L,
      fmin_rel_range = fmin_rel_range,
      stringsAsFactors = FALSE
    )
  }))
  row.names(dataset) <- NULL

  fmin_tol <- 1e-2
  agree_frac <- function(v) {
    v <- v[!is.na(v)]
    if (!length(v)) NA_real_ else mean(v <= fmin_tol)
  }
  message(sprintf(
    paste0("Solution check: methods agree on fmin within 1%% in %.1f%% ",
           "of datasets; improper flag agrees in %.1f%%."),
    100 * agree_frac(dataset$fmin_rel_range),
    100 * mean(dataset$improper_agree, na.rm = TRUE)))

  improper_n <- do.call(rbind, lapply(split(dataset, dataset$n), function(x) {
    data.frame(n = x$n[1L], datasets = nrow(x),
               improper_rate = mean(x$improper, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }))
  improper_n <- improper_n[order(improper_n$n), ]

  write_csv(raw, file.path(raw_dir, "irls_ernst_convergence_raw.csv"))
  write_csv(dataset, file.path(raw_dir, "irls_ernst_convergence_datasets.csv"))
  write_csv(summary_n, file.path(results_dir(), "irls_ernst_convergence_summary.csv"))
  write_csv(summary_k, file.path(results_dir(), "irls_ernst_convergence_kappa.csv"))
  write_csv(improper_n, file.path(results_dir(), "irls_ernst_improper_by_n.csv"))
  write_metadata(
    file.path(results_dir(), "metadata.csv"),
    values = list(
      reps = args$reps,
      seed_base = args$seed_base,
      n_values = args$n_values,
      beta = args$beta,
      kbins = args$kbins,
      max_iter = args$max_iter,
      progress_every = args$progress_every,
      methods = methods,
      n_cells = nrow(grid),
      n_fit_rows = nrow(raw),
      elapsed_sec = elapsed
    ),
    packages = "magmaan"
  )

  message("Wrote ", file.path(raw_dir, "irls_ernst_convergence_raw.csv"))
  message("Wrote ", file.path(results_dir(), "irls_ernst_convergence_summary.csv"))
  message("Wrote ", file.path(results_dir(), "irls_ernst_convergence_kappa.csv"))
  message("Wrote ", file.path(results_dir(), "metadata.csv"))

  keep_n <- intersect(c(10L, 15L, 25L, 40L, 70L, 100L), args$n_values)
  cat("\nAudit-failure rate by N (per method):\n",
      "(solver_error_rate = hard errors; failure_rate = audit non-convergence)\n",
      sep = "")
  print(summary_n[summary_n$n %in% keep_n,
                  c("n", "method", "solver_error_rate", "failure_rate",
                    "convergence_rate", "median_fevals", "median_wall_usec")],
        row.names = FALSE, digits = 4)
  cat("\nImproper-solution rate by N (per dataset):\n")
  print(improper_n[improper_n$n %in% keep_n, ], row.names = FALSE,
        digits = 4)
}

main()
