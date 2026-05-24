#!/usr/bin/env Rscript

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (length(x) == 1L && is.na(x)) return(y)
  x
}

repo_root <- function(start = getwd()) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "CMakeLists.txt")) &&
        file.exists(file.path(path, "r-package", "DESCRIPTION"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) stop("Could not find magmaan repository root")
    path <- parent
  }
}

experiment_dir <- function() {
  file.path(repo_root(), "experiments", "04-near_singular_ml_continuation")
}

experiment_path <- function(...) file.path(experiment_dir(), ...)

parse_csv_numeric <- function(x) {
  out <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  as.numeric(out[nzchar(out)])
}

parse_args <- function(args) {
  out <- list(
    reps = 50L,
    n = 8L,
    p = 6L,
    loading = 0.995,
    residual = 0.01,
    seed = 20260524L,
    optimizer = "port",
    max_iter = 1000L,
    alphas = c(0.50, 0.20, 0.10, 0.05, 0.01)
  )
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    value <- NULL
    if (arg %in% c("-h", "--help")) {
      cat(
        "Usage: Rscript run_experiment.R [--reps 50] [--n 8] [--p 6] ",
        "[--loading 0.995] [--residual 0.01] ",
        "[--alphas 0.5,0.2,0.1,0.05,0.01] ",
        "[--optimizer port] [--max-iter 1000] [--seed 20260524]\n",
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
    else if (arg == "--n") out$n <- as.integer(need_value())
    else if (arg == "--p") out$p <- as.integer(need_value())
    else if (arg == "--loading") out$loading <- as.numeric(need_value())
    else if (arg == "--residual") out$residual <- as.numeric(need_value())
    else if (arg == "--seed") out$seed <- as.integer(need_value())
    else if (arg == "--optimizer") out$optimizer <- need_value()
    else if (arg == "--max-iter") out$max_iter <- as.integer(need_value())
    else if (arg == "--alphas") out$alphas <- parse_csv_numeric(need_value())
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  if (anyNA(unlist(out[c("reps", "n", "p", "seed", "max_iter")]))) {
    stop("integer arguments must be finite", call. = FALSE)
  }
  if (out$reps < 1L || out$n < 2L || out$p < 3L || out$max_iter < 1L) {
    stop("reps/max_iter must be positive, n >= 2, and p >= 3", call. = FALSE)
  }
  if (!is.finite(out$loading) || !is.finite(out$residual) ||
      out$residual <= 0) {
    stop("loading must be finite and residual must be positive", call. = FALSE)
  }
  if (!length(out$alphas) || anyNA(out$alphas)) {
    stop("alphas must be a non-empty numeric CSV", call. = FALSE)
  }
  out
}

safe_condition <- function(S) {
  ev <- eigen((S + t(S)) / 2, symmetric = TRUE, only.values = TRUE)$values
  if (!length(ev)) return(c(min = NA_real_, max = NA_real_, condition = NA_real_))
  c(min = min(ev), max = max(ev),
    condition = if (min(ev) > 0) max(ev) / min(ev) else Inf)
}

model_syntax <- function(p) {
  paste("f =~", paste0("x", seq_len(p), collapse = " + "))
}

simulate_case <- function(n, p, loading, residual, seed) {
  set.seed(seed)
  lambda <- c(1, rep(loading, p - 1L))
  eta <- stats::rnorm(n)
  eps <- matrix(stats::rnorm(n * p, sd = sqrt(residual)), nrow = n, ncol = p)
  X <- eps + tcrossprod(eta, lambda)
  colnames(X) <- paste0("x", seq_len(p))
  as.data.frame(X)
}

fit_one <- function(method, spec, dat, optimizer, control, alphas) {
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  fit <- withCallingHandlers(
    try(
      if (identical(method, "baseline_ml")) {
        magmaan::magmaan_core$fit_ml(spec, dat, optimizer = optimizer,
                                     control = control)
      } else {
        magmaan::magmaan_core$frontier_fit_ml_ridge_continuation(
          spec, dat, optimizer = optimizer, control = control,
          alphas = alphas)
      },
      silent = TRUE
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  elapsed <- proc.time()[["elapsed"]] - started
  list(fit = fit, elapsed = elapsed, warning = paste(unique(warnings), collapse = " | "))
}

fit_row <- function(rep, seed, method, spec, dat, optimizer, control, alphas,
                    sample_diag) {
  got <- fit_one(method, spec, dat, optimizer, control, alphas)
  fit <- got$fit
  base <- data.frame(
    replicate = rep,
    seed = seed,
    method = method,
    n = dat$nobs[[1L]],
    p = length(dat$ov_names[[1L]]),
    sample_min_eigen = sample_diag[["min"]],
    sample_condition = sample_diag[["condition"]],
    ok = !inherits(fit, "try-error"),
    converged = FALSE,
    optimizer_status = NA_character_,
    fmin = NA_real_,
    grad_norm = NA_real_,
    iterations = NA_integer_,
    f_evals = NA_integer_,
    g_evals = NA_integer_,
    total_iterations = NA_integer_,
    total_f_evals = NA_integer_,
    total_g_evals = NA_integer_,
    n_stages = NA_integer_,
    first_stage_condition = NA_real_,
    elapsed_sec = got$elapsed,
    warning = got$warning,
    error = if (inherits(fit, "try-error")) as.character(fit) else "",
    stringsAsFactors = FALSE
  )
  if (inherits(fit, "try-error")) return(base)

  base$converged <- isTRUE(fit$converged)
  base$optimizer_status <- fit$optimizer_status %||% NA_character_
  base$fmin <- fit$fmin %||% NA_real_
  base$grad_norm <- fit$grad_norm %||% NA_real_
  base$iterations <- fit$iterations %||% NA_integer_
  base$f_evals <- fit$f_evals %||% NA_integer_
  base$g_evals <- fit$g_evals %||% NA_integer_
  if (!is.null(fit$continuation)) {
    path <- fit$continuation$path
    base$total_iterations <- fit$continuation$total_iterations
    base$total_f_evals <- fit$continuation$total_f_evals
    base$total_g_evals <- fit$continuation$total_g_evals
    base$n_stages <- nrow(path)
    base$first_stage_condition <- path$sample_condition[[1L]]
  } else {
    base$total_iterations <- base$iterations
    base$total_f_evals <- base$f_evals
    base$total_g_evals <- base$g_evals
    base$n_stages <- 1L
    base$first_stage_condition <- base$sample_condition
  }
  base
}

summarize_results <- function(rows) {
  split_rows <- split(rows, rows$method)
  out <- lapply(split_rows, function(x) {
    data.frame(
      method = x$method[[1L]],
      attempts = nrow(x),
      ok = sum(x$ok, na.rm = TRUE),
      converged = sum(x$converged, na.rm = TRUE),
      ok_rate = mean(x$ok, na.rm = TRUE),
      converged_rate = mean(x$converged, na.rm = TRUE),
      median_fmin = stats::median(x$fmin, na.rm = TRUE),
      median_grad_norm = stats::median(x$grad_norm, na.rm = TRUE),
      median_total_f_evals = stats::median(x$total_f_evals, na.rm = TRUE),
      median_elapsed_sec = stats::median(x$elapsed_sec, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(experiment_path("results"), showWarnings = FALSE, recursive = TRUE)

  if (!requireNamespace("magmaan", quietly = TRUE)) {
    stop("Install the local R package first, e.g. `just r-install`", call. = FALSE)
  }

  spec <- magmaan::model_spec(model_syntax(args$p))
  control <- list(max_iter = args$max_iter)
  rows <- vector("list", args$reps * 2L)
  pos <- 1L

  for (r in seq_len(args$reps)) {
    seed <- args$seed + r - 1L
    df <- simulate_case(args$n, args$p, args$loading, args$residual, seed)
    dat <- magmaan::df_to_data(df, spec)
    sample_diag <- safe_condition(dat$S[[1L]])
    rows[[pos]] <- fit_row(r, seed, "baseline_ml", spec, dat, args$optimizer,
                           control, args$alphas, sample_diag)
    pos <- pos + 1L
    rows[[pos]] <- fit_row(r, seed, "ridge_continuation", spec, dat,
                           args$optimizer, control, args$alphas, sample_diag)
    pos <- pos + 1L
  }

  rows <- do.call(rbind, rows)
  summary <- summarize_results(rows)
  write.csv(rows, experiment_path("results", "fits.csv"), row.names = FALSE)
  write.csv(summary, experiment_path("results", "summary.csv"), row.names = FALSE)

  print(summary, row.names = FALSE)
  cat("Wrote ", experiment_path("results", "fits.csv"), "\n", sep = "")
}

main()
