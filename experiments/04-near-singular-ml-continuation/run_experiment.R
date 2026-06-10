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
    target = "diagonal",
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
        "[--target diagonal] [--optimizer port] [--max-iter 1000] ",
        "[--seed 20260524]\n",
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
    else if (arg == "--target") out$target <- need_value()
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
  if (!out$target %in% c("diagonal", "scaled_identity", "identity")) {
    stop("target must be one of diagonal, scaled_identity, or identity",
         call. = FALSE)
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

fit_one <- function(method, spec, dat, optimizer, control, alphas, target) {
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
          alphas = alphas, target = target)
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

audit_field <- function(fit, name, default = NA) {
  if (is.null(fit$audit) || is.null(fit$audit[[name]])) return(default)
  fit$audit[[name]]
}

audit_converged <- function(fit) {
  stationary <- audit_field(fit, "stationary", NA)
  if (length(stationary) == 1L && !is.na(stationary)) {
    return(isTRUE(stationary))
  }
  isTRUE(fit$converged)
}

fit_row <- function(rep, seed, method, spec, dat, optimizer, control, alphas,
                    target, sample_diag) {
  got <- fit_one(method, spec, dat, optimizer, control, alphas, target)
  fit <- got$fit
  base <- data.frame(
    replicate = rep,
    seed = seed,
    method = method,
    target = if (identical(method, "baseline_ml")) NA_character_ else target,
    n = dat$nobs[[1L]],
    p = length(dat$ov_names[[1L]]),
    sample_min_eigen = sample_diag[["min"]],
    sample_condition = sample_diag[["condition"]],
    ok = !inherits(fit, "try-error"),
    converged = FALSE,
    optimizer_converged = FALSE,
    optimizer_status = NA_character_,
    fmin = NA_real_,
    grad_norm = NA_real_,
    audit_stationary = NA,
    audit_grad_inf_norm = NA_real_,
    audit_stationarity_rhs = NA_real_,
    audit_f_consistent = NA,
    audit_f_finite = NA,
    audit_advisory_status = NA_character_,
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

  base$converged <- audit_converged(fit)
  base$optimizer_converged <- isTRUE(fit$converged)
  base$optimizer_status <- fit$optimizer_status %||% NA_character_
  base$fmin <- fit$fmin %||% NA_real_
  base$grad_norm <- fit$grad_norm %||% NA_real_
  base$audit_stationary <- audit_field(fit, "stationary", NA)
  base$audit_grad_inf_norm <- audit_field(fit, "grad_inf_norm", NA_real_)
  base$audit_stationarity_rhs <- audit_field(fit, "stationarity_rhs", NA_real_)
  base$audit_f_consistent <- audit_field(fit, "f_consistent", NA)
  base$audit_f_finite <- audit_field(fit, "f_finite", NA)
  base$audit_advisory_status <- audit_field(fit, "advisory_status", NA_character_)
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
      optimizer_converged = sum(x$optimizer_converged, na.rm = TRUE),
      ok_rate = mean(x$ok, na.rm = TRUE),
      converged_rate = mean(x$converged, na.rm = TRUE),
      optimizer_converged_rate = mean(x$optimizer_converged, na.rm = TRUE),
      median_fmin = stats::median(x$fmin, na.rm = TRUE),
      median_audit_grad_inf_norm = stats::median(
        x$audit_grad_inf_norm, na.rm = TRUE),
      median_audit_stationarity_rhs = stats::median(
        x$audit_stationarity_rhs, na.rm = TRUE),
      median_total_f_evals = stats::median(x$total_f_evals, na.rm = TRUE),
      median_elapsed_sec = stats::median(x$elapsed_sec, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  ensure_results_dir()

  require_pkg("magmaan", "Install the local R package first, e.g. `just r-install`")

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
                           control, args$alphas, args$target, sample_diag)
    pos <- pos + 1L
    rows[[pos]] <- fit_row(r, seed, "ridge_continuation", spec, dat,
                           args$optimizer, control, args$alphas, args$target,
                           sample_diag)
    pos <- pos + 1L
  }

  rows <- do.call(rbind, rows)
  summary <- summarize_results(rows)
  write_csv(rows, experiment_path("results", "fits.csv"))
  write_csv(summary, experiment_path("results", "summary.csv"))
  write_metadata(
    experiment_path("results", "metadata.csv"),
    values = list(
      reps = args$reps,
      n = args$n,
      p = args$p,
      loading = args$loading,
      residual = args$residual,
      seed = args$seed,
      optimizer = args$optimizer,
      max_iter = args$max_iter,
      target = args$target,
      alphas = args$alphas,
      n_fit_rows = nrow(rows)
    ),
    packages = "magmaan"
  )

  print(summary, row.names = FALSE)
  cat("Wrote ", experiment_path("results", "fits.csv"), "\n", sep = "")
}

main()
