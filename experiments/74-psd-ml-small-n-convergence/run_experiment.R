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
set_single_threaded_math()
require_pkg("magmaan", "install the current R package first")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Compare ordinary complete-data NTML with covariance-honest PSD-ML on\n",
  "the De Jonckere-Rosseel / Ernst six-indicator small-N design. Ordinary\n",
  "SLSQP is included to separate optimizer effects from parameterization.\n\n",
  "Profiles:\n",
  "  --smoke   N = 10,20,50; 3 replications (default)\n",
  "  --pilot   published N grid; 200 replications\n",
  "  --full    published N grid; 1000 replications\n\n",
  "Options:\n",
  "  --reps N --n-values 10,20,50 --seed-base N --max-iter N\n",
  paste0(
    "  --methods ml-lbfgs,ml-slsqp,psd-ml-slsqp,psd-ml-ipopt ",
    "--results-dir PATH\n"
  ),
  "  --progress-every N\n",
  sep = "")

parse_int_csv <- function(x) {
  out <- as.integer(parse_csv_arg(x))
  if (!length(out) || anyNA(out)) {
    stop("integer CSV argument is empty or invalid", call. = FALSE)
  }
  out
}

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    reps = NULL,
    n_values = NULL,
    seed_base = 20260729L,
    max_iter = 5000L,
    methods = c("ml-lbfgs", "ml-slsqp", "psd-ml-slsqp"),
    results_dir = NULL,
    progress_every = 1L
  )
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
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
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--max-iter") out$max_iter <- as.integer(take())
    else if (arg == "--methods") out$methods <- parse_csv_arg(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else if (arg == "--progress-every") {
      out$progress_every <- as.integer(take())
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }

  published_n <- c(10L, 15L, 20L, 25L, 30L, 40L,
                   50L, 60L, 70L, 80L, 90L, 100L)
  defaults <- switch(
    out$profile,
    smoke = list(reps = 3L, n_values = c(10L, 20L, 50L)),
    pilot = list(reps = 200L, n_values = published_n),
    full = list(reps = 1000L, n_values = published_n)
  )
  if (is.null(out$reps)) out$reps <- defaults$reps
  if (is.null(out$n_values)) out$n_values <- defaults$n_values

  allowed <- c(
    "ml-lbfgs", "ml-slsqp", "psd-ml-slsqp", "psd-ml-ipopt")
  if (!length(out$methods) || any(!out$methods %in% allowed)) {
    stop("methods must be selected from: ", paste(allowed, collapse = ", "),
         call. = FALSE)
  }
  out$methods <- unique(out$methods)
  out$n_values <- sort(unique(out$n_values))
  ints <- c(out$reps, out$n_values, out$seed_base, out$max_iter,
            out$progress_every)
  if (anyNA(ints) || any(!is.finite(ints)) || out$reps < 1L ||
      any(out$n_values <= 6L) || out$seed_base < 1L ||
      out$max_iter < 1L || out$progress_every < 1L) {
    stop("reps, seed, max-iter, and progress must be positive; N must exceed 6",
         call. = FALSE)
  }
  out
}

model_syntax <- function() {
  paste(
    "Y =~ y1 + y2 + y3",
    "X =~ x1 + x2 + x3",
    "Y ~ X",
    sep = "\n"
  )
}

simulate_data <- function(n, beta = 0.25, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  loadings <- c(1.0, 0.8, 0.6)
  eta_x <- stats::rnorm(n)
  eta_y <- beta * eta_x + stats::rnorm(n)
  epsilon <- matrix(stats::rnorm(n * 6L), nrow = n, ncol = 6L)
  out <- cbind(
    tcrossprod(eta_y, loadings),
    tcrossprod(eta_x, loadings)
  ) + epsilon
  out <- as.data.frame(out)
  names(out) <- c("y1", "y2", "y3", "x1", "x2", "x3")
  out
}

audit_converged <- function(fit) {
  is.list(fit) && is.list(fit$audit) &&
    identical(fit$audit$advisory_status, "converged")
}

audit_number <- function(fit, name) {
  if (!is.list(fit) || !is.list(fit$audit)) return(NA_real_)
  value <- fit$audit[[name]]
  if (is.null(value) || length(value) != 1L) return(NA_real_)
  as.numeric(value)
}

admissible <- function(fit) {
  is.list(fit) && is.list(fit$diagnostics) &&
    is.list(fit$diagnostics$admissibility) &&
    isTRUE(fit$diagnostics$admissibility$admissible)
}

minimum_eigenvalue <- function(fit) {
  if (!is.list(fit) || !is.list(fit$diagnostics) ||
      !is.list(fit$diagnostics$admissibility)) {
    return(NA_real_)
  }
  a <- fit$diagnostics$admissibility
  blocks <- c(a$theta, a$psi)
  values <- vapply(
    blocks,
    function(x) {
      value <- x$min_eigenvalue
      if (is.null(value) || !length(value)) NA_real_ else as.numeric(value[[1L]])
    },
    numeric(1)
  )
  values <- values[is.finite(values)]
  if (length(values)) min(values) else NA_real_
}

beta_estimate <- function(fit) {
  if (!is.list(fit) || !is.data.frame(fit$partable)) return(NA_real_)
  pt <- fit$partable
  hit <- which(pt$lhs == "Y" & pt$op == "~" & pt$rhs == "X")
  if (length(hit) == 1L) as.numeric(pt$est[[hit]]) else NA_real_
}

fit_method <- function(method, model, data, control) {
  switch(
    method,
    `ml-lbfgs` = magmaan_core$fit_ml(
      model, data, optimizer = "nlopt-lbfgs", control = control),
    `ml-slsqp` = magmaan_core$fit_ml(
      model, data, optimizer = "nlopt-slsqp", control = control),
    `psd-ml-slsqp` = frontier_fit_ml_psd(
      model, data, optimizer = "nlopt-slsqp", control = control),
    `psd-ml-ipopt` = frontier_fit_ml_psd(
      model, data, optimizer = "ipopt", control = control),
    stop("unknown method: ", method, call. = FALSE)
  )
}

fit_row <- function(method, model, data, n, rep, seed, kappa_s,
                    beta, control) {
  start <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    suppressWarnings(fit_method(method, model, data, control)),
    error = identity
  )
  elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - start)
  ok <- !inherits(fit, "error")
  stationary <- ok && audit_converged(fit)
  fit_admissible <- ok && admissible(fit)
  min_eigen <- if (ok) minimum_eigenvalue(fit) else NA_real_
  beta_hat <- if (ok) beta_estimate(fit) else NA_real_

  data.frame(
    n = n,
    rep = rep,
    seed = seed,
    method = method,
    kappa_s = kappa_s,
    returned = ok,
    solver_converged = ok && isTRUE(fit$converged),
    audit_converged = stationary,
    admissible = fit_admissible,
    usable = stationary && fit_admissible,
    covariance_boundary =
      ok && is.finite(min_eigen) && min_eigen <= 1e-6,
    min_covariance_eigenvalue = min_eigen,
    beta_hat = beta_hat,
    beta_error = beta_hat - beta,
    fmin = if (ok) as.numeric(fit$fmin) else NA_real_,
    stationarity_inf =
      if (ok) audit_number(fit, "grad_inf_norm") else NA_real_,
    raw_gradient_inf =
      if (ok) audit_number(fit, "raw_grad_inf_norm") else NA_real_,
    constraint_violation_inf =
      if (ok) audit_number(fit, "constraint_violation_inf") else NA_real_,
    constraint_jacobian_rank =
      if (ok) audit_number(fit, "constraint_jacobian_rank") else NA_real_,
    iterations = if (ok) as.integer(fit$iterations) else NA_integer_,
    f_evals = if (ok) as.integer(fit$f_evals) else NA_integer_,
    elapsed_ms = elapsed_ms,
    optimizer_status =
      if (ok) as.character(fit$optimizer_status) else NA_character_,
    error = if (ok) "" else conditionMessage(fit),
    stringsAsFactors = FALSE
  )
}

mean_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) mean(x) else NA_real_
}

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

summarize_method <- function(x) {
  usable_beta <- x$beta_error[x$usable & is.finite(x$beta_error)]
  data.frame(
    n = x$n[[1L]],
    method = x$method[[1L]],
    reps = nrow(x),
    return_rate = mean(x$returned),
    solver_convergence_rate = mean(x$solver_converged),
    audit_convergence_rate = mean(x$audit_converged),
    admissible_rate = mean(x$admissible),
    usable_rate = mean(x$usable),
    covariance_boundary_rate = mean(x$covariance_boundary),
    beta_bias = mean_or_na(usable_beta),
    beta_rmse = if (length(usable_beta)) sqrt(mean(usable_beta^2)) else NA_real_,
    median_f_evals = median_or_na(x$f_evals[x$returned]),
    median_elapsed_ms = median_or_na(x$elapsed_ms),
    stringsAsFactors = FALSE
  )
}

summarize_by_method <- function(raw) {
  key <- interaction(raw$n, raw$method, drop = TRUE, lex.order = TRUE)
  rows <- lapply(split(raw, key), summarize_method)
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$n, out$method), ]
}

pair_summary <- function(raw, baseline, alternative) {
  left <- raw[raw$method == baseline, ]
  right <- raw[raw$method == alternative, ]
  joined <- merge(
    left, right,
    by = c("n", "rep", "seed", "kappa_s"),
    suffixes = c("_baseline", "_alternative"),
    sort = FALSE
  )
  rows <- lapply(split(joined, joined$n), function(x) {
    data.frame(
      n = x$n[[1L]],
      baseline = baseline,
      alternative = alternative,
      reps = nrow(x),
      convergence_rescue_rate =
        mean(!x$audit_converged_baseline & x$audit_converged_alternative),
      convergence_harm_rate =
        mean(x$audit_converged_baseline & !x$audit_converged_alternative),
      usability_rescue_rate =
        mean(!x$usable_baseline & x$usable_alternative),
      usability_harm_rate =
        mean(x$usable_baseline & !x$usable_alternative),
      both_audit_converged_rate =
        mean(x$audit_converged_baseline & x$audit_converged_alternative),
      median_fmin_difference = median_or_na(
        x$fmin_alternative[
          x$audit_converged_baseline & x$audit_converged_alternative
        ] -
        x$fmin_baseline[
          x$audit_converged_baseline & x$audit_converged_alternative
        ]
      ),
      median_time_ratio = median_or_na(
        x$elapsed_ms_alternative / x$elapsed_ms_baseline
      ),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$n), ]
}

make_pairs <- function(raw) {
  pairs <- list()
  if (all(c("ml-lbfgs", "psd-ml-slsqp") %in% raw$method)) {
    pairs[[length(pairs) + 1L]] <-
      pair_summary(raw, "ml-lbfgs", "psd-ml-slsqp")
  }
  if (all(c("ml-slsqp", "psd-ml-slsqp") %in% raw$method)) {
    pairs[[length(pairs) + 1L]] <-
      pair_summary(raw, "ml-slsqp", "psd-ml-slsqp")
  }
  if (all(c("ml-lbfgs", "psd-ml-ipopt") %in% raw$method)) {
    pairs[[length(pairs) + 1L]] <-
      pair_summary(raw, "ml-lbfgs", "psd-ml-ipopt")
  }
  if (all(c("psd-ml-slsqp", "psd-ml-ipopt") %in% raw$method)) {
    pairs[[length(pairs) + 1L]] <-
      pair_summary(raw, "psd-ml-slsqp", "psd-ml-ipopt")
  }
  if (length(pairs)) do.call(rbind, pairs) else data.frame()
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  output_dir <- if (is.null(args$results_dir)) {
    experiment_path("results", args$profile)
  } else {
    normalizePath(args$results_dir, mustWork = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  raw_path <- file.path(output_dir, "raw.csv")
  if (file.exists(raw_path)) unlink(raw_path)

  beta <- 0.25
  model <- model_spec(model_syntax())
  control <- list(max_iter = args$max_iter, gtol = 1e-8, ftol = 1e-12)
  total <- length(args$n_values) * args$reps * length(args$methods)
  message(
    "PSD-ML small-N convergence: ", length(args$n_values), " N cells, ",
    args$reps, " replications, ", length(args$methods), " methods, ",
    total, " fits"
  )

  started <- Sys.time()
  completed <- 0L
  for (n_index in seq_along(args$n_values)) {
    n <- args$n_values[[n_index]]
    rows <- vector("list", args$reps * length(args$methods))
    row_index <- 0L
    for (rep in seq_len(args$reps)) {
      seed <- args$seed_base + 1000003L * n + rep
      x <- simulate_data(n, beta = beta, seed = seed)
      kappa_s <- kappa(stats::cov(x), exact = TRUE)
      data <- df_to_data(x, model, scaling = "n-1")
      method_order <- sample(args$methods, length(args$methods))
      for (method in method_order) {
        row_index <- row_index + 1L
        rows[[row_index]] <- fit_row(
          method, model, data, n, rep, seed, kappa_s, beta, control)
        completed <- completed + 1L
      }
    }
    append_csv(do.call(rbind, rows), raw_path)
    if (n_index %% args$progress_every == 0L ||
        n_index == length(args$n_values)) {
      elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      rate <- completed / max(elapsed, 1e-9)
      eta <- (total - completed) / max(rate, 1e-9)
      message(sprintf(
        "  N=%d (%d/%d fits), elapsed %.1fs, ETA %.1fs",
        n, completed, total, elapsed, eta
      ))
    }
  }

  raw <- read.csv(raw_path, stringsAsFactors = FALSE, check.names = FALSE)
  summary <- summarize_by_method(raw)
  paired <- make_pairs(raw)
  failures <- raw[!raw$returned, c(
    "n", "rep", "seed", "method", "elapsed_ms", "error")]

  write_csv(summary, file.path(output_dir, "summary.csv"))
  write_csv(paired, file.path(output_dir, "paired.csv"))
  write_csv(failures, file.path(output_dir, "failures.csv"))
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  ref <- magmaan_cache_ref()
  write_metadata(
    file.path(output_dir, "metadata.csv"),
    values = list(
      experiment = "74-psd-ml-small-n-convergence",
      profile = args$profile,
      reps = args$reps,
      n_values = args$n_values,
      seed_base = args$seed_base,
      beta = beta,
      max_iter = args$max_iter,
      methods = args$methods,
      n_fit_rows = nrow(raw),
      elapsed_sec = elapsed,
      magmaan_git_head = ref$git_head,
      magmaan_git_dirty = ref$git_dirty,
      ordinary_start = "current FABIN3 default",
      psd_start = "same start, covariance blocks projected to the PSD cone"
    ),
    packages = "magmaan"
  )

  expected <- length(args$n_values) * args$reps * length(args$methods)
  psd_audit_ok <- raw$method == "psd-ml-slsqp" & raw$audit_converged
  valid <- nrow(raw) == expected &&
    !anyDuplicated(raw[c("n", "rep", "method")]) &&
    all(raw$admissible[psd_audit_ok])

  keep_n <- intersect(c(10L, 20L, 50L, 100L), args$n_values)
  cat("\nConvergence and converged-plus-admissible rates by method:\n")
  print(
    summary[summary$n %in% keep_n, c(
      "n", "method", "audit_convergence_rate", "admissible_rate",
      "usable_rate", "covariance_boundary_rate", "beta_rmse",
      "median_elapsed_ms")],
    row.names = FALSE, digits = 4
  )
  if (nrow(paired)) {
    cat("\nPaired PSD-ML rescue/harm rates:\n")
    print(
      paired[paired$n %in% keep_n, c(
        "n", "baseline", "alternative", "convergence_rescue_rate",
        "convergence_harm_rate", "usability_rescue_rate",
        "usability_harm_rate", "median_time_ratio")],
      row.names = FALSE, digits = 4
    )
  }
  message("Wrote results to ", normalizePath(output_dir))
  if (!valid) quit(save = "no", status = 1L)
}

main()
