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
require_pkg("jsonlite")
require_pkg("magmaan", "install the current R package first")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Describe four covariance-repair cases and compare ordinary NTML with\n",
  "PSD-ML in a controlled primitive-covariance boundary experiment.\n\n",
  "Profiles:\n",
  "  --smoke   N = 30,100; lambda = 0,.01,.20; 5 replications (default)\n",
  "  --pilot   N = 25,50,100,250; five lambdas; 200 replications\n",
  "  --full    same grid; 1000 replications\n\n",
  "Options:\n",
  "  --reps N --n-values 25,50 --lambda-values 0,.01,.20\n",
  "  --seed-base N --max-iter N --results-dir PATH --progress-every N\n",
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
    lambda_values = NULL,
    seed_base = 20260731L,
    max_iter = 5000L,
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
    else if (arg == "--lambda-values") {
      out$lambda_values <- parse_csv_numeric(take())
    } else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--max-iter") out$max_iter <- as.integer(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else if (arg == "--progress-every") {
      out$progress_every <- as.integer(take())
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }

  common_n <- c(25L, 50L, 100L, 250L)
  common_lambda <- c(0, 0.001, 0.01, 0.05, 0.20)
  defaults <- switch(
    out$profile,
    smoke = list(
      reps = 5L, n_values = c(30L, 100L),
      lambda_values = c(0, 0.01, 0.20)),
    pilot = list(
      reps = 200L, n_values = common_n,
      lambda_values = common_lambda),
    full = list(
      reps = 1000L, n_values = common_n,
      lambda_values = common_lambda)
  )
  if (is.null(out$reps)) out$reps <- defaults$reps
  if (is.null(out$n_values)) out$n_values <- defaults$n_values
  if (is.null(out$lambda_values)) {
    out$lambda_values <- defaults$lambda_values
  }
  out$n_values <- sort(unique(out$n_values))
  out$lambda_values <- sort(unique(out$lambda_values))
  ints <- c(out$reps, out$n_values, out$seed_base, out$max_iter,
            out$progress_every)
  if (anyNA(ints) || any(!is.finite(ints)) || out$reps < 1L ||
      any(out$n_values <= 3L) || out$seed_base < 1L ||
      out$max_iter < 1L || out$progress_every < 1L ||
      !length(out$lambda_values) || any(!is.finite(out$lambda_values)) ||
      any(out$lambda_values < 0) || any(out$lambda_values > 1)) {
    stop("invalid profile values; N must exceed 3 and lambda must be in [0,1]",
         call. = FALSE)
  }
  out
}

matrix_rows <- function(x) {
  do.call(rbind, lapply(x, function(row) as.numeric(unlist(row))))
}

read_corpus_cases <- function() {
  payload <- jsonlite::read_json(
    experiment_path("inputs", "corpus_cases.json"), simplifyVector = FALSE)
  lapply(payload$cases, function(x) {
    S <- matrix_rows(x$sample_cov)
    observed <- as.character(unlist(x$observed_names, use.names = FALSE))
    dimnames(S) <- list(observed, observed)
    mu <- as.numeric(unlist(x$sample_mean, use.names = FALSE))
    if (!length(mu)) mu <- NULL
    if (!is.null(mu)) names(mu) <- observed
    list(
      set = x$set,
      id = x$id,
      geometry = x$geometry,
      model = x$model,
      fixed_x = isTRUE(x$fixed_x),
      meanstructure = isTRUE(x$meanstructure),
      stats = list(
        S = list(S),
        mean = if (is.null(mu)) NULL else list(mu),
        nobs = as.integer(x$n_obs)),
      ordinary_start = as.numeric(unlist(x$ordinary_start, use.names = FALSE))
    )
  })
}

with_theta_start <- function(spec, theta) {
  out <- spec
  rows <- which(out$partable$free > 0L)
  if (length(theta) != max(out$partable$free)) {
    stop("start vector does not match the free-parameter count", call. = FALSE)
  }
  out$partable$ustart[rows] <- theta[out$partable$free[rows]]
  out
}

with_fit_start <- function(spec, fit) {
  with_theta_start(spec, as.numeric(fit$theta))
}

audit_converged <- function(fit) {
  is.list(fit) && is.list(fit$audit) &&
    identical(fit$audit$advisory_status, "converged")
}

admissible <- function(fit) {
  is.list(fit) && is.list(fit$diagnostics) &&
    is.list(fit$diagnostics$admissibility) &&
    isTRUE(fit$diagnostics$admissibility$admissible)
}

block_min <- function(fit, kind) {
  if (!is.list(fit) || !is.list(fit$diagnostics) ||
      !is.list(fit$diagnostics$admissibility)) return(NA_real_)
  blocks <- fit$diagnostics$admissibility[[kind]]
  if (!length(blocks)) return(NA_real_)
  values <- vapply(blocks, function(x) {
    value <- x$min_eigenvalue
    if (is.null(value) || !length(value)) NA_real_ else as.numeric(value[[1L]])
  }, numeric(1))
  values <- values[is.finite(values)]
  if (length(values)) min(values) else NA_real_
}

timed_fit <- function(expr) {
  started <- Sys.time()
  value <- tryCatch(suppressWarnings(force(expr)), error = identity)
  elapsed_ms <- 1000 * as.numeric(difftime(
    Sys.time(), started, units = "secs"))
  list(
    fit = if (inherits(value, "error")) NULL else value,
    error = if (inherits(value, "error")) conditionMessage(value) else "",
    elapsed_ms = elapsed_ms
  )
}

implied_sigma <- function(fit) {
  if (!is.list(fit)) return(NULL)
  value <- tryCatch(magmaan_core$model_implied(fit)$sigma[[1L]],
                    error = function(e) NULL)
  if (is.null(value)) NULL else as.matrix(value)
}

relative_frobenius <- function(estimate, truth) {
  if (is.null(estimate) || is.null(truth) ||
      any(!is.finite(estimate)) || any(!is.finite(truth))) return(NA_real_)
  norm(estimate - truth, "F") / max(norm(truth, "F"), .Machine$double.eps)
}

normal_kl <- function(truth, estimate) {
  if (is.null(estimate) || any(!is.finite(estimate))) return(NA_real_)
  ch <- tryCatch(chol(estimate), error = function(e) NULL)
  if (is.null(ch)) return(NA_real_)
  inverse <- chol2inv(ch)
  logdet_estimate <- 2 * sum(log(diag(ch)))
  ch_truth <- chol(truth)
  logdet_truth <- 2 * sum(log(diag(ch_truth)))
  value <- 0.5 * (
    sum(diag(inverse %*% truth)) - nrow(truth) +
      logdet_estimate - logdet_truth)
  max(0, as.numeric(value))
}

largest_parameter_label <- function(fit, delta) {
  if (!length(delta) || all(!is.finite(delta))) return("")
  k <- which.max(abs(delta))
  rows <- which(fit$partable$free == k)
  if (!length(rows)) return(paste0("free#", k))
  r <- rows[[1L]]
  paste0(fit$partable$lhs[[r]], fit$partable$op[[r]],
         fit$partable$rhs[[r]])
}

run_corpus_anatomy <- function(control) {
  cases <- read_corpus_cases()
  rows <- vector("list", length(cases))
  changes <- list()
  for (i in seq_along(cases)) {
    x <- cases[[i]]
    message("  corpus ", x$set, "::", x$id)
    spec <- model_spec(
      x$model, fixed_x = x$fixed_x, meanstructure = x$meanstructure)
    spec <- with_theta_start(spec, x$ordinary_start)
    ordinary_result <- timed_fit(magmaan_core$fit_ml(
      spec, x$stats, optimizer = "nlopt-lbfgs", control = control))
    ordinary <- ordinary_result$fit
    psd_result <- if (is.null(ordinary)) {
      timed_fit(frontier_fit_ml_psd(
        spec, x$stats, optimizer = "nlopt-slsqp", control = control))
    } else {
      timed_fit(frontier_fit_ml_psd(
        with_fit_start(spec, ordinary), x$stats,
        optimizer = "nlopt-slsqp", control = control))
    }
    psd <- psd_result$fit
    ordinary_sigma <- implied_sigma(ordinary)
    psd_sigma <- implied_sigma(psd)
    delta <- if (is.null(ordinary) || is.null(psd)) numeric() else
      as.numeric(psd$theta) - as.numeric(ordinary$theta)
    delta_f <- if (is.null(ordinary) || is.null(psd)) NA_real_ else
      as.numeric(psd$fmin - ordinary$fmin)

    if (length(delta)) {
      keep <- which(ordinary$partable$free > 0L)
      changes[[length(changes) + 1L]] <- data.frame(
        set = x$set,
        id = x$id,
        geometry = x$geometry,
        lhs = ordinary$partable$lhs[keep],
        op = ordinary$partable$op[keep],
        rhs = ordinary$partable$rhs[keep],
        free = ordinary$partable$free[keep],
        ordinary = ordinary$partable$est[keep],
        psd = psd$partable$est[keep],
        change = psd$partable$est[keep] - ordinary$partable$est[keep],
        stringsAsFactors = FALSE)
    }

    rows[[i]] <- data.frame(
      set = x$set,
      id = x$id,
      geometry = x$geometry,
      n = sum(x$stats$nobs),
      ordinary_returned = !is.null(ordinary),
      ordinary_converged = audit_converged(ordinary),
      ordinary_admissible = admissible(ordinary),
      ordinary_theta_min = block_min(ordinary, "theta"),
      ordinary_psi_min = block_min(ordinary, "psi"),
      ordinary_fmin = if (is.null(ordinary)) NA_real_ else ordinary$fmin,
      ordinary_elapsed_ms = ordinary_result$elapsed_ms,
      psd_returned = !is.null(psd),
      psd_converged = audit_converged(psd),
      psd_admissible = admissible(psd),
      psd_theta_min = block_min(psd, "theta"),
      psd_psi_min = block_min(psd, "psi"),
      psd_fmin = if (is.null(psd)) NA_real_ else psd$fmin,
      psd_elapsed_ms = psd_result$elapsed_ms,
      fmin_increase = delta_f,
      lr_scale_increase = 2 * sum(x$stats$nobs) * delta_f,
      max_abs_parameter_change = if (length(delta)) max(abs(delta)) else
        NA_real_,
      largest_changed_parameter = if (length(delta))
        largest_parameter_label(ordinary, delta) else "",
      implied_sigma_relative_distance = relative_frobenius(
        psd_sigma, ordinary_sigma),
      ordinary_error = ordinary_result$error,
      psd_error = psd_result$error,
      stringsAsFactors = FALSE)
  }
  list(
    cases = do.call(rbind, rows),
    changes = if (length(changes)) do.call(rbind, changes) else data.frame())
}

risk_model_syntax <- function() paste(
  "f1 =~ 1*x1",
  "f2 =~ 1*x2",
  "f3 =~ 1*x3",
  "x1 ~~ 0.2*x1",
  "x2 ~~ 0.2*x2",
  "x3 ~~ 0.2*x3",
  "f1 ~~ f1 + f2 + f3",
  "f2 ~~ f2 + f3",
  "f3 ~~ f3",
  sep = "\n")

risk_population <- function(lambda_min) {
  Q <- cbind(
    c(1, 1, 1) / sqrt(3),
    c(1, -1, 0) / sqrt(2),
    c(1, 1, -2) / sqrt(6))
  psi <- Q %*% diag(c(1, 0.4, lambda_min)) %*% t(Q)
  sigma <- psi + diag(0.2, 3L)
  list(psi = psi, sigma = sigma,
       rank = if (lambda_min == 0) 2L else 3L)
}

simulate_risk_data <- function(n, sigma, seed) {
  set.seed(seed)
  out <- matrix(stats::rnorm(n * ncol(sigma)), nrow = n) %*% chol(sigma)
  out <- as.data.frame(out)
  names(out) <- paste0("x", seq_len(ncol(sigma)))
  out
}

extract_risk_psi <- function(fit) {
  if (!is.list(fit) || !is.data.frame(fit$partable)) return(NULL)
  latent <- paste0("f", 1:3)
  out <- matrix(NA_real_, 3L, 3L, dimnames = list(latent, latent))
  pt <- fit$partable
  for (i in seq_along(latent)) {
    for (j in i:length(latent)) {
      hit <- which(
        pt$op == "~~" &
          ((pt$lhs == latent[[i]] & pt$rhs == latent[[j]]) |
           (pt$lhs == latent[[j]] & pt$rhs == latent[[i]])))
      if (length(hit) != 1L) return(NULL)
      out[i, j] <- out[j, i] <- as.numeric(pt$est[[hit]])
    }
  }
  out
}

matrix_rank <- function(x) {
  if (is.null(x) || any(!is.finite(x))) return(NA_integer_)
  values <- eigen(x, symmetric = TRUE, only.values = TRUE)$values
  tolerance <- 1e-6 * max(1, max(abs(values)))
  sum(values > tolerance)
}

negative_eigen_count <- function(x) {
  if (is.null(x) || any(!is.finite(x))) return(NA_integer_)
  values <- eigen(x, symmetric = TRUE, only.values = TRUE)$values
  tolerance <- 1e-8 * max(1, max(abs(values)))
  sum(values < -tolerance)
}

risk_method_row <- function(method, result, population, n, rep, seed,
                            lambda_min) {
  fit <- result$fit
  psi <- extract_risk_psi(fit)
  sigma <- implied_sigma(fit)
  stationary <- audit_converged(fit)
  fit_admissible <- admissible(fit)
  analysis_ok <- stationary && !is.null(psi) && !is.null(sigma) &&
    (method == "ordinary" || fit_admissible)
  data.frame(
    n = n,
    lambda_min = lambda_min,
    true_rank = population$rank,
    rep = rep,
    seed = seed,
    method = method,
    returned = !is.null(fit),
    solver_converged = !is.null(fit) && isTRUE(fit$converged),
    audit_converged = stationary,
    admissible = fit_admissible,
    analysis_ok = analysis_ok,
    covariance_boundary = !is.null(fit) &&
      is.finite(block_min(fit, "psi")) && block_min(fit, "psi") <= 1e-6,
    psi_min_eigenvalue = block_min(fit, "psi"),
    estimated_rank = matrix_rank(psi),
    negative_eigen_count = negative_eigen_count(psi),
    psi_relative_error = if (analysis_ok)
      relative_frobenius(psi, population$psi) else NA_real_,
    sigma_relative_error = if (analysis_ok)
      relative_frobenius(sigma, population$sigma) else NA_real_,
    population_kl = if (analysis_ok)
      normal_kl(population$sigma, sigma) else NA_real_,
    fmin = if (is.null(fit)) NA_real_ else as.numeric(fit$fmin),
    elapsed_ms = result$elapsed_ms,
    iterations = if (is.null(fit)) NA_integer_ else as.integer(fit$iterations),
    f_evals = if (is.null(fit)) NA_integer_ else as.integer(fit$f_evals),
    optimizer_status = if (is.null(fit)) "" else
      as.character(fit$optimizer_status),
    error = result$error,
    stringsAsFactors = FALSE)
}

risk_pair_row <- function(ordinary_row, psd_row) {
  both <- ordinary_row$analysis_ok && psd_row$analysis_ok
  data.frame(
    n = ordinary_row$n,
    lambda_min = ordinary_row$lambda_min,
    true_rank = ordinary_row$true_rank,
    rep = ordinary_row$rep,
    seed = ordinary_row$seed,
    ordinary_admissible = ordinary_row$admissible,
    psd_usable = psd_row$analysis_ok,
    psd_boundary = psd_row$covariance_boundary,
    both_analysis_ok = both,
    fmin_increase = if (both) psd_row$fmin - ordinary_row$fmin else NA_real_,
    lr_scale_increase = if (both)
      2 * ordinary_row$n * (psd_row$fmin - ordinary_row$fmin) else NA_real_,
    psi_error_improvement = if (both)
      ordinary_row$psi_relative_error - psd_row$psi_relative_error else
      NA_real_,
    sigma_error_improvement = if (both)
      ordinary_row$sigma_relative_error - psd_row$sigma_relative_error else
      NA_real_,
    kl_improvement = if (both)
      ordinary_row$population_kl - psd_row$population_kl else NA_real_,
    psd_rank_correct = psd_row$analysis_ok &&
      psd_row$estimated_rank == psd_row$true_rank,
    time_ratio = psd_row$elapsed_ms / ordinary_row$elapsed_ms,
    stringsAsFactors = FALSE)
}

mean_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) mean(x) else NA_real_
}

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

summarize_risk_method <- function(x) {
  ok <- x$analysis_ok
  data.frame(
    n = x$n[[1L]],
    lambda_min = x$lambda_min[[1L]],
    method = x$method[[1L]],
    reps = nrow(x),
    audit_convergence_rate = mean(x$audit_converged),
    admissible_rate = mean(x$admissible),
    analysis_rate = mean(x$analysis_ok),
    covariance_boundary_rate = mean(x$covariance_boundary),
    rank_correct_rate = mean(x$estimated_rank[ok] == x$true_rank[ok]),
    mean_psi_relative_error = mean_or_na(x$psi_relative_error[ok]),
    mean_sigma_relative_error = mean_or_na(x$sigma_relative_error[ok]),
    mean_population_kl = mean_or_na(x$population_kl[ok]),
    median_min_eigenvalue = median_or_na(x$psi_min_eigenvalue[ok]),
    median_elapsed_ms = median_or_na(x$elapsed_ms),
    stringsAsFactors = FALSE)
}

summarize_risk_pair <- function(x) {
  ok <- x$both_analysis_ok
  repaired <- ok & !x$ordinary_admissible
  data.frame(
    n = x$n[[1L]],
    lambda_min = x$lambda_min[[1L]],
    reps = nrow(x),
    ordinary_inadmissible_rate = mean(!x$ordinary_admissible),
    psd_usable_rate = mean(x$psd_usable),
    psd_boundary_rate = mean(x$psd_boundary),
    psd_rank_correct_rate = mean(x$psd_rank_correct[x$psd_usable]),
    mean_fmin_increase = mean_or_na(x$fmin_increase[ok]),
    median_lr_scale_increase = median_or_na(x$lr_scale_increase[ok]),
    mean_psi_error_improvement = mean_or_na(x$psi_error_improvement[ok]),
    mean_sigma_error_improvement = mean_or_na(
      x$sigma_error_improvement[ok]),
    mean_kl_improvement = mean_or_na(x$kl_improvement[ok]),
    repaired_mean_psi_error_improvement = mean_or_na(
      x$psi_error_improvement[repaired]),
    repaired_mean_sigma_error_improvement = mean_or_na(
      x$sigma_error_improvement[repaired]),
    repaired_mean_kl_improvement = mean_or_na(x$kl_improvement[repaired]),
    repaired_psd_better_psi_rate = mean_or_na(
      as.numeric(x$psi_error_improvement[repaired] > 1e-10)),
    repaired_psd_better_sigma_rate = mean_or_na(
      as.numeric(x$sigma_error_improvement[repaired] > 1e-10)),
    repaired_psd_better_kl_rate = mean_or_na(
      as.numeric(x$kl_improvement[repaired] > 1e-10)),
    median_time_ratio = median_or_na(x$time_ratio),
    stringsAsFactors = FALSE)
}

run_risk_grid <- function(args, output_dir, control) {
  model <- model_spec(risk_model_syntax())
  raw_path <- file.path(output_dir, "risk_raw.csv")
  pair_path <- file.path(output_dir, "risk_pairs.csv")
  if (file.exists(raw_path)) unlink(raw_path)
  if (file.exists(pair_path)) unlink(pair_path)
  cells <- expand.grid(
    n = args$n_values,
    lambda_index = seq_along(args$lambda_values),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE)
  cells$lambda_min <- args$lambda_values[cells$lambda_index]
  cells <- cells[order(cells$n, cells$lambda_min), ]
  total <- nrow(cells) * args$reps
  started <- Sys.time()

  for (cell_index in seq_len(nrow(cells))) {
    n <- cells$n[[cell_index]]
    lambda_index <- cells$lambda_index[[cell_index]]
    lambda_min <- cells$lambda_min[[cell_index]]
    population <- risk_population(lambda_min)
    raw_rows <- vector("list", 2L * args$reps)
    pair_rows <- vector("list", args$reps)
    for (rep in seq_len(args$reps)) {
      seed <- args$seed_base + 100003L * n + 1009L * lambda_index + rep
      x <- simulate_risk_data(n, population$sigma, seed)
      data <- df_to_data(x, model, scaling = "n-1")
      ordinary_result <- timed_fit(magmaan_core$fit_ml(
        model, data, optimizer = "nlopt-lbfgs", control = control))
      psd_spec <- if (is.null(ordinary_result$fit)) model else
        with_fit_start(model, ordinary_result$fit)
      psd_result <- timed_fit(frontier_fit_ml_psd(
        psd_spec, data, optimizer = "nlopt-slsqp", control = control))
      ordinary_row <- risk_method_row(
        "ordinary", ordinary_result, population, n, rep, seed, lambda_min)
      psd_row <- risk_method_row(
        "psd", psd_result, population, n, rep, seed, lambda_min)
      raw_rows[[2L * rep - 1L]] <- ordinary_row
      raw_rows[[2L * rep]] <- psd_row
      pair_rows[[rep]] <- risk_pair_row(ordinary_row, psd_row)
    }
    append_csv(do.call(rbind, raw_rows), raw_path)
    append_csv(do.call(rbind, pair_rows), pair_path)
    if (cell_index %% args$progress_every == 0L || cell_index == nrow(cells)) {
      elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      complete <- cell_index * args$reps
      eta <- elapsed * (total - complete) / max(complete, 1L)
      message(sprintf(
        "  risk N=%d lambda=%g (%d/%d datasets), elapsed %.1fs, ETA %.1fs",
        n, lambda_min, complete, total, elapsed, eta))
    }
  }

  raw <- read.csv(raw_path, stringsAsFactors = FALSE, check.names = FALSE)
  pairs <- read.csv(pair_path, stringsAsFactors = FALSE, check.names = FALSE)
  method_key <- interaction(
    raw$n, raw$lambda_min, raw$method, drop = TRUE, lex.order = TRUE)
  method_summary <- do.call(
    rbind, lapply(split(raw, method_key), summarize_risk_method))
  pair_key <- interaction(
    pairs$n, pairs$lambda_min, drop = TRUE, lex.order = TRUE)
  pair_summary <- do.call(
    rbind, lapply(split(pairs, pair_key), summarize_risk_pair))
  row.names(method_summary) <- NULL
  row.names(pair_summary) <- NULL
  method_summary <- method_summary[
    order(method_summary$n, method_summary$lambda_min,
          method_summary$method), ]
  pair_summary <- pair_summary[
    order(pair_summary$n, pair_summary$lambda_min), ]
  write_csv(method_summary, file.path(output_dir, "risk_method_summary.csv"))
  write_csv(pair_summary, file.path(output_dir, "risk_pair_summary.csv"))
  list(raw = raw, pairs = pairs, method_summary = method_summary,
       pair_summary = pair_summary)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  output_dir <- if (is.null(args$results_dir)) {
    experiment_path("results", args$profile)
  } else {
    normalizePath(args$results_dir, mustWork = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  control <- list(max_iter = args$max_iter, gtol = 1e-8, ftol = 1e-12)
  started <- Sys.time()

  message("PSD-ML repair anatomy: four deterministic corpus cases")
  corpus <- run_corpus_anatomy(control)
  write_csv(corpus$cases, file.path(output_dir, "corpus_anatomy.csv"))
  write_csv(corpus$changes, file.path(output_dir, "corpus_parameter_changes.csv"))

  message(
    "PSD-ML near-boundary risk: ", length(args$n_values), " N values, ",
    length(args$lambda_values), " eigenvalues, ", args$reps,
    " replications")
  risk <- run_risk_grid(args, output_dir, control)
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  ref <- magmaan_cache_ref()
  write_metadata(
    file.path(output_dir, "metadata.csv"),
    values = list(
      experiment = "76-psd-ml-repair-risk",
      profile = args$profile,
      reps = args$reps,
      n_values = args$n_values,
      lambda_values = args$lambda_values,
      seed_base = args$seed_base,
      max_iter = args$max_iter,
      risk_residual_variance = 0.2,
      risk_other_psi_eigenvalues = c(1, 0.4),
      corpus_cases = nrow(corpus$cases),
      risk_datasets = nrow(risk$pairs),
      elapsed_sec = elapsed,
      magmaan_git_head = ref$git_head,
      magmaan_git_dirty = ref$git_dirty,
      ordinary_method = "complete-data NTML, NLopt L-BFGS",
      psd_method = "warm PSD-ML refit, NLopt SLSQP",
      lr_scale_note = "2*N*objective difference; descriptive, not a test"
    ),
    packages = c("magmaan", "jsonlite"))

  expected_pairs <- length(args$n_values) * length(args$lambda_values) *
    args$reps
  psd_rows <- risk$raw[risk$raw$method == "psd", ]
  valid <- nrow(corpus$cases) == 4L &&
    all(corpus$cases$ordinary_converged) &&
    all(!corpus$cases$ordinary_admissible) &&
    all(corpus$cases$psd_converged) &&
    all(corpus$cases$psd_admissible) &&
    nrow(risk$pairs) == expected_pairs &&
    nrow(risk$raw) == 2L * expected_pairs &&
    !anyDuplicated(risk$raw[c("n", "lambda_min", "rep", "method")]) &&
    all(psd_rows$admissible[psd_rows$audit_converged])

  cat("\nCorpus repair anatomy:\n")
  print(corpus$cases[, c(
    "set", "id", "geometry", "lr_scale_increase",
    "max_abs_parameter_change", "implied_sigma_relative_distance")],
    row.names = FALSE, digits = 4)
  cat("\nNear-boundary paired summary:\n")
  print(risk$pair_summary[, c(
    "n", "lambda_min", "ordinary_inadmissible_rate",
    "psd_boundary_rate", "mean_psi_error_improvement",
    "mean_sigma_error_improvement", "mean_kl_improvement",
    "median_time_ratio")], row.names = FALSE, digits = 4)
  message("Wrote results to ", normalizePath(output_dir))
  if (!valid) quit(save = "no", status = 1L)
}

main()
