#!/usr/bin/env Rscript
# Small magmaan-owned bias check for the frontier reduced-bias estimators.
#
# The design is deliberately small: one correctly specified one-factor CFA,
# one sample size, and four estimator families (complete-data ML, continuous
# GLS, raw-data FIML with light MCAR missingness, and all-ordinal DWLS). Each
# replicate fits the
# ordinary estimator, then the explicit post-hoc RBM correction and the implicit
# integrated RBM estimator from the ordinary fit.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--ref-n N]
#                            [--estimators ml,gls,fiml,dwls]
#                            [--cores N|auto] [--chunk-size N]
#                            [--results-dir PATH] [--seed-base S] [--smoke]

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

`%||%` <- function(a, b) if (is.null(a)) b else a

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (nm in miss) x[[nm]] <- NA
    x[, cols, drop = FALSE]
  })
  out <- do.call(rbind, xs)
  rownames(out) <- NULL
  out
}

elapsed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- tryCatch(
    list(ok = TRUE, value = force(expr), error = NA_character_),
    error = function(e) list(ok = FALSE, value = NULL,
                             error = conditionMessage(e))
  )
  value$elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - t0)
  value
}

parse_args <- function(args) {
  out <- list(
    reps = 250L,
    n = 160L,
    ref_n = 50000L,
    estimators = c("ml", "gls", "fiml", "dwls"),
    cores = 1L,
    chunk_size = NA_integer_,
    results_dir = "results",
    seed_base = 20260625L,
    smoke = FALSE
  )
  parse_cores <- function(x) {
    if (tolower(x) == "auto") {
      return(max(1L, parallel::detectCores(logical = FALSE) - 1L))
    }
    as.integer(x)
  }
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    take <- function() { i <<- i + 1L; args[[i]] }
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N] [--ref-n N]\n",
          "       [--estimators ml,gls,fiml,dwls] [--cores N|auto]\n",
          "       [--chunk-size N] [--results-dir PATH]\n",
          "       [--seed-base S] [--smoke]\n\n",
          "Fits standard, explicit post-hoc RBM, and implicit integrated RBM\n",
          "variants for a small one-factor CFA under ML, GLS, FIML, and DWLS.\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { out$reps <- as.integer(take())
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") { out$n <- as.integer(take())
    } else if (startsWith(a, "--n=")) { out$n <- as.integer(sub("^--n=", "", a))
    } else if (a == "--ref-n") { out$ref_n <- as.integer(take())
    } else if (startsWith(a, "--ref-n=")) { out$ref_n <- as.integer(sub("^--ref-n=", "", a))
    } else if (a == "--estimators") { out$estimators <- parse_csv_arg(take())
    } else if (startsWith(a, "--estimators=")) { out$estimators <- parse_csv_arg(sub("^--estimators=", "", a))
    } else if (a == "--cores") { out$cores <- parse_cores(take())
    } else if (startsWith(a, "--cores=")) { out$cores <- parse_cores(sub("^--cores=", "", a))
    } else if (a == "--chunk-size") { out$chunk_size <- as.integer(take())
    } else if (startsWith(a, "--chunk-size=")) { out$chunk_size <- as.integer(sub("^--chunk-size=", "", a))
    } else if (a == "--results-dir") { out$results_dir <- take()
    } else if (startsWith(a, "--results-dir=")) { out$results_dir <- sub("^--results-dir=", "", a)
    } else if (a == "--seed-base") { out$seed_base <- as.integer(take())
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out$estimators <- tolower(out$estimators)
  bad <- setdiff(out$estimators, c("ml", "gls", "fiml", "dwls"))
  if (length(bad)) stop("unknown estimator token(s): ", paste(bad, collapse = ", "))
  if (out$smoke) {
    out$reps <- 2L
    out$n <- 120L
    out$ref_n <- 2000L
    out$cores <- min(out$cores, 2L)
    if (is.na(out$chunk_size)) out$chunk_size <- 2L
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (!is.finite(out$n) || out$n < 15L) stop("--n must be >= 15")
  if (!is.finite(out$ref_n) || out$ref_n < 1000L) stop("--ref-n must be >= 1000")
  if (!is.finite(out$cores) || out$cores < 1L) stop("--cores must be positive or 'auto'")
  out$cores <- as.integer(out$cores)
  if (is.na(out$chunk_size)) out$chunk_size <- max(10L, out$cores * 10L)
  if (!is.finite(out$chunk_size) || out$chunk_size < 1L) {
    stop("--chunk-size must be positive")
  }
  out$chunk_size <- as.integer(out$chunk_size)
  if (!nzchar(out$results_dir)) stop("--results-dir must be non-empty")
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- if (grepl("^/", cfg$results_dir)) {
  cfg$results_dir
} else {
  experiment_path(cfg$results_dir)
}
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
progress_path <- file.path(res_dir, "progress.csv")
write_csv(data.frame(
  estimator = character(),
  chunk = integer(),
  chunks = integer(),
  completed = integer(),
  total = integer(),
  elapsed_s = numeric(),
  eta_s = numeric(),
  stringsAsFactors = FALSE
), progress_path)

format_duration <- function(seconds) {
  if (!is.finite(seconds)) return("unknown")
  seconds <- max(0, as.integer(round(seconds)))
  h <- seconds %/% 3600L
  m <- (seconds %% 3600L) %/% 60L
  s <- seconds %% 60L
  if (h > 0L) sprintf("%dh%02dm%02ds", h, m, s)
  else sprintf("%dm%02ds", m, s)
}

append_progress <- function(estimator, chunk, chunks, completed, total,
                            elapsed_s, eta_s) {
  row <- data.frame(
    estimator = estimator,
    chunk = chunk,
    chunks = chunks,
    completed = completed,
    total = total,
    elapsed_s = elapsed_s,
    eta_s = eta_s,
    stringsAsFactors = FALSE
  )
  utils::write.table(row, progress_path, sep = ",", row.names = FALSE,
                     col.names = FALSE, append = TRUE)
  cat(sprintf(
    "[%s] chunk %d/%d: %d/%d reps (%.1f%%), elapsed %s, ETA %s\n",
    toupper(estimator), chunk, chunks, completed, total,
    100 * completed / total, format_duration(elapsed_s),
    format_duration(eta_s)
  ))
  flush.console()
}

ov <- paste0("y", 1:4)
lambda <- c(.75, .70, .65, .60)
thresholds <- c(-0.7, 0.2)
missing_rate <- c(y2 = .15, y4 = .20)
model <- "f =~ y1 + y2 + y3 + y4"

control_fit <- list(max_iter = 1200L, ftol = 1e-9, gtol = 1e-6)
control_rbm <- list(max_iter = 900L, ftol = 1e-9, gtol = 1e-6)

draw_latent_response <- function(n, seed) {
  set.seed(seed)
  f <- rnorm(n)
  z <- sapply(seq_along(lambda), function(j) {
    lambda[[j]] * f + sqrt(1 - lambda[[j]]^2) * rnorm(n)
  })
  z <- as.data.frame(z)
  names(z) <- ov
  z
}

as_fiml_data_frame <- function(z, seed) {
  set.seed(seed)
  d <- z
  for (nm in names(missing_rate)) {
    d[[nm]][runif(nrow(d)) < missing_rate[[nm]]] <- NA_real_
  }
  d
}

as_ordinal_data_frame <- function(z) {
  d <- as.data.frame(lapply(z, function(x) {
    ordered(cut(x, c(-Inf, thresholds, Inf), labels = FALSE),
            levels = seq_len(length(thresholds) + 1L))
  }))
  names(d) <- ov
  d
}

data_for_estimator <- function(estimator, n, seed) {
  z <- draw_latent_response(n, seed)
  switch(estimator,
         ml = z,
         gls = z,
         fiml = as_fiml_data_frame(z, seed + 100000L),
         dwls = as_ordinal_data_frame(z))
}

fit_standard <- function(estimator, data) {
  if (estimator == "ml") {
    return(magmaan::magmaan(model, data, estimator = "ML",
                            control = control_fit))
  }
  if (estimator == "gls") {
    return(magmaan::magmaan(model, data, estimator = "GLS",
                            control = control_fit))
  }
  if (estimator == "fiml") {
    return(magmaan::magmaan(model, data, estimator = "FIML",
                            control = control_fit))
  }
  magmaan::magmaan(model, data, estimator = "DWLS", ordered = ov,
                   control = control_fit)
}

fit_rbm <- function(fit, estimator, method) {
  raw_data <- if (estimator %in% c("ml", "gls")) fit$raw_data else NULL
  core$frontier_rbm(fit, raw_data = raw_data, method = method,
                    control = control_rbm)
}

param_class <- function(lhs, op, rhs) {
  if (op == "=~") return("loading")
  if (op == "~~" && lhs == "f" && rhs == "f") return("latent_variance")
  if (op == "~~" && lhs == rhs && lhs %in% ov) return("residual_variance")
  if (op == "|") return("threshold")
  if (op == "~1") return("intercept")
  "other"
}

estimate_table <- function(fit, estimator, method, rep, truth = NULL) {
  pt <- fit$partable
  group <- if ("group" %in% names(pt)) as.integer(pt$group) else rep(1L, nrow(pt))
  free <- as.integer(pt$free)
  ix <- which(free > 0L)
  if (!length(ix)) return(data.frame())
  out <- data.frame(
    estimator = estimator,
    method = method,
    rep = rep,
    group = group[ix],
    lhs = as.character(pt$lhs[ix]),
    op = as.character(pt$op[ix]),
    rhs = as.character(pt$rhs[ix]),
    free = free[ix],
    estimate = as.numeric(pt$est[ix]),
    converged = isTRUE(fit$converged),
    stringsAsFactors = FALSE
  )
  out$param_id <- paste(out$group, out$lhs, out$op, out$rhs, sep = "|")
  out$param_class <- vapply(seq_len(nrow(out)), function(i) {
    param_class(out$lhs[[i]], out$op[[i]], out$rhs[[i]])
  }, character(1))
  if (!is.null(truth)) {
    m <- match(out$param_id, truth$param_id)
    out$truth <- truth$estimate[m]
    out$bias <- out$estimate - out$truth
  }
  out
}

status_row <- function(estimator, method, rep, step, result) {
  data.frame(
    estimator = estimator,
    method = method,
    rep = rep,
    step = step,
    ok = isTRUE(result$ok),
    converged = if (isTRUE(result$ok)) isTRUE(result$value$converged) else FALSE,
    elapsed_ms = result$elapsed_ms,
    error = result$error %||% NA_character_,
    stringsAsFactors = FALSE
  )
}

skipped_status_row <- function(estimator, method, rep, step, error) {
  data.frame(
    estimator = estimator,
    method = method,
    rep = rep,
    step = step,
    ok = FALSE,
    converged = FALSE,
    elapsed_ms = NA_real_,
    error = error,
    stringsAsFactors = FALSE
  )
}

reference_truth <- function(estimator) {
  d <- data_for_estimator(estimator, cfg$ref_n, cfg$seed_base + 900000L +
                            match(estimator, c("ml", "gls", "fiml", "dwls")) * 10000L)
  r <- elapsed(fit_standard(estimator, d))
  if (!isTRUE(r$ok) || !isTRUE(r$value$converged)) {
    msg <- if (isTRUE(r$ok)) "reference fit did not converge" else r$error
    stop("reference fit failed for ", estimator, ": ", msg, call. = FALSE)
  }
  estimate_table(r$value, estimator, "reference", 0L)
}

cat("Computing large-sample reference targets...\n")
truths <- setNames(lapply(cfg$estimators, reference_truth), cfg$estimators)
truth <- do.call(rbind, truths)
write_csv(truth, file.path(res_dir, "reference_targets.csv"))

estimate_rows <- list()
status_rows <- list()
ek <- 0L
sk <- 0L

run_one_rep <- function(estimator, truth_est, rep) {
  seed <- cfg$seed_base +
    match(estimator, c("ml", "gls", "fiml", "dwls")) * 100000L + rep
  d <- data_for_estimator(estimator, cfg$n, seed)

  erows <- list()
  srows <- list()
  epos <- 0L
  spos <- 0L

  base <- elapsed(fit_standard(estimator, d))
  spos <- spos + 1L
  srows[[spos]] <- status_row(estimator, "standard", rep, "fit", base)
  if (!isTRUE(base$ok) || !isTRUE(base$value$converged)) {
    skip_msg <- if (isTRUE(base$ok)) "base fit did not converge" else base$error
    for (method_name in c("posthoc", "integrated")) {
      spos <- spos + 1L
      srows[[spos]] <- skipped_status_row(estimator, method_name, rep,
                                          "skipped", skip_msg)
    }
    return(list(estimates = rbind_fill(erows), statuses = rbind_fill(srows)))
  }

  epos <- epos + 1L
  erows[[epos]] <- estimate_table(base$value, estimator, "standard",
                                  rep, truth_est)

  rbm_methods <- c(posthoc = "explicit", integrated = "implicit")
  for (method_name in names(rbm_methods)) {
    rb <- elapsed(fit_rbm(base$value, estimator, rbm_methods[[method_name]]))
    spos <- spos + 1L
    srows[[spos]] <- status_row(estimator, method_name, rep,
                                rbm_methods[[method_name]], rb)
    if (!isTRUE(rb$ok) || !isTRUE(rb$value$converged)) next
    epos <- epos + 1L
    erows[[epos]] <- estimate_table(rb$value, estimator, method_name,
                                    rep, truth_est)
  }

  list(estimates = rbind_fill(erows), statuses = rbind_fill(srows))
}

run_estimator_reps <- function(estimator, truth_est) {
  reps <- seq_len(cfg$reps)
  chunks <- split(reps, ceiling(seq_along(reps) / cfg$chunk_size))
  out <- vector("list", 0L)
  started <- proc.time()[["elapsed"]]
  completed <- 0L
  for (i in seq_along(chunks)) {
    reps_i <- chunks[[i]]
    chunk <- if (cfg$cores > 1L && .Platform$OS.type != "windows") {
      parallel::mclapply(
        reps_i,
        function(rep) run_one_rep(estimator, truth_est, rep),
        mc.cores = min(cfg$cores, length(reps_i)),
        mc.preschedule = TRUE
      )
    } else {
      lapply(reps_i, function(rep) run_one_rep(estimator, truth_est, rep))
    }
    out <- c(out, chunk)
    completed <- completed + length(reps_i)
    elapsed_s <- proc.time()[["elapsed"]] - started
    eta_s <- if (completed > 0L) elapsed_s * (cfg$reps - completed) / completed else NA_real_
    append_progress(estimator, i, length(chunks), completed, cfg$reps,
                    elapsed_s, eta_s)
  }
  out
}

for (estimator in cfg$estimators) {
  cat("Running ", toupper(estimator), " reps",
      if (cfg$cores > 1L) paste0(" on ", cfg$cores, " cores") else "",
      "\n", sep = "")
  truth_est <- truths[[estimator]]
  rep_results <- run_estimator_reps(estimator, truth_est)
  for (rr in rep_results) {
    if (nrow(rr$estimates)) {
      ek <- ek + 1L
      estimate_rows[[ek]] <- rr$estimates
    }
    if (nrow(rr$statuses)) {
      sk <- sk + 1L
      status_rows[[sk]] <- rr$statuses
    }
  }
}

estimates <- rbind_fill(estimate_rows)
statuses <- rbind_fill(status_rows)
write_csv(estimates, file.path(res_dir, "estimates.csv"))
write_csv(statuses, file.path(res_dir, "fit_status.csv"))

summarise_estimates <- function(estimates, statuses) {
  if (!nrow(estimates)) return(list(param = data.frame(), method = data.frame()))
  keys <- unique(estimates[c("estimator", "method", "param_id", "param_class",
                             "lhs", "op", "rhs")])
  rows <- vector("list", nrow(keys))
  for (i in seq_len(nrow(keys))) {
    k <- keys[i, , drop = FALSE]
    ix <- estimates$estimator == k$estimator &
      estimates$method == k$method &
      estimates$param_id == k$param_id
    d <- estimates[ix & is.finite(estimates$bias), , drop = FALSE]
    rows[[i]] <- data.frame(
      k,
      replications = nrow(d),
      bias = mean(d$bias, na.rm = TRUE),
      abs_mean_bias = abs(mean(d$bias, na.rm = TRUE)),
      abs_bias = mean(abs(d$bias), na.rm = TRUE),
      sampling_variance = stats::var(d$bias, na.rm = TRUE),
      mse = mean(d$bias^2, na.rm = TRUE),
      rmse = sqrt(mean(d$bias^2, na.rm = TRUE)),
      pu = mean(d$bias < 0, na.rm = TRUE),
      median_bias = stats::median(d$bias, na.rm = TRUE),
      q05_bias = stats::quantile(d$bias, .05, na.rm = TRUE),
      q95_bias = stats::quantile(d$bias, .95, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  param <- do.call(rbind, rows)

  method_keys <- unique(estimates[c("estimator", "method")])
  mrows <- vector("list", nrow(method_keys))
  status_keys <- unique(statuses[c("estimator", "method")])
  method_biases <- function(estimator, method, param_ids) {
    e <- estimates[estimates$estimator == estimator &
                     estimates$method == method &
                     estimates$param_id %in% param_ids &
                     is.finite(estimates$bias), , drop = FALSE]
    if (!nrow(e)) return(numeric())
    e$bias
  }
  mise <- function(estimator, method, param_ids) {
    e <- estimates[estimates$estimator == estimator &
                     estimates$method == method &
                     estimates$param_id %in% param_ids &
                     is.finite(estimates$bias), , drop = FALSE]
    if (!nrow(e)) return(NA_real_)
    per_rep <- tapply(e$bias^2, e$rep, mean, na.rm = TRUE)
    mean(per_rep, na.rm = TRUE)
  }
  for (i in seq_len(nrow(method_keys))) {
    k <- method_keys[i, , drop = FALSE]
    d <- param[param$estimator == k$estimator & param$method == k$method, ,
               drop = FALSE]
    primary <- d$param_class %in% c("loading", "latent_variance")
    primary_ids <- d$param_id[primary]
    all_ids <- d$param_id
    primary_biases <- method_biases(k$estimator, k$method, primary_ids)
    mise_primary <- mise(k$estimator, k$method, primary_ids)
    mise_all <- mise(k$estimator, k$method, all_ids)
    st <- statuses[statuses$estimator == k$estimator &
                     statuses$method == k$method, , drop = FALSE]
    mrows[[i]] <- data.frame(
      k,
      primary_parameters = sum(primary),
      mean_bias_primary = mean(d$bias[primary], na.rm = TRUE),
      mean_abs_mean_bias_primary = mean(d$abs_mean_bias[primary],
                                        na.rm = TRUE),
      mean_abs_error_primary = mean(d$abs_bias[primary], na.rm = TRUE),
      mean_abs_bias_primary = mean(d$abs_bias[primary], na.rm = TRUE),
      mean_sampling_variance_primary = mean(d$sampling_variance[primary],
                                            na.rm = TRUE),
      primary_p99_abs_error = stats::quantile(abs(primary_biases), .99,
                                              na.rm = TRUE),
      primary_extreme_rate_abs10 = mean(abs(primary_biases) > 10,
                                        na.rm = TRUE),
      primary_extreme_rate_abs100 = mean(abs(primary_biases) > 100,
                                         na.rm = TRUE),
      mean_mse_primary = mean(d$mse[primary], na.rm = TRUE),
      mean_rmse_primary = mean(d$rmse[primary], na.rm = TRUE),
      mise_primary = mise_primary,
      rmise_primary = sqrt(mise_primary),
      mean_abs_pu_error_primary = mean(abs(d$pu[primary] - 0.5), na.rm = TRUE),
      mean_bias_all = mean(d$bias, na.rm = TRUE),
      mean_abs_mean_bias_all = mean(d$abs_mean_bias, na.rm = TRUE),
      mean_abs_error_all = mean(d$abs_bias, na.rm = TRUE),
      mean_abs_bias_all = mean(d$abs_bias, na.rm = TRUE),
      mean_sampling_variance_all = mean(d$sampling_variance, na.rm = TRUE),
      mean_mse_all = mean(d$mse, na.rm = TRUE),
      mean_rmse_all = mean(d$rmse, na.rm = TRUE),
      mise_all = mise_all,
      rmise_all = sqrt(mise_all),
      accepted = sum(st$ok & st$converged, na.rm = TRUE),
      attempted = nrow(st),
      acceptance_rate = mean(st$ok & st$converged, na.rm = TRUE),
      median_elapsed_ms = stats::median(st$elapsed_ms[st$ok], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  method <- do.call(rbind, mrows)
  method$relative_mise_primary <- NA_real_
  for (estimator in unique(method$estimator)) {
    ix <- method$estimator == estimator
    base <- method$mise_primary[ix & method$method == "standard"]
    if (length(base) == 1L && is.finite(base) && base > 0) {
      method$relative_mise_primary[ix] <- method$mise_primary[ix] / base
    }
  }
  list(param = param, method = method,
       status_keys = status_keys)
}

summ <- summarise_estimates(estimates, statuses)
write_csv(summ$param, file.path(res_dir, "parameter_summary.csv"))
write_csv(summ$method, file.path(res_dir, "method_summary.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    run_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    reps = cfg$reps,
    n = cfg$n,
    ref_n = cfg$ref_n,
    estimators = cfg$estimators,
    cores = cfg$cores,
    chunk_size = cfg$chunk_size,
    results_dir = cfg$results_dir,
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    model = model,
    loadings = lambda,
    item_reliability = lambda^2,
    average_item_reliability = mean(lambda^2),
    thresholds = thresholds,
    missing_rates = paste(names(missing_rate), missing_rate, sep = "=")
  ),
  packages = "magmaan"
)

cat("Wrote:\n")
cat("  ", file.path(res_dir, "reference_targets.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "estimates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "fit_status.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "parameter_summary.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "method_summary.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "progress.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
