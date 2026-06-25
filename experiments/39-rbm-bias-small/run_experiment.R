#!/usr/bin/env Rscript
# Small magmaan-owned bias check for the frontier reduced-bias estimators.
#
# The design is deliberately small: one correctly specified one-factor CFA,
# one sample size, and three estimator families (continuous GLS, raw-data FIML
# with light MCAR missingness, and all-ordinal DWLS). Each replicate fits the
# ordinary estimator, then the explicit post-hoc RBM correction and the implicit
# integrated RBM estimator from the ordinary fit.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--ref-n N]
#                            [--estimators gls,fiml,dwls]
#                            [--seed-base S] [--smoke]

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
    reps = 20L,
    n = 160L,
    ref_n = 20000L,
    estimators = c("gls", "fiml", "dwls"),
    seed_base = 20260625L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    take <- function() { i <<- i + 1L; args[[i]] }
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N] [--ref-n N]\n",
          "       [--estimators gls,fiml,dwls] [--seed-base S] [--smoke]\n\n",
          "Fits standard, explicit post-hoc RBM, and implicit integrated RBM\n",
          "variants for a small one-factor CFA under GLS, FIML, and DWLS.\n",
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
    } else if (a == "--seed-base") { out$seed_base <- as.integer(take())
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out$estimators <- tolower(out$estimators)
  bad <- setdiff(out$estimators, c("gls", "fiml", "dwls"))
  if (length(bad)) stop("unknown estimator token(s): ", paste(bad, collapse = ", "))
  if (out$smoke) {
    out$reps <- 2L
    out$n <- 120L
    out$ref_n <- 2000L
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (!is.finite(out$n) || out$n < 80L) stop("--n must be >= 80")
  if (!is.finite(out$ref_n) || out$ref_n < 1000L) stop("--ref-n must be >= 1000")
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- ensure_results_dir()

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
         gls = z,
         fiml = as_fiml_data_frame(z, seed + 100000L),
         dwls = as_ordinal_data_frame(z))
}

fit_standard <- function(estimator, data) {
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
  raw_data <- if (estimator == "gls") fit$raw_data else NULL
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
                            match(estimator, c("gls", "fiml", "dwls")) * 10000L)
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

for (estimator in cfg$estimators) {
  cat("Running ", toupper(estimator), " reps\n", sep = "")
  truth_est <- truths[[estimator]]
  for (rep in seq_len(cfg$reps)) {
    seed <- cfg$seed_base +
      match(estimator, c("gls", "fiml", "dwls")) * 100000L + rep
    d <- data_for_estimator(estimator, cfg$n, seed)

    base <- elapsed(fit_standard(estimator, d))
    sk <- sk + 1L
    status_rows[[sk]] <- status_row(estimator, "standard", rep, "fit", base)
    if (!isTRUE(base$ok) || !isTRUE(base$value$converged)) {
      skip_msg <- if (isTRUE(base$ok)) "base fit did not converge" else base$error
      for (method_name in c("posthoc", "integrated")) {
        sk <- sk + 1L
        status_rows[[sk]] <- skipped_status_row(estimator, method_name, rep,
                                                "skipped", skip_msg)
      }
      next
    }

    ek <- ek + 1L
    estimate_rows[[ek]] <- estimate_table(base$value, estimator, "standard",
                                          rep, truth_est)

    rbm_methods <- c(posthoc = "explicit", integrated = "implicit")
    for (method_name in names(rbm_methods)) {
      rb <- elapsed(fit_rbm(base$value, estimator, rbm_methods[[method_name]]))
      sk <- sk + 1L
      status_rows[[sk]] <- status_row(estimator, method_name, rep,
                                      rbm_methods[[method_name]], rb)
      if (!isTRUE(rb$ok) || !isTRUE(rb$value$converged)) next
      ek <- ek + 1L
      estimate_rows[[ek]] <- estimate_table(rb$value, estimator, method_name,
                                            rep, truth_est)
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
      abs_bias = mean(abs(d$bias), na.rm = TRUE),
      rmse = sqrt(mean(d$bias^2, na.rm = TRUE)),
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
  for (i in seq_len(nrow(method_keys))) {
    k <- method_keys[i, , drop = FALSE]
    d <- param[param$estimator == k$estimator & param$method == k$method, ,
               drop = FALSE]
    primary <- d$param_class %in% c("loading", "latent_variance")
    st <- statuses[statuses$estimator == k$estimator &
                     statuses$method == k$method, , drop = FALSE]
    mrows[[i]] <- data.frame(
      k,
      primary_parameters = sum(primary),
      mean_abs_bias_primary = mean(d$abs_bias[primary], na.rm = TRUE),
      mean_rmse_primary = mean(d$rmse[primary], na.rm = TRUE),
      mean_abs_bias_all = mean(d$abs_bias, na.rm = TRUE),
      mean_rmse_all = mean(d$rmse, na.rm = TRUE),
      accepted = sum(st$ok & st$converged, na.rm = TRUE),
      attempted = nrow(st),
      acceptance_rate = mean(st$ok & st$converged, na.rm = TRUE),
      median_elapsed_ms = stats::median(st$elapsed_ms[st$ok], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  list(param = param, method = do.call(rbind, mrows),
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
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    model = model,
    loadings = lambda,
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
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
