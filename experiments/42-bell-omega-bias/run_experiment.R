#!/usr/bin/env Rscript
# Bell et al. (2024)-style omega bias without finite-sample noise:
# evaluate ordinary one-factor omega functionals directly at population
# covariance matrices and subtract Bell's population reliability target.

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

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [--results-dir DIR] [--smoke] [--help]\n",
    "\n",
    "Population-moment Bell-style bias for ordinary one-factor omega. The\n",
    "estimator functionals are evaluated at Bell-inspired population covariance\n",
    "matrices; no finite-N simulation, SE, or coverage is computed.\n",
    "\n",
    "Options:\n",
    "  --results-dir DIR  Output directory. Default: results.\n",
    "  --smoke            Accepted for experiment harness consistency; same run.\n",
    "  --help             Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  out <- list(results_dir = "results", smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--results-dir") {
      i <- i + 1L
      if (i > length(args)) stop("--results-dir needs a value", call. = FALSE)
      out$results_dir <- args[[i]]
    } else if (startsWith(a, "--results-dir=")) {
      out$results_dir <- sub("^--results-dir=", "", a)
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

res_dir <- if (startsWith(cfg$results_dir, "/")) {
  cfg$results_dir
} else {
  experiment_path(cfg$results_dir)
}
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

omega_model <- function(ov) {
  p <- length(ov)
  loads <- paste0("l", seq_len(p), "*", ov)
  resid <- paste0(ov, " ~~ e", seq_len(p), "*", ov)
  paste(c(paste0("f =~ ", paste(loads, collapse = " + ")), resid),
        collapse = "\n")
}

sample_stats_from_cov <- function(Sigma, n = 1000000L) {
  list(S = list(Sigma), mean = list(rep(0, nrow(Sigma))), nobs = as.integer(n))
}

target_reliability <- function(common, Sigma) {
  sum(common) / sum(Sigma)
}

population_record <- function(name, label, reliability_level, Sigma, common,
                              source_table = "Bell et al. Table 1") {
  p <- nrow(Sigma)
  ov <- paste0("y", seq_len(p))
  dimnames(Sigma) <- list(ov, ov)
  dimnames(common) <- list(ov, ov)
  list(
    name = name,
    label = label,
    reliability_level = reliability_level,
    p = p,
    Sigma = Sigma,
    common = common,
    true_reliability = target_reliability(common, Sigma),
    source_table = source_table
  )
}

one_factor_population <- function(reliability_level) {
  lambda <- switch(
    reliability_level,
    low = c(.414, .210, .472, .416, .325, .504, .301, .521),
    high = c(.847, .423, .870, .516, .648, .721, .322, .743),
    stop("unknown reliability level", call. = FALSE)
  )
  common <- tcrossprod(lambda)
  Sigma <- common
  diag(Sigma) <- 1.0
  population_record("one_factor", "One factor", reliability_level, Sigma, common)
}

correlated_error_population <- function(reliability_level) {
  lambda <- switch(
    reliability_level,
    low = c(.504, .293, .412, .506, .451, .574, .434, .610),
    high = c(.823, .593, .837, .706, .751, .834, .746, .744),
    stop("unknown reliability level", call. = FALSE)
  )
  residual_covs <- switch(
    reliability_level,
    low = c(.287, .343, .334, .232, .197, .312),
    high = c(.187, .263, .264, .232, .197, .282),
    stop("unknown reliability level", call. = FALSE)
  )
  common <- tcrossprod(lambda)
  Theta <- diag(1.0 - lambda^2)
  k <- 1L
  for (j in 1:3) {
    for (i in (j + 1):4) {
      Theta[i, j] <- residual_covs[k]
      Theta[j, i] <- residual_covs[k]
      k <- k + 1L
    }
  }
  Sigma <- common + Theta
  population_record("correlated_errors", "One factor + residual covariance",
                    reliability_level, Sigma, common)
}

bifactor_population <- function(reliability_level) {
  vals <- switch(
    reliability_level,
    low = c(.462, .340, .420, .659, .314, .501, .608, .410,
            .426, .414, .279, .338, 0, 0, 0, 0,
            0, 0, 0, 0, .314, .448, .213, .417),
    high = c(.852, .736, .868, .612, .913, .704, .719, .611,
             .426, .414, .279, .338, 0, 0, 0, 0,
             0, 0, 0, 0, .314, .448, .213, .417),
    stop("unknown reliability level", call. = FALSE)
  )
  Lambda <- matrix(vals, nrow = 8, ncol = 3)
  common <- tcrossprod(Lambda[, 1])
  Sigma <- tcrossprod(Lambda)
  diag(Sigma) <- 1.0
  population_record("bifactor", "Bifactor", reliability_level, Sigma, common)
}

higher_order_population <- function(reliability_level) {
  vals <- switch(
    reliability_level,
    low = c(.60, .72, .64, .62, .68, .73, .83, .54, .64, .44, .75, .66,
            .59, .62, .69, .61),
    high = c(.84, .87, .89, .82, .84, .86, .83, .91, .89, .84, .85, .84,
             .79, .72, .91, .81),
    stop("unknown reliability level", call. = FALSE)
  )
  item_loadings <- vals[1:12]
  higher_loadings <- vals[13:16]
  Lambda <- matrix(0, 12, 4)
  for (k in seq_len(4)) {
    rows <- ((k - 1L) * 3L + 1L):(k * 3L)
    Lambda[rows, k] <- item_loadings[rows]
  }
  Phi_lower <- tcrossprod(higher_loadings)
  diag(Phi_lower) <- 1.0
  indirect <- as.vector(Lambda %*% higher_loadings)
  common <- tcrossprod(indirect)
  Sigma <- Lambda %*% Phi_lower %*% t(Lambda)
  diag(Sigma) <- 1.0
  population_record("higher_order", "Higher-order", reliability_level,
                    Sigma, common)
}

make_populations <- function() {
  constructors <- list(one_factor_population, correlated_error_population,
                       bifactor_population, higher_order_population)
  out <- list()
  for (ctor in constructors) {
    for (rel in c("low", "high")) out[[length(out) + 1L]] <- ctor(rel)
  }
  out
}

fit_omega_rows <- function(pop) {
  ov <- colnames(pop$Sigma)
  model <- omega_model(ov)
  sample_stats <- sample_stats_from_cov(pop$Sigma)
  out <- list()
  idx <- 1L
  for (estimator in c("ML", "ULS", "GLS")) {
    fit <- tryCatch(
      magmaan::magmaan(model, sample_stats, estimator = estimator,
                       std_lv = TRUE),
      error = function(e) e
    )
    if (inherits(fit, "error") || !isTRUE(fit$converged)) {
      for (denom in c("model_implied", "observed_total")) {
        out[[idx]] <- data.frame(
          estimator = estimator,
          omega_variant = paste0("omega_u_", denom),
          denominator = denom,
          estimate = NA_real_,
          converged = FALSE,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
      next
    }
    loadings <- fit$partable$est[fit$partable$op == "=~"]
    numerator <- sum(loadings)^2
    implied <- magmaan_core$model_implied(fit)$sigma[[1L]]
    denominators <- c(model_implied = sum(implied),
                      observed_total = sum(pop$Sigma))
    for (denom in names(denominators)) {
      out[[idx]] <- data.frame(
        estimator = estimator,
        omega_variant = paste0("omega_u_", denom),
        denominator = denom,
        estimate = numerator / denominators[[denom]],
        converged = TRUE,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }
  sg <- magmaan_core$measures_reliability_cov(pop$Sigma)$table
  sg <- sg[sg$coefficient == "spearman_guttman_omega", ]
  out[[idx]] <- data.frame(
    estimator = "closed_form",
    omega_variant = "spearman_guttman_covariance_omega",
    denominator = "observed_total",
    estimate = as.numeric(sg$value),
    converged = TRUE,
    stringsAsFactors = FALSE
  )
  do.call(rbind, out)
}

populations <- make_populations()
target_rows <- do.call(rbind, lapply(populations, function(pop) {
  data.frame(
    population = pop$name,
    population_label = pop$label,
    reliability_level = pop$reliability_level,
    p = pop$p,
    true_reliability = pop$true_reliability,
    total_variance = sum(pop$Sigma),
    common_total_variance = sum(pop$common),
    source_table = pop$source_table,
    stringsAsFactors = FALSE
  )
}))
write_csv(target_rows, file.path(res_dir, "population_targets.csv"))

rows <- do.call(rbind, lapply(populations, function(pop) {
  x <- fit_omega_rows(pop)
  x$population <- pop$name
  x$population_label <- pop$label
  x$reliability_level <- pop$reliability_level
  x$p <- pop$p
  x$true_reliability <- pop$true_reliability
  x$bias <- x$estimate - x$true_reliability
  x$abs_bias <- abs(x$bias)
  x$relative_bias <- x$bias / x$true_reliability
  x
}))
rows <- rows[, c("population", "population_label", "reliability_level", "p",
                 "estimator", "omega_variant", "denominator", "estimate",
                 "true_reliability", "bias", "abs_bias", "relative_bias",
                 "converged")]
rows <- rows[order(rows$population, rows$reliability_level, rows$estimator,
                   rows$omega_variant), ]
write_csv(rows, file.path(res_dir, "omega_bias.csv"))

summary <- do.call(rbind, lapply(split(rows, rows$population), function(d) {
  ok <- d$converged & is.finite(d$abs_bias)
  d <- d[ok, , drop = FALSE]
  data.frame(
    population = d$population[1L],
    population_label = d$population_label[1L],
    max_abs_bias = max(d$abs_bias),
    best_method = paste(d$estimator[which.min(d$abs_bias)],
                        d$denominator[which.min(d$abs_bias)], sep = "/"),
    best_abs_bias = min(d$abs_bias),
    worst_method = paste(d$estimator[which.max(d$abs_bias)],
                         d$denominator[which.max(d$abs_bias)], sep = "/"),
    stringsAsFactors = FALSE
  )
}))
write_csv(summary, file.path(res_dir, "population_summary.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    source = "Bell et al. 2024 Table 1, p=8 cells plus higher-order p=12",
    target = "population reliability = total-score variance due to the generating factor common to all items divided by total-score variance",
    bias = "estimate minus population reliability, evaluated at population moments",
    fitted_models = "ordinary one-factor model with independent residuals",
    estimators = "ML, ULS, GLS, plus closed-form Spearman-Guttman covariance omega",
    denominators = "model-implied total variance and observed/population total variance",
    smoke = cfg$smoke,
    results_dir = res_dir
  ),
  packages = "magmaan"
)

cat("wrote results to: ", res_dir, "\n", sep = "")
