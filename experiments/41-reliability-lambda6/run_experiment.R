#!/usr/bin/env Rscript
# Finite-sample check for covariance-only reliability coefficients and fitted
# normal-theory omega. The covariance functionals use the sample covariance and
# empirical-Gamma delta SEs; the fitted omega uses a one-factor ML model with
# either model SEs or observed-bread empirical sandwich SEs.

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
    "Usage: Rscript run_experiment.R [options]\n",
    "\n",
    "Compares alpha, Guttman's lambda6, Spearman-Guttman covariance omega, and\n",
    "normal-theory fitted omega under one correct and one misspecified normal\n",
    "population.\n",
    "\n",
    "Options:\n",
    "  --reps N           Replications per population x N cell. Default: 250.\n",
    "  --n LIST           Comma-separated sample sizes. Default: 250,500.\n",
    "  --seed-base N      Base RNG seed. Default: 20260626.\n",
    "  --results-dir DIR  Output directory. Default: results.\n",
    "  --smoke            Fast run: reps=30, n=250.\n",
    "  --help             Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  out <- list(reps = 250L, n = c(250L, 500L), seed_base = 20260626L,
              results_dir = "results", smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L
      if (i > length(args)) stop("--reps needs a value", call. = FALSE)
      out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") {
      i <- i + 1L
      if (i > length(args)) stop("--n needs a value", call. = FALSE)
      out$n <- as.integer(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--n=")) {
      out$n <- as.integer(parse_csv_arg(sub("^--n=", "", a)))
    } else if (a == "--seed-base") {
      i <- i + 1L
      if (i > length(args)) stop("--seed-base needs a value", call. = FALSE)
      out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
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
  if (out$smoke) {
    out$reps <- 30L
    out$n <- 250L
  }
  if (!is.finite(out$reps) || out$reps < 2L) {
    stop("--reps must be an integer >= 2", call. = FALSE)
  }
  if (!length(out$n) || any(!is.finite(out$n)) || any(out$n <= 8L)) {
    stop("--n must contain sample sizes above the item count", call. = FALSE)
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

ov <- paste0("y", 1:6)
p <- length(ov)

omega_model_syntax <- function(ov) {
  p <- length(ov)
  loads <- paste0("l", seq_len(p), "*", ov)
  resid <- paste0(ov, " ~~ e", seq_len(p), "*", ov)
  lsum <- paste0("l", seq_len(p), collapse = " + ")
  esum <- paste0("e", seq_len(p), collapse = " + ")
  paste(c(
    paste0("f =~ ", paste(loads, collapse = " + ")),
    resid,
    paste0("omega := (", lsum, ")^2 / ((", lsum, ")^2 + ", esum, ")")
  ), collapse = "\n")
}
omega_model <- omega_model_syntax(ov)

unit_total_reliability <- function(common, Sigma) {
  sum(common) / sum(Sigma)
}

population_one_factor <- function() {
  lambda <- c(0.85, 0.75, 0.70, 0.65, 0.60, 0.55)
  common <- tcrossprod(lambda)
  psi <- 1.0 - lambda^2
  Sigma <- common + diag(psi, p)
  dimnames(Sigma) <- list(ov, ov)
  list(name = "one_factor", label = "Correct one-factor",
       Sigma = Sigma, common = common,
       true_reliability = unit_total_reliability(common, Sigma))
}

population_two_factor <- function() {
  Lambda <- matrix(0, p, 2)
  Lambda[1:3, 1] <- c(0.85, 0.75, 0.65)
  Lambda[4:6, 2] <- c(0.80, 0.70, 0.60)
  Phi <- matrix(c(1.0, 0.35, 0.35, 1.0), 2, 2)
  common <- Lambda %*% Phi %*% t(Lambda)
  psi <- 1.0 - diag(common)
  Sigma <- common + diag(psi, p)
  dimnames(Sigma) <- list(ov, ov)
  list(name = "two_factor_fit_as_one", label = "Two-factor fit as one",
       Sigma = Sigma, common = common,
       true_reliability = unit_total_reliability(common, Sigma))
}

sample_stats_from_cov <- function(Sigma, n = 1000000L) {
  list(S = list(Sigma), mean = list(rep(0, nrow(Sigma))), nobs = as.integer(n))
}

defined_omega <- function(fit, vcov) {
  d <- magmaan::compute_defined(omega_model, fit, vcov)
  as.numeric(d$est[d$lhs == "omega" & d$op == ":="][1L])
}

defined_omega_se <- function(fit, vcov) {
  d <- magmaan::compute_defined(omega_model, fit, vcov)
  as.numeric(d$se[d$lhs == "omega" & d$op == ":="][1L])
}

population_omega_target <- function(Sigma) {
  fit <- magmaan::magmaan(omega_model, sample_stats_from_cov(Sigma),
                          estimator = "ML", std_lv = TRUE)
  if (!isTRUE(fit$converged)) {
    stop("population one-factor omega target did not converge", call. = FALSE)
  }
  V <- magmaan_core$inference_vcov(
    magmaan_core$inference_information_expected(fit), fit)
  defined_omega(fit, V)
}

covariance_targets <- function(pop) {
  tab <- magmaan_core$measures_reliability_cov(pop$Sigma)$table
  out <- data.frame(
    population = pop$name,
    population_label = pop$label,
    method = as.character(tab$coefficient),
    target_functional = as.numeric(tab$value),
    target_kind = "covariance functional",
    true_reliability = pop$true_reliability,
    stringsAsFactors = FALSE
  )
  omega_target <- population_omega_target(pop$Sigma)
  rbind(out, data.frame(
    population = pop$name,
    population_label = pop$label,
    method = c("nt_omega_model_se", "nt_omega_robust_se"),
    target_functional = omega_target,
    target_kind = "one-factor ML pseudo-target",
    true_reliability = pop$true_reliability,
    stringsAsFactors = FALSE
  ))
}

draw_normal <- function(Sigma, n) {
  z <- matrix(rnorm(n * nrow(Sigma)), n, nrow(Sigma))
  x <- z %*% chol(Sigma)
  colnames(x) <- colnames(Sigma)
  x
}

reliability_rows <- function(X) {
  n <- nrow(X)
  S <- magmaan_core$data_sample_stats_from_raw(X)$S[[1L]]
  gamma <- magmaan_core$robust_empirical_gamma(X)
  tab <- magmaan_core$measures_reliability_cov(S, gamma, n)$table
  data.frame(
    method = as.character(tab$coefficient),
    estimate = as.numeric(tab$value),
    se = as.numeric(tab$se),
    converged = TRUE,
    interval = "empirical covariance delta",
    stringsAsFactors = FALSE
  )
}

omega_rows <- function(X) {
  d <- as.data.frame(X)
  fit <- tryCatch(
    magmaan::magmaan(omega_model, d, estimator = "ML", std_lv = TRUE),
    error = function(e) e
  )
  empty <- function(method, interval) data.frame(
    method = method, estimate = NA_real_, se = NA_real_, converged = FALSE,
    interval = interval, stringsAsFactors = FALSE)
  if (inherits(fit, "error") || !isTRUE(fit$converged)) {
    return(rbind(
      empty("nt_omega_model_se", "one-factor model delta"),
      empty("nt_omega_robust_se", "observed-bread empirical sandwich")))
  }
  V_model <- tryCatch(magmaan_core$inference_vcov(
    magmaan_core$inference_information_expected(fit), fit),
    error = function(e) NULL)
  V_robust <- tryCatch(magmaan_core$robust_se_raw_fit(
    fit, X, bread = "observed", cov = "empirical")$vcov,
    error = function(e) NULL)
  if (is.null(V_model)) {
    model_row <- empty("nt_omega_model_se", "one-factor model delta")
  } else {
    model_row <- data.frame(
      method = "nt_omega_model_se",
      estimate = defined_omega(fit, V_model),
      se = defined_omega_se(fit, V_model),
      converged = TRUE,
      interval = "one-factor model delta",
      stringsAsFactors = FALSE
    )
  }
  if (is.null(V_robust)) {
    robust_row <- empty("nt_omega_robust_se", "observed-bread empirical sandwich")
  } else {
    robust_row <- data.frame(
      method = "nt_omega_robust_se",
      estimate = defined_omega(fit, V_robust),
      se = defined_omega_se(fit, V_robust),
      converged = TRUE,
      interval = "observed-bread empirical sandwich",
      stringsAsFactors = FALSE
    )
  }
  rbind(model_row, robust_row)
}

with_targets <- function(rows, targets) {
  key <- match(rows$method, targets$method)
  rows$target_functional <- targets$target_functional[key]
  rows$target_kind <- targets$target_kind[key]
  rows$true_reliability <- targets$true_reliability[key]
  ok <- is.finite(rows$estimate) & is.finite(rows$se)
  rows$ci_lower <- rows$estimate - 1.96 * rows$se
  rows$ci_upper <- rows$estimate + 1.96 * rows$se
  rows$cover_functional <- ifelse(ok, rows$ci_lower <= rows$target_functional &
                                    rows$ci_upper >= rows$target_functional, NA)
  rows$cover_true <- ifelse(ok, rows$ci_lower <= rows$true_reliability &
                              rows$ci_upper >= rows$true_reliability, NA)
  rows$squared_error_functional <- (rows$estimate - rows$target_functional)^2
  rows$squared_error_true <- (rows$estimate - rows$true_reliability)^2
  rows
}

summarize_group <- function(d) {
  ok <- is.finite(d$estimate) & d$converged
  x <- d[ok, , drop = FALSE]
  data.frame(
    population = d$population[1L],
    population_label = d$population_label[1L],
    n = d$n[1L],
    method = d$method[1L],
    interval = d$interval[1L],
    reps = nrow(d),
    n_ok = nrow(x),
    target_functional = d$target_functional[1L],
    target_kind = d$target_kind[1L],
    true_reliability = d$true_reliability[1L],
    mean_estimate = mean(x$estimate, na.rm = TRUE),
    sd_estimate = stats::sd(x$estimate, na.rm = TRUE),
    mean_se = mean(x$se, na.rm = TRUE),
    bias_functional = mean(x$estimate - x$target_functional, na.rm = TRUE),
    rmse_functional = sqrt(mean(x$squared_error_functional, na.rm = TRUE)),
    bias_true = mean(x$estimate - x$true_reliability, na.rm = TRUE),
    rmse_true = sqrt(mean(x$squared_error_true, na.rm = TRUE)),
    coverage_functional = mean(x$cover_functional, na.rm = TRUE),
    coverage_true = mean(x$cover_true, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

populations <- list(population_one_factor(), population_two_factor())
targets <- do.call(rbind, lapply(populations, covariance_targets))
write_csv(targets, file.path(res_dir, "population_targets.csv"))

cat(sprintf("simulation: %d populations x %d sample sizes x %d reps\n",
            length(populations), length(cfg$n), cfg$reps))
raw_rows <- list()
idx <- 1L
for (pi in seq_along(populations)) {
  pop <- populations[[pi]]
  pop_targets <- targets[targets$population == pop$name, , drop = FALSE]
  for (n in cfg$n) {
    cat(sprintf("  %-22s N=%-5d", pop$name, n))
    for (r in seq_len(cfg$reps)) {
      set.seed(cfg$seed_base + pi * 1000000L + n * 1000L + r)
      X <- draw_normal(pop$Sigma, n)
      rows <- rbind(reliability_rows(X), omega_rows(X))
      rows <- with_targets(rows, pop_targets)
      rows$population <- pop$name
      rows$population_label <- pop$label
      rows$n <- n
      rows$rep <- r
      raw_rows[[idx]] <- rows
      idx <- idx + 1L
      if (r %% max(1L, floor(cfg$reps / 5L)) == 0L) cat(".")
    }
    cat(" done\n")
  }
}

raw <- do.call(rbind, raw_rows)
raw <- raw[, c("population", "population_label", "n", "rep", "method",
               "interval", "estimate", "se", "converged", "target_functional",
               "target_kind", "true_reliability", "ci_lower", "ci_upper",
               "cover_functional", "cover_true", "squared_error_functional",
               "squared_error_true")]
write_csv(raw, file.path(res_dir, "simulation_raw.csv"))

parts <- split(raw, paste(raw$population, raw$n, raw$method, sep = "\r"))
summary <- do.call(rbind, lapply(parts, summarize_group))
summary <- summary[order(summary$population, summary$n, summary$method), ]
write_csv(summary, file.path(res_dir, "simulation_summary.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps,
    n = cfg$n,
    p = p,
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    populations = paste(vapply(populations, `[[`, character(1), "name"),
                        collapse = ","),
    covariance_intervals = "empirical Gamma delta method on vech(S)",
    fitted_omega = "one-factor normal-theory ML; model SE and observed-bread empirical sandwich SE",
    results_dir = res_dir
  ),
  packages = "magmaan"
)

cat("\nwrote results to: ", res_dir, "\n", sep = "")
