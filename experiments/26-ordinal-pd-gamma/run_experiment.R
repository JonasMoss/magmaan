#!/usr/bin/env Rscript
# Ordinal pairwise-deletion Gamma probe.
#
# Two-group, two-factor ordinal CFA with MCAR missingness. Simulate p-values
# under a fake null model and a real loading-noninvariance alternative, then
# report rejection rates for the standard WLSMV/FMG test battery under the
# nominal-N and overlap-corrected pairwise-deletion Gamma matrices.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--missing-rate P]
#                            [--seed-base S] [--truths fake,real] [--smoke]

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

parse_csv_arg <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

rbind_fill <- function(xs) {
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (nm in miss) x[[nm]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, xs)
}

parse_args <- function(args) {
  out <- list(reps = 200L, n = c(1000L), missing_rate = c(.20),
              seed_base = 20260618L, truths = c("fake", "real"),
              smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N[,N]] ",
          "[--missing-rate P[,P]] [--seed-base S] ",
          "[--truths fake,real] [--smoke]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") {
      i <- i + 1L; out$n <- as.integer(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--n=")) {
      out$n <- as.integer(parse_csv_arg(sub("^--n=", "", a)))
    } else if (a == "--missing-rate") {
      i <- i + 1L; out$missing_rate <- as.numeric(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--missing-rate=")) {
      out$missing_rate <- as.numeric(parse_csv_arg(sub("^--missing-rate=", "", a)))
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--truths") {
      i <- i + 1L; out$truths <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--truths=")) {
      out$truths <- parse_csv_arg(sub("^--truths=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 20L
    out$n <- 1000L
    out$missing_rate <- .20
    out$truths <- c("fake", "real")
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n)) || any(out$n < 80L)) stop("--n must be >= 80")
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .95)) {
    stop("--missing-rate must be in [0, .95)")
  }
  bad_truth <- setdiff(out$truths, c("fake", "real"))
  if (length(bad_truth)) stop("unknown truth: ", paste(bad_truth, collapse = ", "))
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaan))

res_dir <- ensure_results_dir()
ov <- paste0("y", 1:8)
model <- paste(
  "f1 =~ y1 + y2 + y3 + y4",
  "f2 =~ y5 + y6 + y7 + y8",
  "f1 ~~ f2",
  sep = "\n")
spec_h1 <- model_spec(model, ordered = ov, group = "group",
                      group_labels = c("A", "B"), parameterization = "delta")
spec_h0 <- model_spec(model, ordered = ov, group = "group",
                      group_labels = c("A", "B"), parameterization = "delta",
                      group_equal = "loadings")
thresholds <- c(-0.8, -0.15, 0.55, 1.15)
lambda_a <- c(.82, .76, .70, .64, .80, .74, .68, .62)
factor_cor <- .35

gof_methods <- function() {
  c(SB = "sb", SS = "ss", SF = "sf",
    EBA2 = "eba2", EBA4 = "eba4", EBA6 = "eba6",
    pEBA2 = "peba2", pEBA4 = "peba4", pEBA6 = "peba6",
    pall = "pall", pOLS = "pols2", all = "all")
}

extract_fmg <- function(tab, methods) {
  key <- sub("_(ml|rls|ls)$", "", tab$label)
  p <- vapply(methods, function(m) {
    hit <- which(key == m)
    if (length(hit)) tab$p_value[hit[1L]] else NA_real_
  }, numeric(1))
  names(p) <- names(methods)
  p
}

draw_group <- function(n, truth, group_label) {
  z1 <- rnorm(n)
  z2 <- factor_cor * z1 + sqrt(1 - factor_cor^2) * rnorm(n)
  lambda <- lambda_a
  if (truth == "real" && group_label == "B") {
    lambda[4L] <- min(.92, lambda[4L] * 1.35)
    lambda[8L] <- max(.35, lambda[8L] * .70)
  }
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in 1:4) {
    z[, j] <- lambda[j] * z1 + rnorm(n, sd = sqrt(1 - lambda[j]^2))
  }
  for (j in 5:8) {
    z[, j] <- lambda[j] * z2 + rnorm(n, sd = sqrt(1 - lambda[j]^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$group <- group_label
  out
}

draw_data <- function(n_per_group, truth, missing_rate) {
  dat <- rbind(draw_group(n_per_group, truth, "A"),
               draw_group(n_per_group, truth, "B"))
  if (missing_rate > 0) {
    for (nm in ov) {
      miss <- runif(nrow(dat)) < missing_rate
      dat[[nm]][miss] <- NA
    }
  }
  dat
}

fit_models <- function(dat, pd_gamma) {
  stats <- magmaan_core$data_ordinal_stats_from_df(
    dat, spec_h1, ordered = ov, group = "group",
    missing = "pairwise", pd_gamma = pd_gamma, full_wls_weight = FALSE)
  fit_h1 <- magmaan(spec_h1, stats, estimator = "DWLS")
  fit_h0 <- magmaan(spec_h0, stats, estimator = "DWLS")
  if (!isTRUE(fit_h1$converged) || !isTRUE(fit_h0$converged)) {
    stop("DWLS fit did not converge", call. = FALSE)
  }
  list(stats = stats, h1 = fit_h1, h0 = fit_h0)
}

gof_battery <- function(fit, stats) {
  methods <- gof_methods()
  tab <- magmaan::fmg_tests_ordinal(fit, stats, tests = names(methods),
                                    weight = "DWLS")
  p_fmg <- extract_fmg(tab, methods)
  base <- tab$base_statistic[1L]
  df <- tab$df[1L]
  spectrum <- tryCatch(as.numeric(tab$eigenvalues[[1L]]), error = function(e) NULL)
  data.frame(
    outcome = "gof",
    method = c("naive", names(p_fmg)),
    p_value = c(stats::pchisq(base, df, lower.tail = FALSE), unname(p_fmg)),
    base_stat = base,
    df = df,
    trace = if (!is.null(spectrum)) sum(spectrum) else NA_real_,
    scaling_factor = if (!is.null(spectrum)) sum(spectrum) / df else NA_real_,
    stringsAsFactors = FALSE)
}

nested_battery <- function(fit_h1, fit_h0, stats) {
  nt <- robust_nested_lrt(
    fit_h1, fit_h0, data = stats, method = "restriction_map",
    A.method = "delta", weight = "DWLS")
  base <- data.frame(
    outcome = "nested",
    method = c("naive", "SB", "adjusted", "mixture"),
    p_value = c(nt$p_unscaled, nt$p_scaled, nt$p_adjusted, nt$p_mixture),
    base_stat = nt$T_diff,
    df = nt$df_diff,
    trace = if (!is.null(nt$eigenvalues)) sum(as.numeric(nt$eigenvalues)) else NA_real_,
    scaling_factor = nt$scale_c %||% NA_real_,
    stringsAsFactors = FALSE)
  methods <- gof_methods()
  tab <- magmaan::fmg_nested_ordinal(
    fit_h1, fit_h0, stats, tests = setdiff(names(methods), "SB"),
    weight = "DWLS", A.method = "delta")
  extra <- data.frame(
    outcome = "nested",
    method = setdiff(names(methods), "SB"),
    p_value = extract_fmg(tab, methods[setdiff(names(methods), "SB")]),
    base_stat = tab$base_statistic[1L],
    df = tab$df[1L],
    trace = if (length(tab$eigenvalues)) sum(as.numeric(tab$eigenvalues[[1L]])) else NA_real_,
    scaling_factor = NA_real_,
    stringsAsFactors = FALSE)
  rbind(base, extra)
}

safe_rep <- function(dat, pd_gamma) {
  tryCatch({
    fits <- fit_models(dat, pd_gamma)
    rbind(gof_battery(fits$h0, fits$stats),
          nested_battery(fits$h1, fits$h0, fits$stats))
  }, error = function(e) {
    data.frame(outcome = c("gof", "nested"), method = "fit_failed",
               p_value = NA_real_, base_stat = NA_real_, df = NA_integer_,
               trace = NA_real_, scaling_factor = NA_real_,
               error = conditionMessage(e), stringsAsFactors = FALSE)
  })
}

rows <- list()
ix <- 0L
for (n in cfg$n) {
  for (mr in cfg$missing_rate) {
    for (truth in cfg$truths) {
      for (rep in seq_len(cfg$reps)) {
        set.seed(cfg$seed_base +
                   as.integer(n * 1000 + round(mr * 1000) * 10 + rep) +
                   if (truth == "real") 500000L else 0L)
        dat <- draw_data(n, truth, mr)
        realized_missing <- mean(is.na(dat[, ov]))
        for (gamma in c("nominal", "overlap")) {
          got <- safe_rep(dat, gamma)
          got$n_per_group <- n
          got$missing_rate <- mr
          got$realized_missing <- realized_missing
          got$truth <- truth
          got$pd_gamma <- gamma
          got$rep <- rep
          got$reject_05 <- is.finite(got$p_value) & got$p_value < .05
          ix <- ix + 1L
          rows[[ix]] <- got
        }
        cat(sprintf("n=%d missing=%.2f truth=%s rep=%d/%d\n",
                    n, mr, truth, rep, cfg$reps))
      }
    }
  }
}

p_values <- rbind_fill(rows)
front <- c("n_per_group", "missing_rate", "realized_missing", "truth", "rep",
           "pd_gamma", "outcome", "method", "p_value", "reject_05",
           "base_stat", "df", "trace", "scaling_factor")
p_values <- p_values[, c(front, setdiff(names(p_values), front))]
write_csv(p_values, file.path(res_dir, "p_values.csv"))
write_csv(p_values, file.path(res_dir, "rejections.csv"))

split_key <- interaction(p_values$n_per_group, p_values$missing_rate,
                         p_values$truth, p_values$pd_gamma,
                         p_values$outcome, p_values$method, drop = TRUE)
summary_rows <- lapply(split(p_values, split_key), function(d) {
  ok <- is.finite(d$p_value)
  rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
  n_ok <- sum(ok)
  data.frame(
    n_per_group = d$n_per_group[1L],
    missing_rate = d$missing_rate[1L],
    realized_missing = mean(d$realized_missing, na.rm = TRUE),
    truth = d$truth[1L],
    pd_gamma = d$pd_gamma[1L],
    outcome = d$outcome[1L],
    method = d$method[1L],
    reps = nrow(d),
    usable = n_ok,
    failures = sum(!ok),
    rejection_rate = rate,
    mc_se = if (n_ok > 0) sqrt(rate * (1 - rate) / n_ok) else NA_real_,
    mean_base = mean(d$base_stat, na.rm = TRUE),
    mean_trace = mean(d$trace, na.rm = TRUE),
    stringsAsFactors = FALSE)
})
rejection_rates <- do.call(rbind, summary_rows)
rejection_rates <- rejection_rates[order(rejection_rates$n_per_group,
                                         rejection_rates$missing_rate,
                                         rejection_rates$truth,
                                         rejection_rates$outcome,
                                         rejection_rates$method,
                                         rejection_rates$pd_gamma), ]
write_csv(rejection_rates, file.path(res_dir, "rejection_rates.csv"))
write_csv(rejection_rates, file.path(res_dir, "summary.csv"))

metadata <- data.frame(
  reps = cfg$reps,
  n = paste(cfg$n, collapse = ","),
  missing_rate = paste(cfg$missing_rate, collapse = ","),
  truths = paste(cfg$truths, collapse = ","),
  seed_base = cfg$seed_base,
  smoke = cfg$smoke,
  model = "two_factor_ordinal_metric_invariance",
  battery = "naive,SB,SS,SF,EBA2/4/6,pEBA2/4/6,pall,pOLS,all; nested adjusted/mixture",
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

cat("wrote:\n")
cat("  ", file.path(res_dir, "p_values.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "rejection_rates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
