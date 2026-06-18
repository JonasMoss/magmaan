#!/usr/bin/env Rscript
# Pairwise-composite ordinal nested LR probe.
#
# Reuses the experiment 26 two-group ordinal CFA data-generating setup, but
# analyzes only the nested loading-invariance test using the frontier
# pairwise/composite ordinal estimator and its Godambe LR correction.
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

parse_csv_arg <- function(x) trimws(strsplit(x, ",", fixed = TRUE)[[1L]])

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
  out <- list(reps = 50L, n = c(300L), missing_rate = c(.20),
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
    out$reps <- 6L
    out$n <- 180L
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
ov <- paste0("y", 1:4)
threshold_rows <- paste(sprintf("%s | t1 + t2 + t3 + t4", ov), collapse = "\n")
unit_rows <- paste(sprintf("%s ~*~ 1*%s", ov, ov), collapse = "\n")
model_h1 <- paste(
  "f =~ y1 + y2 + y3 + y4",
  threshold_rows,
  unit_rows,
  sep = "\n")
model_h0 <- paste(
  "f =~ y1 + a*y2 + a*y3 + y4",
  threshold_rows,
  unit_rows,
  sep = "\n")
spec_h1 <- model_spec(model_h1, ordered = ov, parameterization = "delta")
spec_h0 <- model_spec(model_h0, ordered = ov, parameterization = "delta")
thresholds <- c(-0.8, -0.15, 0.55, 1.15)
lambda_a <- c(.82, .70, .70, .64)

draw_group <- function(n, truth) {
  z1 <- rnorm(n)
  lambda <- lambda_a
  if (truth == "real") {
    lambda[3L] <- .45
  }
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    z[, j] <- lambda[j] * z1 + rnorm(n, sd = sqrt(1 - lambda[j]^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out
}

draw_data <- function(n, truth, missing_rate) {
  dat <- draw_group(n, truth)
  if (missing_rate > 0) {
    for (nm in ov) {
      miss <- runif(nrow(dat)) < missing_rate
      dat[[nm]][miss] <- NA
    }
  }
  dat
}

raw_blocks <- function(dat) {
  m <- sapply(dat[, ov, drop = FALSE], as.integer)
  storage.mode(m) <- "double"
  dimnames(m) <- list(NULL, ov)
  list(m)
}

safe_rep <- function(dat) {
  tryCatch({
    fit <- magmaan_core$frontier_pairwise_ordinal_composite_nested(
      spec_h1$partable, spec_h0$partable, raw_blocks(dat),
      n_levels = list(rep(length(thresholds) + 1L, length(ov))),
      control = list(max_iter = 120L),
      fd_step = 2e-5)
    lr <- fit$lr
    data.frame(
      method = c("naive", "scaled", "adjusted", "mixture"),
      p_value = c(lr$p_unscaled, lr$p_scaled, lr$p_adjusted, lr$p_mixture),
      base_stat = lr$T_diff,
      df = lr$df_diff,
      trace = sum(as.numeric(lr$eigenvalues)),
      scaling_factor = lr$scale_c,
      h1_fmin = fit$h1$fmin,
      h0_fmin = fit$h0$fmin,
      h1_condition_bread = fit$h1$godambe$condition_bread,
      h0_condition_bread = fit$h0$godambe$condition_bread,
      stringsAsFactors = FALSE)
  }, error = function(e) {
    data.frame(method = "fit_failed", p_value = NA_real_,
               base_stat = NA_real_, df = NA_integer_, trace = NA_real_,
               scaling_factor = NA_real_, h1_fmin = NA_real_,
               h0_fmin = NA_real_, h1_condition_bread = NA_real_,
               h0_condition_bread = NA_real_,
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
        got <- safe_rep(dat)
        got$n_per_group <- n
        got$missing_rate <- mr
        got$realized_missing <- mean(is.na(dat[, ov]))
        got$truth <- truth
        got$rep <- rep
        got$reject_05 <- is.finite(got$p_value) & got$p_value < .05
        ix <- ix + 1L
        rows[[ix]] <- got
        cat(sprintf("n=%d missing=%.2f truth=%s rep=%d/%d\n",
                    n, mr, truth, rep, cfg$reps))
      }
    }
  }
}

p_values <- rbind_fill(rows)
front <- c("n_per_group", "missing_rate", "realized_missing", "truth", "rep",
           "method", "p_value", "reject_05", "base_stat", "df", "trace",
           "scaling_factor", "h1_condition_bread", "h0_condition_bread")
p_values <- p_values[, c(front, setdiff(names(p_values), front))]
write_csv(p_values, file.path(res_dir, "p_values.csv"))

split_key <- interaction(p_values$n_per_group, p_values$missing_rate,
                         p_values$truth, p_values$method, drop = TRUE)
summary_rows <- lapply(split(p_values, split_key), function(d) {
  ok <- is.finite(d$p_value)
  rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
  n_ok <- sum(ok)
  data.frame(
    n_per_group = d$n_per_group[1L],
    missing_rate = d$missing_rate[1L],
    realized_missing = mean(d$realized_missing, na.rm = TRUE),
    truth = d$truth[1L],
    method = d$method[1L],
    reps = nrow(d),
    usable = n_ok,
    failures = sum(!ok),
    rejection_rate = rate,
    mc_se = if (n_ok > 0) sqrt(rate * (1 - rate) / n_ok) else NA_real_,
    mean_base = mean(d$base_stat, na.rm = TRUE),
    mean_trace = mean(d$trace, na.rm = TRUE),
    mean_h1_condition_bread = mean(d$h1_condition_bread, na.rm = TRUE),
    mean_h0_condition_bread = mean(d$h0_condition_bread, na.rm = TRUE),
    stringsAsFactors = FALSE)
})
rejection_rates <- do.call(rbind, summary_rows)
rejection_rates <- rejection_rates[order(rejection_rates$n_per_group,
                                         rejection_rates$missing_rate,
                                         rejection_rates$truth,
                                         rejection_rates$method), ]
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
  estimator = "frontier pairwise ordinal composite",
  battery = "nested LR: unscaled, scaled, adjusted, mixture",
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

cat("wrote:\n")
cat("  ", file.path(res_dir, "p_values.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "rejection_rates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
