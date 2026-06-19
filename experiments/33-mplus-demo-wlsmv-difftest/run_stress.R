#!/usr/bin/env Rscript
# Minimal Chen-style WLSMV_PD stress probe.
#
# The single-seed run_experiment.R checks formulas and Mplus/magmaan parity.
# This script repeats the invariant configural -> scalar comparison over a small
# seed grid to see whether the WLSMV_PD difference statistic inflates under
# pairwise missing ordinal data.
#
# Usage:
#   Rscript run_stress.R [--reps 25] [--mplus-reps 10] [--n-total 1000]
#                        [--missing-rates 0,0.5]
#                        [--threshold-set symmetric|asymmetric]
#                        [--mplus-command mpdemo]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]),
                            mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_stress.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
  if (!length(xs)) return(data.frame())
  cols <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    miss <- setdiff(cols, names(x))
    for (nm in miss) x[[nm]] <- NA
    x[, cols, drop = FALSE]
  })
  do.call(rbind, xs)
}

parse_args <- function(args) {
  out <- list(seed_start = 2026061900L, reps = 25L, mplus_reps = 10L,
              n_total = 1000L, missing_rates = c(0, .50),
              threshold_set = "symmetric", mplus_command = "mpdemo",
              lavaan = TRUE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_stress.R [--reps N] [--mplus-reps N] ",
          "[--seed-start S] [--n-total N] [--missing-rates 0,0.5] ",
          "[--threshold-set symmetric|asymmetric] ",
          "[--mplus-command mpdemo] [--no-lavaan]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--mplus-reps") {
      i <- i + 1L; out$mplus_reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--mplus-reps=")) {
      out$mplus_reps <- as.integer(sub("^--mplus-reps=", "", a))
    } else if (a == "--seed-start") {
      i <- i + 1L; out$seed_start <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-start=")) {
      out$seed_start <- as.integer(sub("^--seed-start=", "", a))
    } else if (a == "--n-total") {
      i <- i + 1L; out$n_total <- as.integer(args[[i]])
    } else if (startsWith(a, "--n-total=")) {
      out$n_total <- as.integer(sub("^--n-total=", "", a))
    } else if (a == "--missing-rates") {
      i <- i + 1L; out$missing_rates <- parse_csv_numeric(args[[i]])
    } else if (startsWith(a, "--missing-rates=")) {
      out$missing_rates <- parse_csv_numeric(sub("^--missing-rates=", "", a))
    } else if (a == "--threshold-set") {
      i <- i + 1L; out$threshold_set <- args[[i]]
    } else if (startsWith(a, "--threshold-set=")) {
      out$threshold_set <- sub("^--threshold-set=", "", a)
    } else if (a == "--mplus-command") {
      i <- i + 1L; out$mplus_command <- args[[i]]
    } else if (startsWith(a, "--mplus-command=")) {
      out$mplus_command <- sub("^--mplus-command=", "", a)
    } else if (a == "--no-lavaan") {
      out$lavaan <- FALSE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  if (!is.finite(out$mplus_reps) || out$mplus_reps < 0L) {
    stop("--mplus-reps must be a nonnegative integer", call. = FALSE)
  }
  if (!is.finite(out$n_total) || out$n_total < 120L ||
      out$n_total %% 2L != 0L) {
    stop("--n-total must be an even integer >= 120", call. = FALSE)
  }
  if (!length(out$missing_rates) ||
      any(!is.finite(out$missing_rates)) ||
      any(out$missing_rates < 0 | out$missing_rates >= .95)) {
    stop("--missing-rates must be finite values in [0, .95)", call. = FALSE)
  }
  if (!out$threshold_set %in% c("symmetric", "asymmetric")) {
    stop("--threshold-set must be symmetric or asymmetric", call. = FALSE)
  }
  out$mplus_reps <- min(out$mplus_reps, out$reps)
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
if (isTRUE(cfg$lavaan)) require_pkg("lavaan")
suppressPackageStartupMessages({
  library(magmaan)
  if (isTRUE(cfg$lavaan)) library(lavaan)
})

res_dir <- ensure_results_dir()
stress_dir <- file.path(
  res_dir,
  sprintf("stress_%s_n%d_reps%d_mplus%d",
          cfg$threshold_set, cfg$n_total, cfg$reps, cfg$mplus_reps))
dir.create(stress_dir, recursive = TRUE, showWarnings = FALSE)

ov <- paste0("y", 1:6)
thresholds <- switch(
  cfg$threshold_set,
  symmetric = c(-1.3, -0.47, 0.47, 1.3),
  asymmetric = c(-0.253, 0.385, 0.842, 1.282))
loading <- .60
model <- paste0("f =~ ", paste(ov, collapse = " + "))
scalar_mplus_extra <- paste(
  c("f ~ c(0, NA)*1", sprintf("%s ~ c(0, 0)*1", ov)),
  collapse = "\n")
model_scalar_mplus <- paste(model, scalar_mplus_extra, sep = "\n")
n_group <- as.integer(cfg$n_total / 2L)

draw_group <- function(n, group_label) {
  eta <- stats::rnorm(n)
  aux <- 0.5 * eta + sqrt(1 - 0.5^2) * stats::rnorm(n)
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    z[, j] <- loading * eta + stats::rnorm(n, sd = sqrt(1 - loading^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$aux <- aux
  out$group <- group_label
  out
}

apply_rank_missing <- function(dat, missing_rate) {
  b_rows <- which(dat$group == "B")
  nb <- length(b_rows)
  target <- as.integer(round(missing_rate * nb))
  if (target <= 0L) {
    dat$group <- factor(dat$group, levels = c("A", "B"))
    return(dat)
  }
  ranks <- rank(dat$aux[b_rows], ties.method = "random")
  weights <- pmax(nb - ranks, 0)
  if (!any(weights > 0)) weights <- rep(1, nb)
  for (nm in ov[4:6]) {
    picked <- sample.int(nb, size = target, replace = FALSE, prob = weights)
    dat[[nm]][b_rows[picked]] <- NA
  }
  dat$group <- factor(dat$group, levels = c("A", "B"))
  dat
}

make_data <- function(seed, missing_rate) {
  set.seed(seed)
  dat <- rbind(draw_group(n_group, "A"), draw_group(n_group, "B"))
  apply_rank_missing(dat, missing_rate)
}

to_mplus_data <- function(dat) {
  out <- data.frame(
    g = ifelse(dat$group == "A", 1L, 2L),
    aux = dat$aux,
    stringsAsFactors = FALSE
  )
  for (nm in ov) out[[nm]] <- as.integer(dat[[nm]])
  out
}

mplus_number <- function(line) {
  matches <- regmatches(line, gregexpr(
    "[-+]?[0-9]*\\.?[0-9]+([DdEe][-+]?[0-9]+)?", line,
    perl = TRUE))[[1L]]
  if (!length(matches)) return(NA_real_)
  as.numeric(gsub("[Dd]", "E", tail(matches, 1L)))
}

section_number <- function(lines, heading, field, occurrence = 1L) {
  idx <- which(trimws(lines) == heading)
  if (length(idx) < occurrence) return(NA_real_)
  win <- lines[idx[[occurrence]]:min(length(lines), idx[[occurrence]] + 25L)]
  hit <- grep(paste0("^\\s*", field, "\\s+"), win, value = TRUE)
  if (!length(hit)) return(NA_real_)
  mplus_number(hit[[1L]])
}

parse_mplus_fit <- function(path, n_total) {
  lines <- readLines(path, warn = FALSE)
  fmin <- section_number(lines,
                         "Optimum Function Value for Weighted Least-Squares Estimator",
                         "Value")
  data.frame(
    raw = if (is.finite(fmin)) 2 * n_total * fmin else NA_real_,
    scaled = section_number(lines, "Chi-Square Test of Model Fit", "Value"),
    converged = any(grepl("THE MODEL ESTIMATION TERMINATED NORMALLY", lines,
                          fixed = TRUE)),
    stringsAsFactors = FALSE
  )
}

parse_mplus_difftest <- function(path) {
  lines <- readLines(path, warn = FALSE)
  data.frame(
    stat = section_number(lines, "Chi-Square Test for Difference Testing",
                          "Value"),
    df = section_number(lines, "Chi-Square Test for Difference Testing",
                        "Degrees of Freedom"),
    p_value = section_number(lines, "Chi-Square Test for Difference Testing",
                             "P-Value"),
    stringsAsFactors = FALSE
  )
}

run_mplus_command <- function(input, output, work_dir, command) {
  old <- setwd(work_dir)
  on.exit(setwd(old), add = TRUE)
  out <- tryCatch(
    suppressWarnings(system2(command, c(input, output),
                             stdout = TRUE, stderr = TRUE)),
    error = function(e) structure(conditionMessage(e), status = 127L)
  )
  status <- attr(out, "status") %||% 0L
  list(status = as.integer(status), stdout = paste(out, collapse = "\n"))
}

write_mplus_inputs <- function(dat, case_dir) {
  dir.create(case_dir, recursive = TRUE, showWarnings = FALSE)
  write.table(to_mplus_data(dat), file.path(case_dir, "stress.dat"),
              row.names = FALSE, col.names = FALSE, quote = FALSE,
              na = "-999")
  base <- c(
    "DATA: FILE IS stress.dat;",
    "VARIABLE:",
    " NAMES ARE g aux y1-y6;",
    " USEVARIABLES ARE y1-y6;",
    " CATEGORICAL ARE y1-y6;",
    " GROUPING IS g (1 = A 2 = B);",
    " MISSING ARE ALL (-999);",
    "ANALYSIS:",
    " ESTIMATOR = WLSMV;",
    " PARAMETERIZATION = DELTA;")
  h1 <- c(
    "TITLE: stress H1 configural;",
    base,
    "MODEL:",
    " f BY y1-y6;",
    "MODEL B:",
    " {y1-y6@1};",
    " [f@0];",
    " f BY y1@1 y2-y6;",
    " [y1$1-y6$4];",
    "SAVEDATA:",
    " DIFFTEST IS h1diff.dat;")
  h0 <- c(
    "TITLE: stress H0 scalar;",
    base,
    " DIFFTEST IS h1diff.dat;",
    "MODEL:",
    " f BY y1-y6;")
  writeLines(h1, file.path(case_dir, "h1.inp"))
  writeLines(h0, file.path(case_dir, "h0_scalar.inp"))
}

run_mplus_case <- function(dat, rep_id, seed, missing_rate) {
  mr_label <- sprintf("m%03d", as.integer(round(100 * missing_rate)))
  case_dir <- file.path(stress_dir, sprintf("rep%03d_seed%d_%s",
                                            rep_id, seed, mr_label))
  write_mplus_inputs(dat, case_dir)
  h1 <- run_mplus_command("h1.inp", "h1.out", case_dir, cfg$mplus_command)
  h0 <- run_mplus_command("h0_scalar.inp", "h0_scalar.out", case_dir,
                          cfg$mplus_command)
  err <- ""
  if (h1$status != 0L || h0$status != 0L) {
    err <- paste("Mplus status", h1$status, h0$status)
  }
  diff <- if (file.exists(file.path(case_dir, "h0_scalar.out"))) {
    parse_mplus_difftest(file.path(case_dir, "h0_scalar.out"))
  } else {
    data.frame(stat = NA_real_, df = NA_real_, p_value = NA_real_)
  }
  h1_fit <- if (file.exists(file.path(case_dir, "h1.out"))) {
    parse_mplus_fit(file.path(case_dir, "h1.out"), cfg$n_total)
  } else {
    data.frame(raw = NA_real_, scaled = NA_real_,
               converged = FALSE)
  }
  h0_fit <- if (file.exists(file.path(case_dir, "h0_scalar.out"))) {
    parse_mplus_fit(file.path(case_dir, "h0_scalar.out"), cfg$n_total)
  } else {
    data.frame(raw = NA_real_, scaled = NA_real_,
               converged = FALSE)
  }
  data.frame(
    rep = rep_id,
    seed = seed,
    missing_rate = missing_rate,
    engine = "mplus",
    variant = "wlsmv_pd",
    stat = diff$stat,
    df = diff$df,
    p_value = diff$p_value,
    reject_05 = is.finite(diff$p_value) && diff$p_value < .05,
    h1_raw = h1_fit$raw,
    h0_raw = h0_fit$raw,
    h1_scaled = h1_fit$scaled,
    h0_scaled = h0_fit$scaled,
    converged = h1_fit$converged && h0_fit$converged,
    error = err,
    stringsAsFactors = FALSE)
}

spec_h1 <- magmaan::model_spec(
  model, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta")
spec_scalar_mplus <- magmaan::model_spec(
  model_scalar_mplus, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta", group_equal = c("loadings", "thresholds"))
core <- magmaan::magmaan_core

run_magmaan_case <- function(dat, rep_id, seed, missing_rate) {
  dat2 <- dat
  dat2[ov] <- lapply(dat2[ov], ordered)
  out <- tryCatch({
    stats <- core$data_ordinal_stats_from_df(
      dat2, spec_h1, ordered = ov, group = "group", missing = "pairwise",
      pd_gamma = "overlap", full_wls_weight = FALSE)
    fit_h1 <- magmaan::magmaan(spec_h1, stats, estimator = "DWLS")
    fit_h0 <- magmaan::magmaan(spec_scalar_mplus, stats, estimator = "DWLS")
    nt <- magmaan::robust_nested_lrt(
      fit_h1, fit_h0, data = stats, method = "restriction_map",
      A.method = "delta", weight = "DWLS")
    data.frame(
      stat = nt$scaled_shifted$chi2_adj,
      df = nt$scaled_shifted$df,
      p_value = nt$scaled_shifted$pvalue,
      h1_raw = 2 * fit_h1$ntotal * fit_h1$fmin,
      h0_raw = 2 * fit_h0$ntotal * fit_h0$fmin,
      h1_scaled = NA_real_,
      h0_scaled = NA_real_,
      converged = isTRUE(fit_h1$converged) && isTRUE(fit_h0$converged),
      error = "",
      stringsAsFactors = FALSE)
  }, error = function(e) {
    data.frame(stat = NA_real_, df = NA_real_, p_value = NA_real_,
               h1_raw = NA_real_, h0_raw = NA_real_,
               h1_scaled = NA_real_, h0_scaled = NA_real_,
               converged = FALSE, error = conditionMessage(e),
               stringsAsFactors = FALSE)
  })
  data.frame(
    rep = rep_id,
    seed = seed,
    missing_rate = missing_rate,
    engine = "magmaan",
    variant = "pd_gamma_overlap_scaled_shifted",
    out,
    reject_05 = is.finite(out$p_value) && out$p_value < .05,
    stringsAsFactors = FALSE)
}

run_lavaan_case <- function(dat, rep_id, seed, missing_rate) {
  if (!isTRUE(cfg$lavaan)) return(NULL)
  dat2 <- dat
  dat2[ov] <- lapply(dat2[ov], ordered)
  out <- tryCatch({
    fit_h1 <- lavaan::cfa(
      model, data = dat2, ordered = ov, group = "group",
      estimator = "WLSMV", parameterization = "delta", missing = "pairwise",
      se = "none")
    fit_h0 <- lavaan::cfa(
      model_scalar_mplus, data = dat2, ordered = ov, group = "group",
      group.equal = c("loadings", "thresholds"), estimator = "WLSMV",
      parameterization = "delta", missing = "pairwise", se = "none")
    tab <- suppressMessages(lavaan::lavTestLRT(
      fit_h1, fit_h0, method = "satorra.2000", A.method = "delta",
      scaled.shifted = TRUE))
    fm1 <- lavaan::fitMeasures(fit_h1, c("chisq", "chisq.scaled"))
    fm0 <- lavaan::fitMeasures(fit_h0, c("chisq", "chisq.scaled"))
    data.frame(
      stat = as.numeric(tab[2L, "Chisq diff"]),
      df = as.numeric(tab[2L, "Df diff"]),
      p_value = as.numeric(tab[2L, "Pr(>Chisq)"]),
      h1_raw = unname(fm1[["chisq"]]),
      h0_raw = unname(fm0[["chisq"]]),
      h1_scaled = unname(fm1[["chisq.scaled"]]),
      h0_scaled = unname(fm0[["chisq.scaled"]]),
      converged = lavaan::lavInspect(fit_h1, "converged") &&
        lavaan::lavInspect(fit_h0, "converged"),
      error = "",
      stringsAsFactors = FALSE)
  }, error = function(e) {
    data.frame(stat = NA_real_, df = NA_real_, p_value = NA_real_,
               h1_raw = NA_real_, h0_raw = NA_real_,
               h1_scaled = NA_real_, h0_scaled = NA_real_,
               converged = FALSE, error = conditionMessage(e),
               stringsAsFactors = FALSE)
  })
  data.frame(
    rep = rep_id,
    seed = seed,
    missing_rate = missing_rate,
    engine = "lavaan",
    variant = "wlsmv_pairwise_delta_scaled_shifted",
    out,
    reject_05 = is.finite(out$p_value) && out$p_value < .05,
    stringsAsFactors = FALSE)
}

summarise_results <- function(results) {
  ok <- results[is.finite(results$stat) & is.finite(results$p_value), ]
  if (!nrow(ok)) return(data.frame())
  groups <- split(ok, paste(ok$engine, ok$variant, ok$missing_rate, sep = "\r"))
  rows <- lapply(groups, function(x) {
    data.frame(
      engine = x$engine[[1L]],
      variant = x$variant[[1L]],
      missing_rate = x$missing_rate[[1L]],
      reps_ok = nrow(x),
      mean_stat = mean(x$stat),
      median_stat = stats::median(x$stat),
      sd_stat = if (nrow(x) > 1L) stats::sd(x$stat) else NA_real_,
      mean_p = mean(x$p_value),
      reject_rate_05 = mean(x$p_value < .05),
      mean_h0_raw = mean(x$h0_raw, na.rm = TRUE),
      mean_h1_raw = mean(x$h1_raw, na.rm = TRUE),
      stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

compare_mplus_magmaan <- function(results) {
  mplus <- results[results$engine == "mplus" & is.finite(results$stat),
                   c("rep", "seed", "missing_rate", "stat", "p_value")]
  mag <- results[results$engine == "magmaan" & is.finite(results$stat),
                 c("rep", "seed", "missing_rate", "stat", "p_value")]
  if (!nrow(mplus) || !nrow(mag)) return(data.frame())
  names(mplus)[4:5] <- c("mplus_stat", "mplus_p")
  names(mag)[4:5] <- c("magmaan_stat", "magmaan_p")
  merged <- merge(mplus, mag, by = c("rep", "seed", "missing_rate"))
  merged$stat_diff <- merged$magmaan_stat - merged$mplus_stat
  merged$p_diff <- merged$magmaan_p - merged$mplus_p
  merged
}

rows <- list()
counter <- 0L
for (rep_id in seq_len(cfg$reps)) {
  seed <- cfg$seed_start + rep_id - 1L
  for (missing_rate in cfg$missing_rates) {
    counter <- counter + 1L
    cat(sprintf("[%03d] seed=%d missing=%.2f\n", counter, seed,
                missing_rate))
    dat <- make_data(seed, missing_rate)
    rows <- c(rows, list(run_magmaan_case(dat, rep_id, seed, missing_rate)))
    rows <- c(rows, list(run_lavaan_case(dat, rep_id, seed, missing_rate)))
    if (rep_id <= cfg$mplus_reps) {
      rows <- c(rows, list(run_mplus_case(dat, rep_id, seed, missing_rate)))
    }
  }
}

results <- rbind_fill(rows)
results <- results[, c("rep", "seed", "missing_rate", "engine", "variant",
                       "stat", "df", "p_value", "reject_05", "h1_raw",
                       "h0_raw", "h1_scaled", "h0_scaled", "converged",
                       "error")]
summary <- summarise_results(results)
parity <- compare_mplus_magmaan(results)
parity_summary <- if (nrow(parity)) {
  data.frame(
    missing_rate = sort(unique(parity$missing_rate)),
    max_abs_stat_diff = vapply(sort(unique(parity$missing_rate)), function(mr) {
      max(abs(parity$stat_diff[parity$missing_rate == mr]), na.rm = TRUE)
    }, numeric(1)),
    mean_abs_stat_diff = vapply(sort(unique(parity$missing_rate)), function(mr) {
      mean(abs(parity$stat_diff[parity$missing_rate == mr]), na.rm = TRUE)
    }, numeric(1)),
    stringsAsFactors = FALSE)
} else {
  data.frame()
}

write_csv(results, file.path(res_dir, "stress_reps.csv"))
write_csv(summary, file.path(res_dir, "stress_summary.csv"))
write_csv(parity, file.path(res_dir, "stress_mplus_magmaan_parity.csv"))
write_csv(parity_summary, file.path(res_dir, "stress_parity_summary.csv"))
write_metadata(
  file.path(res_dir, "stress_metadata.csv"),
  values = list(
    seed_start = cfg$seed_start,
    reps = cfg$reps,
    mplus_reps = cfg$mplus_reps,
    n_total = cfg$n_total,
    n_per_group = n_group,
    missing_rates = paste(cfg$missing_rates, collapse = ","),
    missing_items_group_b = paste(ov[4:6], collapse = ","),
    threshold_set = cfg$threshold_set,
    thresholds = paste(thresholds, collapse = ","),
    loading = loading,
    mplus_command = cfg$mplus_command,
    stress_work_dir = stress_dir),
  packages = c("lavaan", "magmaan"))

print(summary, row.names = FALSE)
if (nrow(parity_summary)) {
  cat("\nMplus/magmaan parity over Mplus-run reps:\n")
  print(parity_summary, row.names = FALSE)
}
cat("\nWrote stress results to ", res_dir, "\n", sep = "")
