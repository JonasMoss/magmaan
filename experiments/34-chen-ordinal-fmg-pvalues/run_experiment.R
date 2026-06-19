#!/usr/bin/env Rscript
# Chen-style ordinal pairwise-missing nested p-value battery.
#
# No Mplus dependency: this uses magmaan's pairwise ordinal stats and the full
# FMG eigenvalue-tail family on the configural -> scalar difference spectrum.
#
# Usage:
#   Rscript run_experiment.R [--reps 100] [--n-total 1000]
#                            [--missing-rate 0,0.5]
#                            [--thresholds symmetric|asymmetric]
#                            [--pd-gamma overlap[,nominal]]
#                            [--missing-mechanism mar|mcar]
#                            [--seed-base S] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]),
                            mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
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
  out <- list(reps = 100L, n_total = 1000L, missing_rate = c(0, .50),
              thresholds = "symmetric", pd_gamma = "overlap",
              missing_mechanism = "mar",
              seed_base = 20260620L, smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n-total N[,N]] ",
          "[--missing-rate P[,P]] [--thresholds symmetric,asymmetric] ",
          "[--pd-gamma overlap[,nominal]] ",
          "[--missing-mechanism mar[,mcar]] [--seed-base S] [--smoke]\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n-total") {
      i <- i + 1L; out$n_total <- as.integer(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--n-total=")) {
      out$n_total <- as.integer(parse_csv_arg(sub("^--n-total=", "", a)))
    } else if (a == "--missing-rate") {
      i <- i + 1L; out$missing_rate <- as.numeric(parse_csv_arg(args[[i]]))
    } else if (startsWith(a, "--missing-rate=")) {
      out$missing_rate <- as.numeric(parse_csv_arg(sub("^--missing-rate=", "", a)))
    } else if (a == "--thresholds") {
      i <- i + 1L; out$thresholds <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--thresholds=")) {
      out$thresholds <- parse_csv_arg(sub("^--thresholds=", "", a))
    } else if (a == "--pd-gamma") {
      i <- i + 1L; out$pd_gamma <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--pd-gamma=")) {
      out$pd_gamma <- parse_csv_arg(sub("^--pd-gamma=", "", a))
    } else if (a == "--missing-mechanism") {
      i <- i + 1L; out$missing_mechanism <- parse_csv_arg(args[[i]])
    } else if (startsWith(a, "--missing-mechanism=")) {
      out$missing_mechanism <- parse_csv_arg(sub("^--missing-mechanism=", "", a))
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- 3L
    out$n_total <- 300L
    out$missing_rate <- c(0, .50)
    out$thresholds <- "symmetric"
    out$pd_gamma <- "overlap"
    out$missing_mechanism <- c("mar", "mcar")
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  if (any(!is.finite(out$n_total)) || any(out$n_total < 120L) ||
      any(out$n_total %% 2L != 0L)) {
    stop("--n-total must contain even integers >= 120", call. = FALSE)
  }
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .95)) {
    stop("--missing-rate must be in [0, .95)", call. = FALSE)
  }
  bad_thresholds <- setdiff(out$thresholds, c("symmetric", "asymmetric"))
  if (length(bad_thresholds)) {
    stop("unknown thresholds: ", paste(bad_thresholds, collapse = ", "),
         call. = FALSE)
  }
  bad_gamma <- setdiff(out$pd_gamma, c("overlap", "nominal"))
  if (length(bad_gamma)) {
    stop("unknown pd-gamma: ", paste(bad_gamma, collapse = ", "),
         call. = FALSE)
  }
  bad_mech <- setdiff(out$missing_mechanism, c("mar", "mcar"))
  if (length(bad_mech)) {
    stop("unknown missing-mechanism: ", paste(bad_mech, collapse = ", "),
         call. = FALSE)
  }
  out
}

threshold_values <- function(shape) {
  switch(shape,
         symmetric = c(-1.3, -0.47, 0.47, 1.3),
         asymmetric = c(-0.253, 0.385, 0.842, 1.282),
         stop("unknown threshold shape", call. = FALSE))
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

res_dir <- ensure_results_dir()
ov <- paste0("v", 1:10)
loading <- .60
model <- paste0("f =~ ", paste(ov, collapse = " + "))
scalar_extra <- paste(
  c("f ~ c(0, NA)*1", sprintf("%s ~ c(0, 0)*1", ov)),
  collapse = "\n")
model_scalar <- paste(model, scalar_extra, sep = "\n")

spec_h1 <- magmaan::model_spec(
  model, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta")
spec_h0 <- magmaan::model_spec(
  model_scalar, ordered = ov, group = "group", group_labels = c("A", "B"),
  parameterization = "delta", group_equal = c("loadings", "thresholds"))

draw_group <- function(n, group_label, thresholds) {
  eta <- rnorm(n)
  aux <- 0.5 * eta + sqrt(1 - 0.5^2) * rnorm(n)
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    z[, j] <- loading * eta + rnorm(n, sd = sqrt(1 - loading^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$aux <- aux
  out$group <- group_label
  out
}

apply_group_b_missing <- function(dat, missing_rate, mechanism) {
  if (missing_rate <= 0) {
    dat$group <- factor(dat$group, levels = c("A", "B"))
    return(dat)
  }
  b_rows <- which(dat$group == "B")
  nb <- length(b_rows)
  target <- as.integer(round(missing_rate * nb))
  if (mechanism == "mar") {
    # Missingness on items 8-10 in group B is driven by the rank of the
    # auxiliary (which correlates r = .5 with the factor): MAR.
    ranks <- rank(dat$aux[b_rows], ties.method = "random")
    weights <- pmax(nb - ranks, 0)
    if (!any(weights > 0)) weights <- rep(1, nb)
  } else {
    # MCAR: uniform within group B, independent of the auxiliary and items.
    weights <- rep(1, nb)
  }
  for (nm in ov[8:10]) {
    picked <- sample.int(nb, size = target, replace = FALSE, prob = weights)
    dat[[nm]][b_rows[picked]] <- NA
  }
  dat$group <- factor(dat$group, levels = c("A", "B"))
  dat
}

draw_data <- function(n_total, threshold_shape, missing_rate, mechanism) {
  n_group <- as.integer(n_total / 2L)
  th <- threshold_values(threshold_shape)
  dat <- rbind(draw_group(n_group, "A", th),
               draw_group(n_group, "B", th))
  apply_group_b_missing(dat, missing_rate, mechanism)
}

realized_missing_b_items <- function(dat) {
  b <- dat$group == "B"
  mean(is.na(dat[b, ov[8:10]]))
}

fmg_methods <- function() {
  data.frame(
    method = c("SS", "SF", "EBA2", "EBA4", "EBA6",
               "pEBA2", "pEBA4", "pEBA6", "pall", "pOLS", "all"),
    cpp_method = c("scaled_shifted", "scaled_f", "eba", "eba", "eba",
                   "peba", "peba", "peba", "penalized_all", "pols", "all"),
    param = c(NA, NA, 2, 4, 6, 2, 4, 6, NA, 2, NA),
    stringsAsFactors = FALSE)
}

fmg_pvalue <- function(chisq, df, eigvals, method, param = NA_real_) {
  out <- magmaan::magmaan_core$robust_fmg_test(
    chisq, as.integer(df), as.numeric(eigvals), method = method,
    param = if (is.finite(param)) param else 4)
  out$p_value
}

standard_rows <- function(nt) {
  data.frame(
    method = c("naive", "SB", "adjusted", "scaled_shifted", "mixture"),
    p_value = c(nt$p_unscaled, nt$p_scaled, nt$p_adjusted,
                nt$scaled_shifted$pvalue, nt$p_mixture),
    statistic = c(nt$T_diff, nt$T_scaled, nt$T_adjusted,
                  nt$scaled_shifted$chi2_adj, nt$T_diff),
    df = c(nt$df_diff, nt$df_diff, nt$adjust_d0,
           nt$scaled_shifted$df, nt$df_diff),
    family = "low_moment",
    stringsAsFactors = FALSE)
}

fmg_rows <- function(nt) {
  specs <- fmg_methods()
  p <- mapply(fmg_pvalue, MoreArgs = list(chisq = nt$T_diff,
                                          df = nt$df_diff,
                                          eigvals = nt$eigenvalues),
              method = specs$cpp_method, param = specs$param)
  data.frame(
    method = specs$method,
    p_value = as.numeric(p),
    statistic = nt$T_diff,
    df = nt$df_diff,
    family = ifelse(specs$method %in% c("SS", "SF"),
                    "low_moment", "full_spectrum"),
    stringsAsFactors = FALSE)
}

fit_rep <- function(dat, pd_gamma) {
  tryCatch({
    stats <- magmaan::magmaan_core$data_ordinal_stats_from_df(
      dat, spec_h1, ordered = ov, group = "group", missing = "pairwise",
      pd_gamma = pd_gamma, full_wls_weight = FALSE)
    fit_h1 <- magmaan::magmaan(spec_h1, stats, estimator = "DWLS")
    fit_h0 <- magmaan::magmaan(spec_h0, stats, estimator = "DWLS")
    if (!isTRUE(fit_h1$converged) || !isTRUE(fit_h0$converged)) {
      stop("DWLS fit did not converge", call. = FALSE)
    }
    nt <- magmaan::robust_nested_lrt(
      fit_h1, fit_h0, data = stats, method = "restriction_map",
      A.method = "delta", weight = "DWLS")
    rows <- rbind(standard_rows(nt), fmg_rows(nt))
    rows$T_diff <- nt$T_diff
    rows$df_diff <- nt$df_diff
    rows$spectrum_trace <- sum(as.numeric(nt$eigenvalues))
    rows$spectrum_min <- min(as.numeric(nt$eigenvalues))
    rows$spectrum_max <- max(as.numeric(nt$eigenvalues))
    rows$scale_c <- nt$scale_c
    rows$converged <- TRUE
    rows$error <- ""
    rows
  }, error = function(e) {
    data.frame(
      method = "fit_failed",
      p_value = NA_real_,
      statistic = NA_real_,
      df = NA_real_,
      family = "failure",
      T_diff = NA_real_,
      df_diff = NA_real_,
      spectrum_trace = NA_real_,
      spectrum_min = NA_real_,
      spectrum_max = NA_real_,
      scale_c = NA_real_,
      converged = FALSE,
      error = conditionMessage(e),
      stringsAsFactors = FALSE)
  })
}

rows <- list()
ix <- 0L
for (n_total in cfg$n_total) {
  for (threshold_shape in cfg$thresholds) {
    for (missing_rate in cfg$missing_rate) {
      for (mechanism in cfg$missing_mechanism) {
        # Complete-data cells are identical across mechanisms; run the shared
        # baseline once, under the primary mechanism only.
        if (missing_rate <= 0 && mechanism != cfg$missing_mechanism[1L]) next
        for (rep in seq_len(cfg$reps)) {
          seed <- cfg$seed_base +
            as.integer(n_total * 1000 + round(missing_rate * 1000) * 10 + rep) +
            (if (threshold_shape == "asymmetric") 500000L else 0L) +
            (if (mechanism == "mcar") 2000000L else 0L)
          set.seed(seed)
          dat <- draw_data(n_total, threshold_shape, missing_rate, mechanism)
          realized_missing <- realized_missing_b_items(dat)
          for (pd_gamma in cfg$pd_gamma) {
            got <- fit_rep(dat, pd_gamma)
            got$n_total <- n_total
            got$n_per_group <- as.integer(n_total / 2L)
            got$thresholds <- threshold_shape
            got$missing_rate <- missing_rate
            got$missing_mechanism <- mechanism
            got$realized_missing <- realized_missing
            got$pd_gamma <- pd_gamma
            got$rep <- rep
            got$seed <- seed
            got$reject_05 <- is.finite(got$p_value) & got$p_value < .05
            ix <- ix + 1L
            rows[[ix]] <- got
          }
          cat(sprintf("n=%d thresholds=%s missing=%.2f mechanism=%s rep=%d/%d\n",
                      n_total, threshold_shape, missing_rate, mechanism,
                      rep, cfg$reps))
        }
      }
    }
  }
}

p_values <- rbind_fill(rows)
front <- c("n_total", "n_per_group", "thresholds", "missing_rate",
           "missing_mechanism", "realized_missing", "pd_gamma", "rep", "seed",
           "method", "family", "p_value", "reject_05", "statistic", "df",
           "T_diff", "df_diff", "spectrum_trace", "spectrum_min",
           "spectrum_max", "scale_c", "converged", "error")
p_values <- p_values[, c(front, setdiff(names(p_values), front))]
write_csv(p_values, file.path(res_dir, "p_values.csv"))

split_key <- interaction(p_values$n_total, p_values$thresholds,
                         p_values$missing_rate, p_values$missing_mechanism,
                         p_values$pd_gamma, p_values$method, drop = TRUE)
summary_rows <- lapply(split(p_values, split_key), function(d) {
  ok <- is.finite(d$p_value)
  rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
  data.frame(
    n_total = d$n_total[1L],
    n_per_group = d$n_per_group[1L],
    thresholds = d$thresholds[1L],
    missing_rate = d$missing_rate[1L],
    missing_mechanism = d$missing_mechanism[1L],
    realized_missing = mean(d$realized_missing, na.rm = TRUE),
    pd_gamma = d$pd_gamma[1L],
    method = d$method[1L],
    family = d$family[1L],
    reps = nrow(d),
    usable = sum(ok),
    failures = sum(!ok),
    rejection_rate = rate,
    mc_se = if (sum(ok) > 0) sqrt(rate * (1 - rate) / sum(ok)) else NA_real_,
    mean_p = mean(d$p_value, na.rm = TRUE),
    mean_statistic = mean(d$statistic, na.rm = TRUE),
    median_statistic = stats::median(d$statistic, na.rm = TRUE),
    mean_T_diff = mean(d$T_diff, na.rm = TRUE),
    mean_trace = mean(d$spectrum_trace, na.rm = TRUE),
    mean_scale_c = mean(d$scale_c, na.rm = TRUE),
    stringsAsFactors = FALSE)
})
summary <- do.call(rbind, summary_rows)
summary <- summary[order(summary$n_total, summary$thresholds,
                         summary$missing_rate, summary$missing_mechanism,
                         summary$pd_gamma,
                         match(summary$method,
                               c("naive", "SB", "adjusted", "scaled_shifted",
                                 "SS", "SF", "EBA2", "EBA4", "EBA6",
                                 "pEBA2", "pEBA4", "pEBA6", "pall", "pOLS",
                                 "all", "mixture"))), ]
rownames(summary) <- NULL
write_csv(summary, file.path(res_dir, "summary.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps,
    n_total = paste(cfg$n_total, collapse = ","),
    missing_rate = paste(cfg$missing_rate, collapse = ","),
    missing_mechanism = paste(cfg$missing_mechanism, collapse = ","),
    thresholds = paste(cfg$thresholds, collapse = ","),
    pd_gamma = paste(cfg$pd_gamma, collapse = ","),
    seed_base = cfg$seed_base,
    loading = loading,
    missing_items_group_b = paste(ov[8:10], collapse = ","),
    model = "chen_one_factor_10_indicator_configural_to_mplus_style_scalar",
    analysis = "magmaan DWLS pairwise ordinal, robust nested LRT plus FMG full-spectrum p-values",
    smoke = cfg$smoke),
  packages = "magmaan")

print(summary[, c("n_total", "thresholds", "missing_rate", "missing_mechanism",
                  "pd_gamma", "method", "rejection_rate", "mean_statistic",
                  "mean_p")],
      row.names = FALSE)
cat("\nWrote results to ", res_dir, "\n", sep = "")
