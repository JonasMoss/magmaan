#!/usr/bin/env Rscript
# Chen et al. (2020) WLSMV_PD Type-I replication probe.
#
# Simulates the cheapest published bad cell: invariant one-factor ordinal CFA,
# two groups, N = 1000, symmetric thresholds, 50% rank-dependent missingness on
# group B items 8-10. Fits current lavaan WLSMV with pairwise deletion, includes
# the auxiliary variable as saturated observed covariances, and tests configural
# versus scalar invariance.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n-total N[,N]]
#                            [--missing-rate P[,P]]
#                            [--thresholds symmetric,asymmetric]
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

rbind_fill <- function(xs) {
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
  out <- list(
    reps = 100L,
    n_total = 1000L,
    missing_rate = .50,
    thresholds = "symmetric",
    seed_base = 20260618L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n-total N[,N]] ",
          "[--missing-rate P[,P]] [--thresholds symmetric,asymmetric] ",
          "[--seed-base S] [--smoke]\n", sep = "")
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
    out$reps <- 5L
    out$n_total <- 300L
    out$missing_rate <- .50
    out$thresholds <- "symmetric"
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n_total)) || any(out$n_total < 100L) ||
      any(out$n_total %% 2L != 0L)) {
    stop("--n-total must contain even integers >= 100", call. = FALSE)
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
  out
}

threshold_values <- function(shape) {
  switch(shape,
         symmetric = c(-1.3, -0.47, 0.47, 1.3),
         asymmetric = c(-0.253, 0.385, 0.842, 1.282),
         stop("unknown threshold shape", call. = FALSE))
}

chen_wlsmv_pd_target <- function(n_total, thresholds, missing_rate) {
  grid <- data.frame(
    thresholds = rep(c("asymmetric", "symmetric"), each = 9L),
    n_total = rep(rep(c(300L, 600L, 1000L), each = 3L), 2L),
    missing_rate = rep(c(0, .30, .50), 6L),
    chen_type1 = c(
      .062, .072, .130, .058, .094, .250, .046, .108, .388,
      .062, .088, .180, .036, .092, .292, .052, .122, .550
    )
  )
  hit <- grid$n_total == n_total &
    grid$thresholds == thresholds &
    abs(grid$missing_rate - missing_rate) < 1e-12
  if (any(hit)) grid$chen_type1[which(hit)[1L]] else NA_real_
}

first_finite <- function(x) {
  hit <- which(is.finite(x))
  if (length(hit)) hit[1L] else NA_integer_
}

fit_measure <- function(fit, name) {
  out <- tryCatch(lavaan::fitMeasures(fit, name), error = function(e) NA_real_)
  as.numeric(out[[1L]])
}

capture_warnings <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = warnings)
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("lavaan")
suppressPackageStartupMessages(library(lavaan))

res_dir <- ensure_results_dir()
ov <- paste0("v", 1:10)
model <- paste0("f =~ ", paste(ov, collapse = " + "))
analysis_model <- paste(
  model,
  paste("aux ~~", paste(c("aux", ov), collapse = " + ")),
  sep = "\n")

draw_group <- function(n, group_label, thresholds) {
  eta <- rnorm(n)
  aux <- 0.5 * eta + sqrt(1 - 0.5^2) * rnorm(n)
  z <- matrix(NA_real_, n, length(ov), dimnames = list(NULL, ov))
  for (j in seq_along(ov)) {
    z[, j] <- 0.6 * eta + rnorm(n, sd = sqrt(1 - 0.6^2))
  }
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out$aux <- aux
  out$group <- group_label
  out
}

apply_group_b_missing <- function(dat, missing_rate) {
  if (missing_rate <= 0) {
    dat$group <- factor(dat$group, levels = c("A", "B"))
    return(dat)
  }
  b_rows <- which(dat$group == "B")
  nb <- length(b_rows)
  target <- as.integer(round(missing_rate * nb))
  ranks <- rank(dat$aux[b_rows], ties.method = "random")
  weights <- pmax(nb - ranks, 0)
  if (!any(weights > 0)) weights <- rep(1, nb)
  for (nm in ov[8:10]) {
    picked <- sample.int(nb, size = target, replace = FALSE, prob = weights)
    dat[[nm]][b_rows[picked]] <- NA
  }
  dat$group <- factor(dat$group, levels = c("A", "B"))
  dat
}

draw_data <- function(n_total, threshold_shape, missing_rate) {
  n_group <- as.integer(n_total / 2L)
  thresholds <- threshold_values(threshold_shape)
  dat <- rbind(draw_group(n_group, "A", thresholds),
               draw_group(n_group, "B", thresholds))
  apply_group_b_missing(dat, missing_rate)
}

realized_missing_b_items <- function(dat) {
  b <- dat$group == "B"
  mean(is.na(dat[b, ov[8:10]]))
}

fit_rep <- function(dat, n_total, threshold_shape, missing_rate, rep) {
  out <- tryCatch(capture_warnings({
    fit_h1 <- lavaan::cfa(
      analysis_model, data = dat, ordered = ov, group = "group",
      estimator = "WLSMV", missing = "pairwise", se = "none",
      fixed.x = FALSE
    )
    fit_h0 <- lavaan::cfa(
      analysis_model, data = dat, ordered = ov, group = "group",
      estimator = "WLSMV", missing = "pairwise", se = "none",
      fixed.x = FALSE, group.equal = c("loadings", "thresholds")
    )
    lrt <- as.data.frame(lavaan::lavTestLRT(fit_h1, fit_h0,
                                            method = "satorra.2000"))
    diff_row <- first_finite(lrt[["Chisq diff"]])
    robust <- data.frame(
      method = "lavaan_wlsmv_pairwise_satorra2000",
      p_value = lrt[["Pr(>Chisq)"]][diff_row],
      stat = lrt[["Chisq diff"]][diff_row],
      df_diff = lrt[["Df diff"]][diff_row],
      stringsAsFactors = FALSE
    )
    chisq_h1 <- fit_measure(fit_h1, "chisq")
    chisq_h0 <- fit_measure(fit_h0, "chisq")
    df_h1 <- fit_measure(fit_h1, "df")
    df_h0 <- fit_measure(fit_h0, "df")
    stat_naive <- chisq_h0 - chisq_h1
    df_naive <- df_h0 - df_h1
    naive <- data.frame(
      method = "lavaan_wlsmv_pairwise_naive_diff",
      p_value = stats::pchisq(stat_naive, df_naive, lower.tail = FALSE),
      stat = stat_naive,
      df_diff = df_naive,
      stringsAsFactors = FALSE
    )
    rows <- rbind(robust, naive)
    rows$df_h1 <- df_h1
    rows$df_h0 <- df_h0
    rows$npar_h1 <- fit_measure(fit_h1, "npar")
    rows$npar_h0 <- fit_measure(fit_h0, "npar")
    rows$converged_h1 <- lavaan::lavInspect(fit_h1, "converged")
    rows$converged_h0 <- lavaan::lavInspect(fit_h0, "converged")
    rows
  }), error = function(e) {
    list(
      value = data.frame(
        method = c("lavaan_wlsmv_pairwise_satorra2000",
                   "lavaan_wlsmv_pairwise_naive_diff"),
        p_value = NA_real_, stat = NA_real_, df_diff = NA_real_,
        df_h1 = NA_real_, df_h0 = NA_real_,
        npar_h1 = NA_real_, npar_h0 = NA_real_,
        converged_h1 = FALSE, converged_h0 = FALSE,
        stringsAsFactors = FALSE
      ),
      warnings = character(),
      error = conditionMessage(e)
    )
  })
  rows <- out$value
  rows$n_total <- n_total
  rows$n_per_group <- as.integer(n_total / 2L)
  rows$thresholds <- threshold_shape
  rows$missing_rate <- missing_rate
  rows$realized_missing_b_items <- realized_missing_b_items(dat)
  rows$rep <- rep
  rows$reject_05 <- is.finite(rows$p_value) & rows$p_value < .05
  rows$chen_type1 <- chen_wlsmv_pd_target(n_total, threshold_shape, missing_rate)
  rows$error <- out$error %||% ""
  rows$warnings <- paste(unique(out$warnings), collapse = " | ")
  rows
}

rows <- list()
ix <- 0L
for (n_total in cfg$n_total) {
  for (missing_rate in cfg$missing_rate) {
    for (threshold_shape in cfg$thresholds) {
      for (rep in seq_len(cfg$reps)) {
        set.seed(cfg$seed_base +
                   as.integer(n_total * 1000) +
                   as.integer(round(missing_rate * 1000) * 10) +
                   match(threshold_shape, c("symmetric", "asymmetric")) * 1000000L +
                   rep)
        dat <- draw_data(n_total, threshold_shape, missing_rate)
        got <- fit_rep(dat, n_total, threshold_shape, missing_rate, rep)
        ix <- ix + 1L
        rows[[ix]] <- got
        cat(sprintf("N=%d thresholds=%s missing=%.2f rep=%d/%d\n",
                    n_total, threshold_shape, missing_rate, rep, cfg$reps))
      }
    }
  }
}

p_values <- rbind_fill(rows)
front <- c("n_total", "n_per_group", "thresholds", "missing_rate",
           "realized_missing_b_items", "rep", "method", "p_value",
           "reject_05", "stat", "df_diff", "df_h1", "df_h0",
           "npar_h1", "npar_h0", "converged_h1", "converged_h0",
           "chen_type1", "error", "warnings")
p_values <- p_values[, c(front, setdiff(names(p_values), front))]
write_csv(p_values, file.path(res_dir, "p_values.csv"))

split_key <- interaction(p_values$n_total, p_values$thresholds,
                         p_values$missing_rate, p_values$method, drop = TRUE)
summary_rows <- lapply(split(p_values, split_key), function(d) {
  ok <- is.finite(d$p_value)
  rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
  n_ok <- sum(ok)
  data.frame(
    n_total = d$n_total[1L],
    n_per_group = d$n_per_group[1L],
    thresholds = d$thresholds[1L],
    missing_rate = d$missing_rate[1L],
    realized_missing_b_items = mean(d$realized_missing_b_items, na.rm = TRUE),
    method = d$method[1L],
    reps = nrow(d),
    usable = n_ok,
    failures = sum(!ok),
    rejection_rate = rate,
    mc_se = if (n_ok > 0) sqrt(rate * (1 - rate) / n_ok) else NA_real_,
    median_p = if (n_ok > 0) stats::median(d$p_value[ok]) else NA_real_,
    mean_stat = mean(d$stat, na.rm = TRUE),
    mean_df_diff = mean(d$df_diff, na.rm = TRUE),
    df_diff_min = suppressWarnings(min(d$df_diff, na.rm = TRUE)),
    df_diff_max = suppressWarnings(max(d$df_diff, na.rm = TRUE)),
    chen_type1 = d$chen_type1[1L],
    stringsAsFactors = FALSE
  )
})
rejection_rates <- do.call(rbind, summary_rows)
rejection_rates <- rejection_rates[order(rejection_rates$n_total,
                                         rejection_rates$thresholds,
                                         rejection_rates$missing_rate,
                                         rejection_rates$method), ]
write_csv(rejection_rates, file.path(res_dir, "rejection_rates.csv"))
write_csv(rejection_rates, file.path(res_dir, "summary.csv"))

write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps,
    n_total = cfg$n_total,
    missing_rate = cfg$missing_rate,
    thresholds = cfg$thresholds,
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    model = "chen_2020_one_factor_10_ordinal_invariant",
    missing_generator = "exact weighted rank sampling on group B items 8-10",
    auxiliary = "included as saturated aux ~~ aux + items covariances",
    lavaan_test = "WLSMV pairwise deletion; lavTestLRT method=satorra.2000"
  ),
  packages = "lavaan"
)

cat("wrote:\n")
cat("  ", file.path(res_dir, "p_values.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "rejection_rates.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
