#!/usr/bin/env Rscript
# Ordinal pairwise-deletion Gamma probe.
#
# Compare the conventional nominal-N ordinal pairwise NACOV against the
# overlap-corrected NACOV from docs/research/notes/ordinal_pd_gamma.tex.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--missing-rate P]
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

parse_args <- function(args) {
  out <- list(reps = 50L, n = c(300L, 1000L), missing_rate = c(0, .30, .50),
              seed_base = 20260618L, smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N[,N]] ",
          "[--missing-rate P[,P]] [--seed-base S] [--smoke]\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") {
      i <- i + 1L; out$n <- as.integer(strsplit(args[[i]], ",")[[1L]])
    } else if (startsWith(a, "--n=")) {
      out$n <- as.integer(strsplit(sub("^--n=", "", a), ",")[[1L]])
    } else if (a == "--missing-rate") {
      i <- i + 1L; out$missing_rate <- as.numeric(strsplit(args[[i]], ",")[[1L]])
    } else if (startsWith(a, "--missing-rate=")) {
      out$missing_rate <- as.numeric(strsplit(sub("^--missing-rate=", "", a), ",")[[1L]])
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
    out$n <- 160L
    out$missing_rate <- c(0, .50)
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n)) || any(out$n < 50L)) stop("--n must be >= 50")
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .95)) {
    stop("--missing-rate must be in [0, .95)")
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaan))

res_dir <- ensure_results_dir()
ov <- paste0("y", 1:6)
model <- paste("f =~", paste(ov, collapse = " + "))
spec_h1 <- model_spec(model, ordered = ov, group = "group",
                      group_labels = c("A", "B"), parameterization = "delta")
spec_h0 <- model_spec(model, ordered = ov, group = "group",
                      group_labels = c("A", "B"), parameterization = "delta",
                      group_equal = "loadings")
thresholds <- c(-0.6, 0.2, 0.9)
loadings <- c(.80, .75, .70, .70, .65, .60)

draw_group <- function(n) {
  eta <- rnorm(n)
  z <- vapply(loadings, function(lam) {
    lam * eta + rnorm(n, sd = sqrt(1 - lam^2))
  }, numeric(n))
  z <- matrix(z, nrow = n, dimnames = list(NULL, ov))
  out <- as.data.frame(lapply(seq_len(ncol(z)), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE))
  }))
  names(out) <- ov
  out
}

draw_data <- function(n_per_group, missing_rate) {
  a <- draw_group(n_per_group)
  b <- draw_group(n_per_group)
  a$group <- "A"
  b$group <- "B"
  if (missing_rate > 0) {
    for (nm in c("y5", "y6")) {
      miss <- runif(n_per_group) < missing_rate
      b[[nm]][miss] <- NA
    }
  }
  rbind(a, b)
}

fit_cell <- function(dat, pd_gamma) {
  stats <- magmaan_core$data_ordinal_stats_from_df(
    dat, spec_h1, ordered = ov, group = "group",
    missing = "pairwise", pd_gamma = pd_gamma, full_wls_weight = FALSE)
  fit_h1 <- magmaan(spec_h1, stats, estimator = "DWLS")
  fit_h0 <- magmaan(spec_h0, stats, estimator = "DWLS")
  nested <- robust_nested_lrt(
    fit_h1, fit_h0, data = stats, method = "restriction_map",
    A.method = "delta", weight = "DWLS")
  p <- nested$p_scaled %||% nested$pvalue %||% nested$p
  data.frame(
    pd_gamma = pd_gamma,
    converged = isTRUE(fit_h1$converged) && isTRUE(fit_h0$converged),
    T_diff = nested$T_diff %||% NA_real_,
    T_scaled = nested$T_scaled %||% NA_real_,
    df_diff = nested$df_diff %||% NA_integer_,
    pvalue = as.numeric(p)[1L],
    stringsAsFactors = FALSE)
}

safe_fit_cell <- function(dat, pd_gamma) {
  tryCatch(fit_cell(dat, pd_gamma), error = function(e) {
    data.frame(pd_gamma = pd_gamma, converged = FALSE, T_diff = NA_real_,
               T_scaled = NA_real_, df_diff = NA_integer_, pvalue = NA_real_,
               error = conditionMessage(e), stringsAsFactors = FALSE)
  })
}

rows <- list()
ix <- 0L
for (n in cfg$n) {
  for (mr in cfg$missing_rate) {
    for (rep in seq_len(cfg$reps)) {
      set.seed(cfg$seed_base + as.integer(n * 1000 + round(mr * 1000) * 10 + rep))
      dat <- draw_data(n, mr)
      for (gamma in c("nominal", "overlap")) {
        ix <- ix + 1L
        got <- safe_fit_cell(dat, gamma)
        got$n_per_group <- n
        got$missing_rate <- mr
        got$rep <- rep
        got$reject_05 <- is.finite(got$pvalue) & got$pvalue < .05
        rows[[ix]] <- got
      }
      cat(sprintf("n=%d missing=%.2f rep=%d/%d\n", n, mr, rep, cfg$reps))
    }
  }
}

rejections <- do.call(rbind, rows)
rejections <- rejections[, c("n_per_group", "missing_rate", "rep", "pd_gamma",
                             "converged", "T_diff", "T_scaled", "df_diff",
                             "pvalue", "reject_05",
                             setdiff(names(rejections),
                                     c("n_per_group", "missing_rate", "rep",
                                       "pd_gamma", "converged", "T_diff",
                                       "T_scaled", "df_diff", "pvalue",
                                       "reject_05")))]
write_csv(rejections, file.path(res_dir, "rejections.csv"))

split_key <- interaction(rejections$n_per_group, rejections$missing_rate,
                         rejections$pd_gamma, drop = TRUE)
summary_rows <- lapply(split(rejections, split_key), function(d) {
  ok <- is.finite(d$pvalue)
  rate <- if (any(ok)) mean(d$reject_05[ok]) else NA_real_
  n_ok <- sum(ok)
  data.frame(
    n_per_group = d$n_per_group[1L],
    missing_rate = d$missing_rate[1L],
    pd_gamma = d$pd_gamma[1L],
    reps = nrow(d),
    usable = n_ok,
    failures = sum(!ok),
    rejection_rate = rate,
    mc_se = if (n_ok > 0) sqrt(rate * (1 - rate) / n_ok) else NA_real_,
    stringsAsFactors = FALSE)
})
summary <- do.call(rbind, summary_rows)
summary <- summary[order(summary$n_per_group, summary$missing_rate,
                         summary$pd_gamma), ]
write_csv(summary, file.path(res_dir, "summary.csv"))

metadata <- data.frame(
  reps = cfg$reps,
  n = paste(cfg$n, collapse = ","),
  missing_rate = paste(cfg$missing_rate, collapse = ","),
  seed_base = cfg$seed_base,
  smoke = cfg$smoke,
  magmaan_version = as.character(utils::packageVersion("magmaan")),
  stringsAsFactors = FALSE)
write_csv(metadata, file.path(res_dir, "metadata.csv"))

cat("wrote:\n")
cat("  ", file.path(res_dir, "rejections.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "summary.csv"), "\n", sep = "")
cat("  ", file.path(res_dir, "metadata.csv"), "\n", sep = "")
