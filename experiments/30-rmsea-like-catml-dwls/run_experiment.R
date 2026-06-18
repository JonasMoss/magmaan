#!/usr/bin/env Rscript
# RMSEA-like CATML-at-DWLS consistency probe.
#
# The target is lavaan's categorical robust RMSEA ingredient for all-ordinal
# DWLS/WLSMV fits: evaluate the CATML correlation criterion at the already-fit
# DWLS solution. Under misspecification this is a criterion-at-estimator object,
# not a refitted CATML optimum.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N[,N]]
#       [--missing-rate P[,P]] [--n-ref N] [--seed-base S] [--smoke]

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

elapsed_ms <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, elapsed_ms = 1000 * (proc.time()[["elapsed"]] - t0))
}

parse_args <- function(args) {
  out <- list(
    reps = 20L,
    n = c(500L, 2000L, 8000L),
    missing_rate = c(.30),
    n_ref = 100000L,
    seed_base = 20260618L,
    smoke = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N[,N]]\n",
          "       [--missing-rate P[,P]] [--n-ref N] [--seed-base S] [--smoke]\n",
          sep = "")
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
    } else if (a == "--n-ref") {
      i <- i + 1L; out$n_ref <- as.integer(args[[i]])
    } else if (startsWith(a, "--n-ref=")) {
      out$n_ref <- as.integer(sub("^--n-ref=", "", a))
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
    out$reps <- 2L
    out$n <- c(300L, 1000L)
    out$missing_rate <- .30
    out$n_ref <- 12000L
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be positive")
  if (any(!is.finite(out$n)) || any(out$n < 80L)) stop("--n must be >= 80")
  if (!is.finite(out$n_ref) || out$n_ref < 1000L) stop("--n-ref must be >= 1000")
  if (any(!is.finite(out$missing_rate)) ||
      any(out$missing_rate < 0 | out$missing_rate >= .80)) {
    stop("--missing-rate must be in [0, .80)", call. = FALSE)
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

res_dir <- ensure_results_dir()
core <- magmaan::magmaan_core
ov <- paste0("y", 1:6)
model <- paste0("f =~ ", paste(ov, collapse = " + "))
spec <- model_spec(model, ordered = ov, parameterization = "delta")

true_correlation <- function() {
  load <- c(.82, .76, .70, .80, .73, .66)
  phi <- matrix(c(1, .25, .25, 1), 2, 2)
  lambda <- matrix(0, length(load), 2)
  lambda[1:3, 1] <- load[1:3]
  lambda[4:6, 2] <- load[4:6]
  R <- lambda %*% phi %*% t(lambda)
  diag(R) <- 1
  R
}

thresholds <- c(-.85, -.20, .55, 1.20)
R_true <- true_correlation()
R_chol <- chol(R_true)

draw_data <- function(n, seed) {
  set.seed(seed)
  z <- matrix(rnorm(n * length(ov)), nrow = n) %*% R_chol
  out <- as.data.frame(lapply(seq_along(ov), function(j) {
    ordered(cut(z[, j], c(-Inf, thresholds, Inf), labels = FALSE),
            levels = seq_len(length(thresholds) + 1L))
  }))
  names(out) <- ov
  out
}

apply_mcar <- function(dat, missing_rate, seed) {
  if (missing_rate <= 0) return(dat)
  set.seed(seed)
  for (nm in ov) {
    dat[[nm]][runif(nrow(dat)) < missing_rate] <- NA
  }
  dat
}

fit_once <- function(n, missing_rate, seed, role = "replicate") {
  timed <- elapsed_ms({
    dat <- draw_data(n, seed)
    dat <- apply_mcar(dat, missing_rate, seed + 1000003L)
    stats <- core$data_ordinal_stats_from_df(
      dat, spec, missing = "pairwise", pd_gamma = "overlap")
    fit <- core$fit_dwls_ordinal(
      spec, stats,
      control = list(max_iter = 4000L, ftol = 1e-12, gtol = 1e-8))
    cm <- core$measures_ordinal_catml_dwls_rmsea(fit)
    list(stats = stats, fit = fit, cm = cm, missing_realized = mean(is.na(dat[ov])))
  })
  x <- timed$value
  cm <- x$cm
  data.frame(
    role = role,
    n = as.integer(n),
    missing_rate = missing_rate,
    missing_realized = x$missing_realized,
    seed = as.integer(seed),
    converged = isTRUE(x$fit$converged),
    iterations = as.integer(x$fit$iterations),
    elapsed_ms = timed$elapsed_ms,
    XX3 = as.numeric(cm$XX3),
    F_catml_at_dwls = as.numeric(cm$XX3) / n,
    df3 = as.integer(cm$df3),
    c_hat3 = as.numeric(cm[["c.hat3"]]),
    XX3_scaled = as.numeric(cm[["XX3.scaled"]]),
    rmsea_robust = as.numeric(cm[["rmsea.robust"]]),
    stringsAsFactors = FALSE
  )
}

safe_fit_once <- function(...) {
  tryCatch(fit_once(...), error = function(e) {
    args <- list(...)
    data.frame(
      role = args$role %||% "replicate",
      n = as.integer(args$n),
      missing_rate = as.numeric(args$missing_rate),
      missing_realized = NA_real_,
      seed = as.integer(args$seed),
      converged = FALSE,
      iterations = NA_integer_,
      elapsed_ms = NA_real_,
      XX3 = NA_real_,
      F_catml_at_dwls = NA_real_,
      df3 = NA_integer_,
      c_hat3 = NA_real_,
      XX3_scaled = NA_real_,
      rmsea_robust = NA_real_,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
}

target_rows <- lapply(seq_along(cfg$missing_rate), function(i) {
  mr <- cfg$missing_rate[[i]]
  seed <- cfg$seed_base + 900000L + 1000L * i
  safe_fit_once(n = cfg$n_ref, missing_rate = mr, seed = seed, role = "target")
})
targets <- rbind_fill(target_rows)

rep_rows <- list()
k <- 0L
for (mr_i in seq_along(cfg$missing_rate)) {
  mr <- cfg$missing_rate[[mr_i]]
  for (n_i in seq_along(cfg$n)) {
    n <- cfg$n[[n_i]]
    for (rep in seq_len(cfg$reps)) {
      k <- k + 1L
      seed <- cfg$seed_base + 100000L * mr_i + 1000L * n_i + rep
      row <- safe_fit_once(n = n, missing_rate = mr, seed = seed,
                           role = "replicate")
      row$rep <- rep
      rep_rows[[length(rep_rows) + 1L]] <- row
      cat(sprintf("rep %d: n=%d missing=%.2f rep=%d status=%s\n",
                  k, n, mr, rep,
                  if (isTRUE(row$converged[[1L]])) "ok" else "fail"))
    }
  }
}
replicates <- rbind_fill(rep_rows)

target_for <- function(mr) {
  targets[abs(targets$missing_rate - mr) < 1e-12, , drop = FALSE][1L, ]
}

summaries <- lapply(split(replicates, list(replicates$n, replicates$missing_rate),
                          drop = TRUE), function(d) {
  mr <- d$missing_rate[[1L]]
  t <- target_for(mr)
  ok <- d[isTRUE(d$converged) | d$converged == TRUE, , drop = FALSE]
  if (!nrow(ok)) {
    return(data.frame(n = d$n[[1L]], missing_rate = mr, usable = 0L,
                      target_F = t$F_catml_at_dwls,
                      mean_F = NA_real_, bias_F = NA_real_,
                      mae_F = NA_real_, target_rmsea = t$rmsea_robust,
                      mean_rmsea = NA_real_, bias_rmsea = NA_real_,
                      mae_rmsea = NA_real_, mean_elapsed_ms = NA_real_))
  }
  data.frame(
    n = d$n[[1L]],
    missing_rate = mr,
    usable = nrow(ok),
    target_F = t$F_catml_at_dwls,
    mean_F = mean(ok$F_catml_at_dwls),
    bias_F = mean(ok$F_catml_at_dwls - t$F_catml_at_dwls),
    mae_F = mean(abs(ok$F_catml_at_dwls - t$F_catml_at_dwls)),
    target_rmsea = t$rmsea_robust,
    mean_rmsea = mean(ok$rmsea_robust),
    bias_rmsea = mean(ok$rmsea_robust - t$rmsea_robust),
    mae_rmsea = mean(abs(ok$rmsea_robust - t$rmsea_robust)),
    mean_elapsed_ms = mean(ok$elapsed_ms, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
})
summary <- rbind_fill(summaries)
summary <- summary[order(summary$missing_rate, summary$n), , drop = FALSE]

write_csv(targets, file.path(res_dir, "targets.csv"))
write_csv(replicates, file.path(res_dir, "replicates.csv"))
write_csv(summary, file.path(res_dir, "summary.csv"))
write_metadata(
  file.path(res_dir, "metadata.csv"),
  values = list(
    reps = cfg$reps,
    n = cfg$n,
    missing_rate = cfg$missing_rate,
    n_ref = cfg$n_ref,
    seed_base = cfg$seed_base,
    smoke = cfg$smoke,
    population = "six ordinal indicators from two correlated Gaussian factors; one-factor DWLS fit"
  ),
  packages = c("magmaan")
)

cat("wrote results under ", res_dir, "\n", sep = "")
