#!/usr/bin/env Rscript
# Rhemtulla, Brosseau-Liard & Savalei (2012) replication.
#
# "When Can Categorical Variables Be Treated as Continuous?" A two-factor CFA
# (10 indicators, loadings .3-.7, factor correlation .3) is fit to ordinal data
# with two robust estimators on the *same* categorized integer matrix:
#
#   cat-LS  — categorical least squares (DWLS on polychoric correlations) with
#             robust mean-and-variance-adjusted chi-square (T_cat-MV) and SEs.
#   ML      — normal-theory ML treating the integer codes as continuous, with
#             Satorra-Bentler / mean-variance-adjusted chi-square (T_ML-MV).
#
# Headline question: how many response categories before treating ordinal data
# as continuous stops costing parameter accuracy? v1 scope: number of categories
# {2..7} x N {100,150,350,600}, symmetric thresholds, underlying y* normal.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--cells FILTER] [--seed-base S]
#                            [--smoke] [--lavaan-parity]
#
# `--cells FILTER` is a comma-separated list of `key=value` pairs over
# `categories` and `N` (e.g. `categories=2,N=600`) restricting the 24-cell grid.

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
  out <- list(reps = 10L, cells_filter = NULL, seed_base = 20260530L,
              smoke = FALSE, lavaan_parity = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--cells FILTER] ",
          "[--seed-base S] [--smoke] [--lavaan-parity]\n", sep = "")
      cat("  --cells: comma-separated key=value over `categories`,`N` ",
          "(e.g. categories=2,N=600)\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cells") {
      i <- i + 1L; out$cells_filter <- args[[i]]
    } else if (startsWith(a, "--cells=")) {
      out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--seed-base") {
      i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) {
      out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--smoke") {
      out$smoke <- TRUE
    } else if (a == "--lavaan-parity") {
      out$lavaan_parity <- TRUE
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) {
    stop("--reps must be a positive integer", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

project_root <- repo_root()
results_dir <- ensure_results_dir()

set_single_threaded_math()

require_pkg("magmaan")
core <- magmaan::magmaan_core

# ── Population model ─────────────────────────────────────────────────────────
# Rhemtulla Model 1: two-factor CFA, five indicators per factor, loadings
# .3/.4/.5/.6/.7 on each factor, factor correlation .3, unit-variance y*.
P_ITEMS <- 10L
TRUE_LOADINGS <- rep(c(0.3, 0.4, 0.5, 0.6, 0.7), times = 2L)
TRUE_FCOR <- 0.3
VARNAMES <- paste0("x", seq_len(P_ITEMS))
SYNTAX <- paste0("f1 =~ ", paste(VARNAMES[1:5], collapse = " + "), "\n",
                 "f2 =~ ", paste(VARNAMES[6:10], collapse = " + "), "\n",
                 "f1 ~~ f2")

build_population <- function() {
  Lambda <- matrix(0.0, P_ITEMS, 2)
  Lambda[1:5, 1L] <- TRUE_LOADINGS[1:5]
  Lambda[6:10, 2L] <- TRUE_LOADINGS[6:10]
  Phi <- matrix(c(1, TRUE_FCOR, TRUE_FCOR, 1), 2, 2)
  Sigma <- Lambda %*% Phi %*% t(Lambda)
  diag(Sigma) <- 1                     # unit-variance y*; residual = 1 - lambda^2
  list(Sigma = Sigma, L = chol(Sigma))
}

# Model specs are fixed across cells; build once.
spec_ord  <- magmaan::model_spec(SYNTAX, ordered = VARNAMES,
                                 parameterization = "delta", std_lv = TRUE)
spec_cont <- magmaan::model_spec(SYNTAX, std_lv = TRUE)

# ── Thresholds (symmetric, analytic) ─────────────────────────────────────────
# K-1 thresholds evenly dividing [-2.5, 2.5] into K categories (paper Method /
# Figure 1). Categories coded 1..K.
symmetric_thresholds <- function(K) {
  seq(-2.5, 2.5, length.out = K + 1L)[2:K]
}

# Marginal skew / excess-kurtosis of the K-category variable under normal y*.
cat_moments <- function(taus) {
  cuts <- c(-Inf, taus, Inf)
  p <- diff(stats::pnorm(cuts))
  if (any(p <= 0)) return(c(NA_real_, NA_real_))
  k <- seq_along(p)
  mu <- sum(k * p)
  s2 <- sum((k - mu)^2 * p)
  m3 <- sum((k - mu)^3 * p)
  m4 <- sum((k - mu)^4 * p)
  if (s2 <= 0) return(c(NA_real_, NA_real_))
  c(m3 / s2^1.5, m4 / s2^2 - 3)
}

# Rhemtulla Table 1, normal underlying / symmetric thresholds (skew 0 for all).
TABLE1_KURT <- c(`2` = -2.00, `3` = -0.54, `4` = -0.53, `5` = -0.47,
                 `6` = -0.42, `7` = -0.41)

discretize <- function(y_cont, taus) {
  matrix(findInterval(y_cont, taus, left.open = TRUE) + 1L,
         nrow = nrow(y_cont), ncol = ncol(y_cont))
}

# ── χ² trace-moment adjustments (continuous ML) ──────────────────────────────
# Verbatim from experiments/07-maydeu-olivares-2017: SB / mean-variance /
# scaled-shifted need only tr(M), tr(M²) and df.
sb_from_moments <- function(T, mom) {
  df <- as.integer(mom$df %||% NA_integer_)
  tr <- as.numeric(mom$trace %||% NA_real_)
  if (!is.finite(T) || !is.finite(tr) || !is.finite(df) || df <= 0) {
    return(list(stat = NA_real_, df = NA_integer_))
  }
  c_scale <- tr / df
  list(stat = if (c_scale > 0) T / c_scale else NA_real_, df = df)
}
mv_from_moments <- function(T, mom) {
  tr <- as.numeric(mom$trace %||% NA_real_)
  tr2 <- as.numeric(mom$trace_sq %||% NA_real_)
  if (!is.finite(T) || !is.finite(tr) || !is.finite(tr2) || tr2 <= 0) {
    return(list(stat = NA_real_, df = NA_real_))
  }
  list(stat = T * tr / tr2, df = tr * tr / tr2)
}

# ── Sample-stats wrapper for the continuous ML path (mirrors 07) ─────────────
sample_data_from_matrix <- function(X) {
  ss <- core$data_sample_stats_from_raw(list(X))
  nm <- colnames(X)
  dimnames(ss$S[[1L]]) <- list(nm, nm)
  if (!is.null(ss$mean) && length(ss$mean)) names(ss$mean[[1L]]) <- nm
  ss$X <- list(X)
  ss$ov_names <- list(nm)
  ss$group_var <- ""
  ss$group_labels <- character()
  ss$scaling <- "n"
  class(ss) <- c("magmaan_data", "list")
  ss
}

# Free-parameter index maps (same partable layout for both estimators).
loading_map <- function(pt) {
  i <- which(pt$op == "=~" & pt$free > 0L)
  list(free = pt$free[i], name = pt$rhs[i], factor = pt$lhs[i])
}
fcor_free <- function(pt) {
  i <- which(pt$op == "~~" & pt$lhs == "f1" & pt$rhs == "f2" & pt$free > 0L)
  if (length(i)) pt$free[i[[1L]]] else NA_integer_
}

# Under std_lv the sign of each factor is unidentified (a whole loading column
# may flip). The population loadings are all positive, so orient each factor's
# loadings positive; the factor correlation flips once per flipped factor.
orient_signs <- function(load_factor, load_est, fcor_est) {
  flip <- c(f1 = 1, f2 = 1)
  for (f in c("f1", "f2")) {
    idx <- which(load_factor == f)
    if (length(idx) && sum(load_est[idx], na.rm = TRUE) < 0) {
      load_est[idx] <- -load_est[idx]
      flip[[f]] <- -1
    }
  }
  if (is.finite(fcor_est)) fcor_est <- fcor_est * flip[["f1"]] * flip[["f2"]]
  list(load_est = load_est, fcor_est = fcor_est)
}

# Build the param long-format rows for one estimator from aligned vectors.
param_rows <- function(estimator, load_names, load_est, load_se,
                       fcor_est, fcor_se) {
  rbind(
    data.frame(estimator = estimator, param_type = "loading",
               param_name = load_names, true = TRUE_LOADINGS,
               est = load_est, se = load_se, stringsAsFactors = FALSE),
    data.frame(estimator = estimator, param_type = "factor_cor",
               param_name = "f1~~f2", true = TRUE_FCOR,
               est = fcor_est, se = fcor_se, stringsAsFactors = FALSE)
  )
}

# ── Per-replicate pipeline ───────────────────────────────────────────────────
# Generate one integer matrix and fit both estimators on it. Each estimator
# contributes param rows, chi2 rows, and a convergence/improper record; an
# estimator that errors or fails to converge contributes only the conv record.
fit_one_rep <- function(pop, taus, N, seed, keep_parity = FALSE) {
  set.seed(seed)
  z <- matrix(stats::rnorm(N * P_ITEMS), N, P_ITEMS)
  y_cont <- z %*% pop$L
  X <- discretize(y_cont, taus)
  storage.mode(X) <- "double"
  colnames(X) <- VARNAMES
  df_ord <- as.data.frame(lapply(seq_len(P_ITEMS), function(j) ordered(X[, j])))
  names(df_ord) <- VARNAMES

  param_list <- list()
  chi2_list <- list()
  conv_list <- list()

  # ---- cat-LS (DWLS on polychorics) ----
  cat_ok <- FALSE; cat_improper <- NA
  cat_fit <- tryCatch({
    d <- core$data_ordinal_stats_from_df(df_ord, spec_ord)
    f <- core$fit_dwls_ordinal(
      spec_ord, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
    list(d = d, f = f)
  }, error = function(e) e)
  cat_parity <- NULL
  if (!inherits(cat_fit, "error") && isTRUE(cat_fit$f$converged)) {
    f <- cat_fit$f
    rob <- tryCatch(core$robust_ordinal(f, cat_fit$d), error = function(e) e)
    pt <- f$partable
    lm <- loading_map(pt); fc <- fcor_free(pt)
    th <- as.numeric(f$theta)
    # Delta-parameterization loadings are already in the correlation metric
    # (unit-variance y*), directly comparable to the population loadings.
    load_est <- th[lm$free]
    fcor_est <- if (is.na(fc)) NA_real_ else th[fc]
    se_vec <- if (inherits(rob, "error")) rep(NA_real_, length(th)) else rob$se
    load_se <- se_vec[lm$free]
    fcor_se <- if (is.na(fc)) NA_real_ else se_vec[fc]
    oriented <- orient_signs(lm$factor, load_est, fcor_est)
    load_est <- oriented$load_est; fcor_est <- oriented$fcor_est
    cat_improper <- any(abs(load_est) > 1, na.rm = TRUE) ||
      (is.finite(fcor_est) && abs(fcor_est) > 1)
    cat_ok <- TRUE
    param_list[["catLS"]] <- param_rows("catLS", lm$name, load_est, load_se,
                                        fcor_est, fcor_se)
    if (!inherits(rob, "error")) {
      chi2_list[["catLS"]] <- data.frame(
        estimator = "catLS",
        stat_name = c("SB", "MV"),
        statistic = c(rob$satorra_bentler$chi2_scaled,
                      rob$mean_var_adjusted$chi2_adj),
        df = c(rob$satorra_bentler$df, rob$mean_var_adjusted$df_adj),
        stringsAsFactors = FALSE)
    }
    if (isTRUE(keep_parity)) {
      cat_parity <- list(load_est = load_est, fcor_est = fcor_est,
                         chi2 = chi2_list[["catLS"]])
    }
  }
  conv_list[["catLS"]] <- data.frame(estimator = "catLS", converged = cat_ok,
                                     improper = isTRUE(cat_improper),
                                     stringsAsFactors = FALSE)

  # ---- continuous robust ML on the same integer matrix ----
  ml_ok <- FALSE; ml_improper <- NA
  ml_parity <- NULL
  dat <- sample_data_from_matrix(X)
  ml <- tryCatch(core$fit_ml(spec_cont, dat), error = function(e) e)
  if (!inherits(ml, "error") && isTRUE(ml$converged)) {
    pt <- ml$partable
    lm <- loading_map(pt); fc <- fcor_free(pt)
    # Robust sandwich (bread = expected, meat = empirical), then standardize the
    # solution so loadings are comparable to the population loadings.
    Zc <- tryCatch(core$robust_casewise_contributions(pt, X),
                   error = function(e) e)
    samp <- core$fit_sample_stats(ml)
    T_ML <- core$inference_chi2_stat(samp, ml$fmin)
    df_ML <- core$inference_df_stat(pt, samp)
    pair <- if (inherits(Zc, "error")) NULL else tryCatch(
      core$robust_se_both_breads_zc(ml, Zc, nrow(X),
                                    moments = "structured", cov = "empirical"),
      error = function(e) NULL)
    tm <- if (inherits(Zc, "error")) NULL else tryCatch(
      core$robust_test_moments_both_breads_zc(ml, Zc, nrow(X),
                                              moments = "structured"),
      error = function(e) NULL)
    # Standardized point estimates + standardized robust SEs.
    robust_vcov <- if (is.null(pair)) NULL else pair$expected$vcov
    sac <- tryCatch(core$measures_standardize_all(
      ml, robust_vcov %||% {
        infoE <- core$inference_information_expected(ml)
        core$inference_vcov_fit(infoE, ml)
      }), error = function(e) NULL)
    th_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$theta
    se_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$se
    load_est <- th_std[lm$free]
    load_se <- se_std[lm$free]
    fcor_est <- if (is.na(fc)) NA_real_ else th_std[fc]
    fcor_se <- if (is.na(fc)) NA_real_ else se_std[fc]
    oriented <- orient_signs(lm$factor, load_est, fcor_est)
    load_est <- oriented$load_est; fcor_est <- oriented$fcor_est
    ml_improper <- any(abs(load_est) > 1, na.rm = TRUE) ||
      (is.finite(fcor_est) && abs(fcor_est) > 1)
    ml_ok <- TRUE
    param_list[["ML"]] <- param_rows("ML", lm$name, load_est, load_se,
                                     fcor_est, fcor_se)
    T_MV <- if (is.null(tm)) list(stat = NA_real_, df = NA_real_) else
      mv_from_moments(T_ML, tm$expected)
    T_SB <- if (is.null(tm)) list(stat = NA_real_, df = NA_integer_) else
      sb_from_moments(T_ML, tm$expected)
    chi2_list[["ML"]] <- data.frame(
      estimator = "ML",
      stat_name = c("naive", "SB", "MV"),
      statistic = c(T_ML, T_SB$stat, T_MV$stat),
      df = c(df_ML, T_SB$df, T_MV$df),
      stringsAsFactors = FALSE)
    if (isTRUE(keep_parity)) {
      ml_parity <- list(load_est = load_est, fcor_est = fcor_est,
                        chi2 = chi2_list[["ML"]])
    }
  }
  conv_list[["ML"]] <- data.frame(estimator = "ML", converged = ml_ok,
                                  improper = isTRUE(ml_improper),
                                  stringsAsFactors = FALSE)

  list(
    params = if (length(param_list)) do.call(rbind, param_list) else NULL,
    chi2 = if (length(chi2_list)) do.call(rbind, chi2_list) else NULL,
    conv = do.call(rbind, conv_list),
    parity = if (isTRUE(keep_parity))
      list(data = df_ord, catLS = cat_parity, ML = ml_parity) else NULL
  )
}

# ── Cell grid ────────────────────────────────────────────────────────────────
parse_cells_filter <- function(s) {
  if (is.null(s) || !nzchar(s)) return(list())
  pairs <- strsplit(s, ",", fixed = TRUE)[[1L]]
  out <- list()
  for (p in pairs) {
    kv <- strsplit(p, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("bad --cells entry: ", p, call. = FALSE)
    out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}

cell_grid <- expand.grid(
  categories = 2:7,
  N = c(100L, 150L, 350L, 600L),
  stringsAsFactors = FALSE
)
if (isTRUE(args$smoke)) {
  cell_grid <- cell_grid[cell_grid$categories %in% c(2L, 5L, 7L) &
                           cell_grid$N %in% c(100L, 600L), , drop = FALSE]
}

filter <- parse_cells_filter(args$cells_filter)
for (key in names(filter)) {
  if (!(key %in% names(cell_grid))) {
    stop("--cells: unknown key '", key, "'", call. = FALSE)
  }
  target <- as.numeric(strsplit(filter[[key]], "|", fixed = TRUE)[[1L]])
  cell_grid <- cell_grid[cell_grid[[key]] %in% target, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)

# Thresholds + the Table-1 reproducibility check, written once.
unique_K <- sort(unique(cell_grid$categories))
threshold_rows <- do.call(rbind, lapply(unique_K, function(K) {
  taus <- symmetric_thresholds(K)
  props <- diff(stats::pnorm(c(-Inf, taus, Inf)))
  mom <- cat_moments(taus)
  data.frame(
    categories = K,
    thresholds = paste(sprintf("%.3f", taus), collapse = ";"),
    proportions = paste(sprintf("%.3f", props), collapse = ";"),
    skew_achieved = mom[[1L]],
    kurt_achieved = mom[[2L]],
    kurt_table1 = unname(TABLE1_KURT[as.character(K)]),
    stringsAsFactors = FALSE)
}))
utils::write.csv(threshold_rows, file.path(results_dir, "thresholds.csv"),
                 row.names = FALSE)
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"),
                 row.names = FALSE)

pop <- build_population()

cat(sprintf(
  "rhemtulla-2012: magmaan %s, %s, reps=%d, cells=%d%s\n",
  as.character(utils::packageVersion("magmaan")),
  R.version.string, args$reps, nrow(cell_grid),
  if (isTRUE(args$smoke)) " (smoke)" else ""))

all_params <- list(); all_chi2 <- list(); all_conv <- list()
all_parity <- vector("list", nrow(cell_grid))
pp <- 0L; cc <- 0L; vv <- 0L

t_global0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  taus <- symmetric_thresholds(cell$categories)
  cat(sprintf("  cell %2d/%2d: categories=%d N=%4d ",
              ci, nrow(cell_grid), cell$categories, cell$N))
  cat_ok <- 0L; ml_ok <- 0L
  t_cell0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    seed <- args$seed_base + ci * 10000L + rep_idx
    keep_parity <- isTRUE(args$lavaan_parity) && rep_idx == 1L
    out <- fit_one_rep(pop, taus, cell$N, seed, keep_parity = keep_parity)
    meta_cols <- data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                            categories = cell$categories, N = cell$N,
                            stringsAsFactors = FALSE)
    if (!is.null(out$params)) {
      pp <- pp + 1L; all_params[[pp]] <- cbind(out$params, meta_cols)
    }
    if (!is.null(out$chi2)) {
      cc <- cc + 1L; all_chi2[[cc]] <- cbind(out$chi2, meta_cols)
    }
    vv <- vv + 1L; all_conv[[vv]] <- cbind(out$conv, meta_cols)
    cat_ok <- cat_ok + isTRUE(out$conv$converged[out$conv$estimator == "catLS"])
    ml_ok <- ml_ok + isTRUE(out$conv$converged[out$conv$estimator == "ML"])
    if (keep_parity) all_parity[[ci]] <- out$parity
  }
  t_cell1 <- proc.time()[["elapsed"]]
  cat(sprintf("catLS_ok=%d ML_ok=%d  (%.1fs)\n", cat_ok, ml_ok,
              t_cell1 - t_cell0))
}
t_global1 <- proc.time()[["elapsed"]]

params_long <- if (length(all_params)) do.call(rbind, all_params) else NULL
chi2_long   <- if (length(all_chi2)) do.call(rbind, all_chi2) else NULL
conv_long   <- if (length(all_conv)) do.call(rbind, all_conv) else NULL

if (!is.null(params_long))
  utils::write.csv(params_long, file.path(results_dir, "fits_param.csv"),
                   row.names = FALSE)
if (!is.null(chi2_long))
  utils::write.csv(chi2_long, file.path(results_dir, "fits_chi2.csv"),
                   row.names = FALSE)
if (!is.null(conv_long))
  utils::write.csv(conv_long, file.path(results_dir, "fits_conv.csv"),
                   row.names = FALSE)

# ── Summaries ────────────────────────────────────────────────────────────────
# Parameter bias + coverage, aggregated by cell x estimator x true value.
summary_param <- if (!is.null(params_long)) {
  key <- with(params_long, paste(cell_idx, estimator, param_type, true,
                                 sep = "\r"))
  do.call(rbind, lapply(split(params_long, key), function(g) {
    cover <- mean(abs(g$est - g$true) <= 1.96 * g$se, na.rm = TRUE)
    data.frame(
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      N = g$N[[1L]], estimator = g$estimator[[1L]],
      param_type = g$param_type[[1L]], true = g$true[[1L]],
      n_reps = length(unique(g$rep)),
      mean_est = mean(g$est, na.rm = TRUE),
      rel_bias = (mean(g$est, na.rm = TRUE) - g$true[[1L]]) / g$true[[1L]],
      se_rel_bias = (mean(g$se, na.rm = TRUE) -
                       stats::sd(g$est, na.rm = TRUE)) /
        stats::sd(g$est, na.rm = TRUE),
      coverage = cover,
      stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_param))
  utils::write.csv(summary_param, file.path(results_dir, "summary_param.csv"),
                   row.names = FALSE)

# χ² rejection rate (Type I error; the fitted model is correctly specified).
summary_chi2 <- if (!is.null(chi2_long)) {
  key <- with(chi2_long, paste(cell_idx, estimator, stat_name, sep = "\r"))
  do.call(rbind, lapply(split(chi2_long, key), function(g) {
    p_vals <- stats::pchisq(g$statistic, df = g$df, lower.tail = FALSE)
    data.frame(
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      N = g$N[[1L]], estimator = g$estimator[[1L]],
      stat_name = g$stat_name[[1L]], n_reps = nrow(g),
      mean_chi2 = mean(g$statistic, na.rm = TRUE),
      mean_df = mean(g$df, na.rm = TRUE),
      reject_05 = mean(p_vals < 0.05, na.rm = TRUE),
      stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_chi2))
  utils::write.csv(summary_chi2, file.path(results_dir, "summary_chi2.csv"),
                   row.names = FALSE)

# Convergence / improper-solution rates by cell x estimator.
summary_conv <- if (!is.null(conv_long)) {
  key <- with(conv_long, paste(cell_idx, estimator, sep = "\r"))
  do.call(rbind, lapply(split(conv_long, key), function(g) {
    data.frame(
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      N = g$N[[1L]], estimator = g$estimator[[1L]],
      n_attempt = nrow(g),
      conv_rate = mean(g$converged),
      improper_rate = mean(g$improper[g$converged]),
      stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_conv))
  utils::write.csv(summary_conv, file.path(results_dir, "summary_conv.csv"),
                   row.names = FALSE)

# ── Metadata ─────────────────────────────────────────────────────────────────
meta <- metadata_frame(
  values = list(
    reps = args$reps,
    seed_base = args$seed_base,
    cells_filter = args$cells_filter %||% "",
    smoke = isTRUE(args$smoke),
    lavaan_parity = isTRUE(args$lavaan_parity),
    n_cells = nrow(cell_grid),
    n_param_rows = if (is.null(params_long)) 0L else nrow(params_long),
    n_chi2_rows = if (is.null(chi2_long)) 0L else nrow(chi2_long),
    total_seconds = sprintf("%.2f", t_global1 - t_global0)
  ),
  packages = "magmaan"
)
write_csv(meta, file.path(results_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1fs — wrote results to %s\n",
            t_global1 - t_global0, results_dir))

# ── Lavaan parity (rep 1 of each cell) ───────────────────────────────────────
if (isTRUE(args$lavaan_parity)) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    cat("--lavaan-parity requested but lavaan is not installed; skipping.\n")
  } else {
    cat("running lavaan parity check (one fit per cell)...\n")
    rows <- list()
    add <- function(ci, cell, estimator, metric, mag, lv) {
      rows[[length(rows) + 1L]] <<- data.frame(
        cell_idx = ci, categories = cell$categories, N = cell$N,
        estimator = estimator, metric = metric,
        magmaan = mag, lavaan = lv, stringsAsFactors = FALSE)
    }
    for (ci in seq_len(nrow(cell_grid))) {
      cell <- as.list(cell_grid[ci, , drop = FALSE])
      cached <- all_parity[[ci]]
      if (is.null(cached)) next
      df <- cached$data
      # cat-LS vs lavaan WLSMV (std loadings + scaled chi).
      if (!is.null(cached$catLS)) {
        lv <- tryCatch(lavaan::cfa(SYNTAX, data = df, ordered = VARNAMES,
                                   estimator = "WLSMV", std.lv = TRUE),
                       error = function(e) e)
        if (!inherits(lv, "error")) {
          sl <- lavaan::standardizedSolution(lv)
          lv_load <- sl$est.std[sl$op == "=~"]
          lv_fcor <- sl$est.std[sl$op == "~~" & sl$lhs == "f1" &
                                  sl$rhs == "f2"]
          tst <- lavaan::lavInspect(lv, "test")
          add(ci, cell, "catLS", "max_abs_load_diff",
              max(abs(cached$catLS$load_est - lv_load)), 0)
          add(ci, cell, "catLS", "fcor_diff",
              cached$catLS$fcor_est, lv_fcor)
          mv <- cached$catLS$chi2[cached$catLS$chi2$stat_name == "MV", ]
          add(ci, cell, "catLS", "chi2_MV_pvalue",
              stats::pchisq(mv$statistic, mv$df, lower.tail = FALSE),
              stats::pchisq(tst[[length(tst)]]$stat,
                            tst[[length(tst)]]$df, lower.tail = FALSE))
        }
      }
      # continuous ML vs lavaan MLM.
      if (!is.null(cached$ML)) {
        lv <- tryCatch(lavaan::sem(SYNTAX, data = df, estimator = "MLM",
                                   std.lv = TRUE),
                       error = function(e) e)
        if (!inherits(lv, "error")) {
          sl <- lavaan::standardizedSolution(lv)
          lv_load <- sl$est.std[sl$op == "=~"]
          add(ci, cell, "ML", "max_abs_load_diff",
              max(abs(cached$ML$load_est - lv_load)), 0)
        }
      }
    }
    if (length(rows)) {
      parity_df <- do.call(rbind, rows)
      parity_df$abs_diff <- abs(parity_df$magmaan - parity_df$lavaan)
      utils::write.csv(parity_df, file.path(results_dir, "lavaan_parity.csv"),
                       row.names = FALSE)
      cat(sprintf("lavaan parity: %d rows written\n", nrow(parity_df)))
    }
  }
}
