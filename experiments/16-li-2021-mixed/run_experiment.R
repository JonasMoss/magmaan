#!/usr/bin/env Rscript
# Li (2021) replication — SEM with a mixture of continuous and categorical
# observed variables.
#
# Cheng-Hsien Li (2021), "Statistical estimation of structural equation models
# with a mixture of continuous and categorical observed variables", Behavior
# Research Methods 53:2191-2213. The defining feature versus Rhemtulla et al.
# (experiment 15) is that continuous and categorical indicators sit in the SAME
# model, so DWLS must combine polychoric (cat-cat), polyserial (cont-cat) and
# Pearson (cont-cont) associations in one weight matrix.
#
# Two population models, both with standardized (unit-variance) latent response
# variables and zero mean structure:
#
#   cfa — correlated three-factor measurement model (12 indicators). The
#         continuous:categorical ratio varies by factor (3:1, 2:2, 1:3), which
#         drives Li's "proportion of categorical" finding. Inter-factor
#         correlations all .3.
#   sem — five-factor structural model (20 indicators). Two exogenous, three
#         endogenous factors, each with a 2:2 continuous:categorical ratio.
#         Recovers structural paths (Gamma, Beta) as well as loadings.
#
# Two estimators are fit on the SAME generated data per replicate:
#
#   DWLS — categoricals as categorical (mixed polychoric/polyserial/Pearson),
#          robust mean-and-variance-adjusted chi-square (T_MV) and SEs. lavaan
#          calls this WLSMV.
#   MLR  — normal-theory ML treating the integer codes as continuous, with
#          Satorra-Bentler / mean-variance-adjusted chi-square and robust
#          (sandwich) SEs.
#
# The fitted model is correctly specified, so chi-square rejection rates are
# Type I error rates. Parameter estimates and SEs are reported in the
# standardized metric, split by parameter family (continuous loadings,
# categorical loadings, inter-factor correlations, structural paths).
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--model cfa|sem|both] [--cells FILTER]
#                            [--seed-base S] [--smoke] [--lavaan-parity]
#
# `--cells FILTER` is a comma-separated list of `key=value` pairs over
# `categories` and `N` (e.g. `categories=2,N=200`) restricting the grid. Use
# `--model` to subset the population models. Each `key` may list `|`-separated
# values (e.g. `categories=2|5|7`).

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
  out <- list(reps = 10L, model = "both", cells_filter = NULL,
              seed_base = 20260530L, smoke = FALSE, lavaan_parity = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--model cfa|sem|both] ",
          "[--cells FILTER] [--seed-base S] [--smoke] [--lavaan-parity]\n",
          sep = "")
      cat("  --model: which population model(s) to run (default both)\n")
      cat("  --cells: comma-separated key=value over `categories`,`N` ",
          "(e.g. categories=2|5,N=200)\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--model") {
      i <- i + 1L; out$model <- args[[i]]
    } else if (startsWith(a, "--model=")) {
      out$model <- sub("^--model=", "", a)
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
  if (!(out$model %in% c("cfa", "sem", "both"))) {
    stop("--model must be one of cfa, sem, both", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

project_root <- repo_root()
results_dir <- ensure_results_dir()

set_single_threaded_math()

require_pkg("magmaan")
core <- magmaan::magmaan_core

# ── Thresholds (Li 2021, pp. 2198-2199; hardcoded, exact) ────────────────────
# K-1 cutpoints on a standard normal. `discretize()` codes categories 1..K.
LI_THRESHOLDS <- list(
  sym = list(
    `2` = 0,
    `3` = c(-0.84, 0.84),
    `4` = c(-1.282, 0, 1.282),
    `5` = c(-1.282, -0.524, 0.524, 1.282),
    `6` = c(-1.645, -0.806, 0, 0.806, 1.645),
    `7` = c(-1.645, -0.954, -0.385, 0.385, 0.954, 1.645)),
  asym = list(
    `2` = -0.553,
    `3` = c(-1.282, -0.202),
    `4` = c(-1.645, -1.08, 0.412),
    `5` = c(-1.751, -1.341, -0.524, 0.706),
    `6` = c(-1.751, -1.341, -1.08, 0, 0.878),
    `7` = c(-1.751, -1.341, -1.036, -0.613, 0.496, 1.341)))

li_thresholds <- function(shape, K) {
  ts <- LI_THRESHOLDS[[shape]][[as.character(K)]]
  if (is.null(ts)) stop("no Li thresholds for shape=", shape, " K=", K)
  ts
}

discretize <- function(y_cont, taus) {
  matrix(findInterval(y_cont, taus, left.open = TRUE) + 1L,
         nrow = nrow(y_cont), ncol = ncol(y_cont))
}

# ── Population model: correlated three-factor CFA (Li Fig 1a) ─────────────────
# Continuous loadings x1..x6 = .8 .7 .6 .8 .7 .7; categorical loadings
# y1..y6 = .7 .8 .7 .8 .7 .6 (Li "Population parameters"). Indicators are listed
# continuous-then-categorical within each factor.
build_cfa <- function() {
  ind <- rbind(
    data.frame(name = "x1", factor = "f1", lambda = 0.8, is_cat = FALSE),
    data.frame(name = "x2", factor = "f1", lambda = 0.7, is_cat = FALSE),
    data.frame(name = "x3", factor = "f1", lambda = 0.6, is_cat = FALSE),
    data.frame(name = "y1", factor = "f1", lambda = 0.7, is_cat = TRUE),
    data.frame(name = "x4", factor = "f2", lambda = 0.8, is_cat = FALSE),
    data.frame(name = "x5", factor = "f2", lambda = 0.7, is_cat = FALSE),
    data.frame(name = "y2", factor = "f2", lambda = 0.8, is_cat = TRUE),
    data.frame(name = "y3", factor = "f2", lambda = 0.7, is_cat = TRUE),
    data.frame(name = "x6", factor = "f3", lambda = 0.7, is_cat = FALSE),
    data.frame(name = "y4", factor = "f3", lambda = 0.8, is_cat = TRUE),
    data.frame(name = "y5", factor = "f3", lambda = 0.7, is_cat = TRUE),
    data.frame(name = "y6", factor = "f3", lambda = 0.6, is_cat = TRUE),
    stringsAsFactors = FALSE)
  factors <- c("f1", "f2", "f3")
  Lambda <- matrix(0.0, nrow(ind), length(factors),
                   dimnames = list(ind$name, factors))
  for (i in seq_len(nrow(ind))) Lambda[i, ind$factor[[i]]] <- ind$lambda[[i]]
  Phi <- matrix(0.3, length(factors), length(factors)); diag(Phi) <- 1
  comm <- diag(Lambda %*% Phi %*% t(Lambda))
  Theta <- diag(1 - comm, nrow(ind))
  Sigma <- Lambda %*% Phi %*% t(Lambda) + Theta

  syntax <- paste(
    "f1 =~ x1 + x2 + x3 + y1",
    "f2 =~ x4 + x5 + y2 + y3",
    "f3 =~ x6 + y4 + y5 + y6",
    "f1 ~~ f2", "f1 ~~ f3", "f2 ~~ f3", sep = "\n")

  # Long-format population truth for the parameter families.
  truth_load <- data.frame(
    param_type = ifelse(ind$is_cat, "loading_cat", "loading_cont"),
    param_name = ind$name, true = ind$lambda, stringsAsFactors = FALSE)
  truth_fcor <- data.frame(
    param_type = "factor_cor",
    param_name = c("f1~~f2", "f1~~f3", "f2~~f3"), true = 0.3,
    stringsAsFactors = FALSE)

  list(name = "cfa", ind = ind, factors = factors,
       Sigma = Sigma, L = chol(Sigma), syntax = syntax,
       cat_names = ind$name[ind$is_cat], cont_names = ind$name[!ind$is_cat],
       varnames = ind$name,
       truth = rbind(truth_load, truth_fcor),
       exog_factors = factors)  # all factors exogenous in a CFA
}

# ── Population model: five-factor SEM (Li Fig 1b / text matrices) ─────────────
# Each factor: continuous loadings .8/.6, categorical loadings .8/.6. Latent
# order (f1,f2,f3,f4,f5) = (xi1, xi2, eta1, eta2, eta3). Structural matrices
# from Li's text (authoritative over the figure where they differ):
#   B = [[0,0,0],[.3,0,0],[.2,.5,0]]   among endogenous eta
#   Gamma = [[.4,.6],[.4,.2],[.1,.1]]  xi -> eta
#   Phi = [[1,.3],[.3,1]] , Psi = diag(.336,.436,.379)
build_sem <- function() {
  fac <- c("f1", "f2", "f3", "f4", "f5")
  exog <- c("f1", "f2"); endo <- c("f3", "f4", "f5")
  # Indicators: cont, cont, cat, cat per factor; loadings .8,.6,.8,.6.
  ind <- do.call(rbind, lapply(seq_along(fac), function(k) {
    base <- (k - 1L) * 2L
    data.frame(
      name = c(sprintf("x%d", base + 1L), sprintf("x%d", base + 2L),
               sprintf("y%d", base + 1L), sprintf("y%d", base + 2L)),
      factor = fac[[k]], lambda = c(0.8, 0.6, 0.8, 0.6),
      is_cat = c(FALSE, FALSE, TRUE, TRUE), stringsAsFactors = FALSE)
  }))

  Phi <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
  B <- matrix(c(0,   0,   0,
                0.3, 0,   0,
                0.2, 0.5, 0), 3, 3, byrow = TRUE)
  Gamma <- matrix(c(0.4, 0.6,
                    0.4, 0.2,
                    0.1, 0.1), 3, 2, byrow = TRUE)
  Psi <- diag(c(0.336, 0.436, 0.379))
  ImB_inv <- solve(diag(3) - B)
  Cov_eta <- ImB_inv %*% (Gamma %*% Phi %*% t(Gamma) + Psi) %*% t(ImB_inv)
  Cov_eta_xi <- ImB_inv %*% Gamma %*% Phi          # 3x2  Cov(eta, xi)
  Latent <- rbind(
    cbind(Phi, t(Cov_eta_xi)),
    cbind(Cov_eta_xi, Cov_eta))
  dimnames(Latent) <- list(fac, fac)
  # Endogenous latent variances should be ~1 under Li's Psi; warn if rounding
  # pushes them off, but keep unit observed variances via the communalities.
  endo_var <- diag(Cov_eta)
  if (any(abs(endo_var - 1) > 0.02)) {
    warning(sprintf("SEM endogenous latent variances off unity: %s",
                    paste(sprintf("%.3f", endo_var), collapse = ", ")))
  }

  Lambda <- matrix(0.0, nrow(ind), length(fac),
                   dimnames = list(ind$name, fac))
  for (i in seq_len(nrow(ind))) Lambda[i, ind$factor[[i]]] <- ind$lambda[[i]]
  comm <- diag(Lambda %*% Latent %*% t(Lambda))
  Theta <- diag(1 - comm, nrow(ind))
  Sigma <- Lambda %*% Latent %*% t(Lambda) + Theta

  ord <- function(k) paste(ind$name[ind$factor == fac[[k]]], collapse = " + ")
  syntax <- paste(
    sprintf("f1 =~ %s", ord(1)), sprintf("f2 =~ %s", ord(2)),
    sprintf("f3 =~ %s", ord(3)), sprintf("f4 =~ %s", ord(4)),
    sprintf("f5 =~ %s", ord(5)),
    "f3 ~ f1 + f2",            # eta1 ~ xi1 + xi2          (g11, g12)
    "f4 ~ f1 + f2 + f3",       # eta2 ~ xi1 + xi2 + eta1   (g21, g22, b21)
    "f5 ~ f1 + f2 + f3 + f4",  # eta3 ~ xi + eta           (g31, g32, b31, b32)
    "f1 ~~ f2", sep = "\n")

  # Population truth for structural paths, keyed by "lhs~rhs".
  paths <- rbind(
    data.frame(param_name = "f3~f1", true = 0.4, kind = "gamma"),
    data.frame(param_name = "f3~f2", true = 0.6, kind = "gamma"),
    data.frame(param_name = "f4~f1", true = 0.4, kind = "gamma"),
    data.frame(param_name = "f4~f2", true = 0.2, kind = "gamma"),
    data.frame(param_name = "f4~f3", true = 0.3, kind = "beta"),
    data.frame(param_name = "f5~f1", true = 0.1, kind = "gamma"),
    data.frame(param_name = "f5~f2", true = 0.1, kind = "gamma"),
    data.frame(param_name = "f5~f3", true = 0.2, kind = "beta"),
    data.frame(param_name = "f5~f4", true = 0.5, kind = "beta"),
    stringsAsFactors = FALSE)
  truth_load <- data.frame(
    param_type = ifelse(ind$is_cat, "loading_cat", "loading_cont"),
    param_name = ind$name, true = ind$lambda, stringsAsFactors = FALSE)
  truth_path <- data.frame(param_type = paths$kind,
                           param_name = paths$param_name, true = paths$true,
                           stringsAsFactors = FALSE)
  truth_fcor <- data.frame(param_type = "factor_cor",
                           param_name = "f1~~f2", true = 0.3,
                           stringsAsFactors = FALSE)

  list(name = "sem", ind = ind, factors = fac,
       Sigma = Sigma, L = chol(Sigma), syntax = syntax,
       cat_names = ind$name[ind$is_cat], cont_names = ind$name[!ind$is_cat],
       varnames = ind$name,
       truth = rbind(truth_load, truth_path, truth_fcor),
       exog_factors = exog)
}

# Build model specs once per population model. Mixed DWLS needs delta
# parameterization + meanstructure; both estimators use std.lv identification
# (exogenous factor variances fixed to 1), and we standardize the solution
# before comparing to Li's standardized population values.
attach_specs <- function(mod) {
  mod$spec_ord <- magmaan::model_spec(
    mod$syntax, ordered = mod$cat_names,
    parameterization = "delta", meanstructure = TRUE, std_lv = TRUE)
  mod$spec_cont <- magmaan::model_spec(mod$syntax, std_lv = TRUE)
  mod
}

# ── Sample-stats wrapper for the continuous ML path (mirrors exp 15) ─────────
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

# ── chi-square trace-moment adjustments for continuous ML (mirrors exp 15) ───
sb_from_moments <- function(T, mom) {
  df <- as.integer(mom$df %||% NA_integer_)
  tr <- as.numeric(mom$trace %||% NA_real_)
  if (!is.finite(T) || !is.finite(tr) || !is.finite(df) || df <= 0) {
    return(list(stat = NA_real_, df = NA_integer_,
                trace = tr, scale_c = NA_real_))
  }
  c_scale <- tr / df
  list(stat = if (c_scale > 0) T / c_scale else NA_real_, df = df,
       trace = tr, scale_c = c_scale)
}
mv_from_moments <- function(T, mom) {
  tr <- as.numeric(mom$trace %||% NA_real_)
  tr2 <- as.numeric(mom$trace_sq %||% NA_real_)
  if (!is.finite(T) || !is.finite(tr) || !is.finite(tr2) || tr2 <= 0) {
    return(list(stat = NA_real_, df = NA_real_, trace = tr,
                scale_c = NA_real_))
  }
  list(stat = T * tr / tr2, df = tr * tr / tr2, trace = tr,
       scale_c = NA_real_)
}

# ── Parameter extraction + sign orientation ──────────────────────────────────
# Build the long-format parameter rows for one estimator from a fitted partable
# and a standardized (theta, se) pair aligned to free-parameter indices. Under
# std.lv each factor's sign is unidentified; population loadings are all
# positive, so orient each factor's loadings positive and propagate the flip to
# factor correlations and structural paths.
extract_params <- function(estimator, pt, th_std, se_std, mod) {
  truth <- mod$truth
  factors <- mod$factors
  free_val <- function(idx, v) if (length(idx)) v[pt$free[idx]] else numeric(0)

  # Loadings (op "=~"): rhs = indicator, lhs = factor.
  li <- which(pt$op == "=~" & pt$free > 0L)
  load_name <- pt$rhs[li]; load_factor <- pt$lhs[li]
  load_est <- free_val(li, th_std); load_se <- free_val(li, se_std)

  # Per-factor sign flip from the loading estimates.
  flip <- setNames(rep(1, length(factors)), factors)
  for (f in factors) {
    idx <- which(load_factor == f)
    if (length(idx) && sum(load_est[idx], na.rm = TRUE) < 0) flip[[f]] <- -1
  }
  load_est <- load_est * flip[load_factor]

  rows <- list()
  # Loading rows carry their family (cont/cat) from the truth table.
  load_type <- truth$param_type[match(load_name, truth$param_name)]
  load_true <- truth$true[match(load_name, truth$param_name)]
  rows[["load"]] <- data.frame(
    estimator = estimator, param_type = load_type, param_name = load_name,
    true = load_true, est = load_est, se = load_se, stringsAsFactors = FALSE)

  # Factor correlations (op "~~", both sides factors, off-diagonal).
  ci <- which(pt$op == "~~" & pt$lhs %in% factors & pt$rhs %in% factors &
                pt$lhs != pt$rhs & pt$free > 0L)
  if (length(ci)) {
    nm <- paste0(pt$lhs[ci], "~~", pt$rhs[ci])
    est <- free_val(ci, th_std) * flip[pt$lhs[ci]] * flip[pt$rhs[ci]]
    se <- free_val(ci, se_std)
    tt <- truth$true[match(nm, truth$param_name)]
    rows[["fcor"]] <- data.frame(
      estimator = estimator, param_type = "factor_cor", param_name = nm,
      true = tt, est = est, se = se, stringsAsFactors = FALSE)
  }

  # Structural paths (op "~"): lhs ~ rhs, both factors.
  pi <- which(pt$op == "~" & pt$lhs %in% factors & pt$rhs %in% factors &
                pt$free > 0L)
  if (length(pi)) {
    nm <- paste0(pt$lhs[pi], "~", pt$rhs[pi])
    est <- free_val(pi, th_std) * flip[pt$lhs[pi]] * flip[pt$rhs[pi]]
    se <- free_val(pi, se_std)
    tt <- truth$true[match(nm, truth$param_name)]
    ty <- truth$param_type[match(nm, truth$param_name)]
    rows[["path"]] <- data.frame(
      estimator = estimator, param_type = ty, param_name = nm,
      true = tt, est = est, se = se, stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

# ── Per-replicate pipeline ───────────────────────────────────────────────────
# Generate one mixed matrix (continuous columns + integer-coded categorical
# columns) and fit both estimators on it. Each estimator contributes param
# rows, chi2 rows, and a convergence/improper record.
fit_one_rep <- function(mod, taus, N, seed, keep_parity = FALSE) {
  p <- nrow(mod$Sigma)
  cat_mask <- mod$varnames %in% mod$cat_names
  set.seed(seed)
  z <- matrix(stats::rnorm(N * p), N, p)
  y <- z %*% mod$L
  colnames(y) <- mod$varnames
  X <- y
  X[, cat_mask] <- discretize(y[, cat_mask, drop = FALSE], taus)
  # Data frame: ordered factors for categorical columns, numeric otherwise.
  df <- as.data.frame(lapply(seq_len(p), function(j) {
    if (cat_mask[[j]]) ordered(X[, j]) else as.numeric(X[, j])
  }))
  names(df) <- mod$varnames

  param_list <- list(); chi2_list <- list(); conv_list <- list()
  parity <- if (keep_parity) list(data = df) else NULL

  # ---- DWLS (mixed polychoric/polyserial/Pearson) ----
  dwls_ok <- FALSE; dwls_improper <- NA
  fit <- tryCatch({
    # DWLS only — skip the dense NACOV inverse (full-WLS weight); it is
    # O(m³) and often singular at small N with many indicators.
    d <- core$data_mixed_ordinal_stats_from_df(df, mod$spec_ord,
                                               full_wls_weight = FALSE)
    f <- core$fit_dwls_mixed_ordinal(
      mod$spec_ord, d, control = list(max_iter = 4000, ftol = 1e-13,
                                      gtol = 1e-8))
    list(d = d, f = f)
  }, error = function(e) e)
  if (!inherits(fit, "error") && isTRUE(fit$f$converged)) {
    f <- fit$f
    rob <- tryCatch(core$robust_mixed_ordinal(f, fit$d), error = function(e) e)
    # Standardized solution (lavaan std.all / standardizedSolution). magmaan's
    # ordinal-aware standardize_all puts categorical-indicator loadings on the
    # unit-variance latent-response scale and rescales structural paths and
    # endogenous-factor loadings by the model-implied latent SDs.
    vcov <- if (inherits(rob, "error")) NULL else rob$vcov
    sac <- if (is.null(vcov)) NULL else
      tryCatch(core$measures_standardize_all(f, vcov), error = function(e) NULL)
    th_std <- if (is.null(sac)) rep(NA_real_, length(f$theta)) else sac$theta
    se_std <- if (is.null(sac)) rep(NA_real_, length(f$theta)) else sac$se
    pr <- extract_params("DWLS", f$partable, th_std, se_std, mod)
    dwls_improper <- any(abs(pr$est[grepl("^loading", pr$param_type)]) > 1,
                         na.rm = TRUE)
    dwls_ok <- TRUE
    param_list[["DWLS"]] <- pr
    if (!inherits(rob, "error")) {
      chi2_list[["DWLS"]] <- data.frame(
        estimator = "DWLS", stat_name = c("SB", "MV"),
        statistic = c(rob$satorra_bentler$chi2_scaled,
                      rob$mean_var_adjusted$chi2_adj),
        df = c(rob$satorra_bentler$df, rob$mean_var_adjusted$df_adj),
        trace = NA_real_,
        scale_c = NA_real_,
        stringsAsFactors = FALSE)
    }
    if (keep_parity) parity$DWLS <- list(params = pr, chi2 = chi2_list[["DWLS"]])
  }
  conv_list[["DWLS"]] <- data.frame(estimator = "DWLS", converged = dwls_ok,
                                    improper = isTRUE(dwls_improper),
                                    stringsAsFactors = FALSE)

  # ---- MLR (continuous robust ML on the same matrix) ----
  ml_ok <- FALSE; ml_improper <- NA
  Xn <- X; storage.mode(Xn) <- "double"
  dat <- sample_data_from_matrix(Xn)
  ml <- tryCatch(core$fit_ml(mod$spec_cont, dat), error = function(e) e)
  if (!inherits(ml, "error") && isTRUE(ml$converged)) {
    Zc <- tryCatch(core$robust_casewise_contributions(ml$partable, Xn),
                   error = function(e) e)
    samp <- core$fit_sample_stats(ml)
    T_ML <- core$inference_chi2_stat(samp, ml$fmin)
    df_ML <- core$inference_df_stat(ml$partable, samp)
    pair <- if (inherits(Zc, "error")) NULL else tryCatch(
      core$robust_se_both_breads_zc(ml, Zc, nrow(Xn),
                                    moments = "structured", cov = "empirical"),
      error = function(e) NULL)
    tm <- if (inherits(Zc, "error")) NULL else tryCatch(
      core$robust_test_moments_both_breads_zc(ml, Zc, nrow(Xn),
                                              moments = "structured"),
      error = function(e) NULL)
    tm_unstruct <- if (inherits(Zc, "error")) NULL else tryCatch(
      core$robust_test_moments_both_breads_zc(ml, Zc, nrow(Xn),
                                              moments = "unstructured"),
      error = function(e) NULL)
    robust_vcov <- if (is.null(pair)) NULL else pair$expected$vcov
    sac <- tryCatch(core$measures_standardize_all(ml, robust_vcov %||% {
      infoE <- core$inference_information_expected(ml)
      core$inference_vcov_fit(infoE, ml)
    }), error = function(e) NULL)
    th_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$theta
    se_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$se
    pr <- extract_params("MLR", ml$partable, th_std, se_std, mod)
    ml_improper <- any(abs(pr$est[grepl("^loading", pr$param_type)]) > 1,
                       na.rm = TRUE)
    ml_ok <- TRUE
    param_list[["MLR"]] <- pr
    T_MV <- if (is.null(tm)) list(stat = NA_real_, df = NA_real_) else
      mv_from_moments(T_ML, tm$expected)
    T_SB <- if (is.null(tm)) list(stat = NA_real_, df = NA_integer_) else
      sb_from_moments(T_ML, tm$expected)
    T_SB_unstruct <- if (is.null(tm_unstruct)) {
      list(stat = NA_real_, df = NA_integer_, trace = NA_real_,
           scale_c = NA_real_)
    } else {
      sb_from_moments(T_ML, tm_unstruct$expected)
    }
    chi2_list[["MLR"]] <- data.frame(
      estimator = "MLR", stat_name = c("naive", "SB_struct", "SB", "MV"),
      statistic = c(T_ML, T_SB$stat, T_SB_unstruct$stat, T_MV$stat),
      df = c(df_ML, T_SB$df, T_SB_unstruct$df, T_MV$df),
      trace = c(NA_real_, T_SB$trace, T_SB_unstruct$trace, T_MV$trace),
      scale_c = c(NA_real_, T_SB$scale_c, T_SB_unstruct$scale_c,
                  T_MV$scale_c),
      stringsAsFactors = FALSE)
    if (keep_parity) parity$MLR <- list(params = pr, chi2 = chi2_list[["MLR"]])
  }
  conv_list[["MLR"]] <- data.frame(estimator = "MLR", converged = ml_ok,
                                   improper = isTRUE(ml_improper),
                                   stringsAsFactors = FALSE)

  list(
    params = if (length(param_list)) do.call(rbind, param_list) else NULL,
    chi2 = if (length(chi2_list)) do.call(rbind, chi2_list) else NULL,
    conv = do.call(rbind, conv_list),
    parity = if (keep_parity) parity else NULL)
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

models <- if (args$model == "both") c("cfa", "sem") else args$model
model_objs <- setNames(lapply(models, function(m)
  attach_specs(if (m == "cfa") build_cfa() else build_sem())), models)

cell_grid <- expand.grid(
  model = models, categories = 2:7, shape = c("sym", "asym"),
  N = c(200L, 500L, 1000L), stringsAsFactors = FALSE)
if (isTRUE(args$smoke)) {
  cell_grid <- cell_grid[cell_grid$categories %in% c(2L, 5L) &
                           cell_grid$shape == "sym" &
                           cell_grid$N %in% c(200L, 500L), , drop = FALSE]
}

filter <- parse_cells_filter(args$cells_filter)
for (key in names(filter)) {
  if (!(key %in% names(cell_grid))) {
    stop("--cells: unknown key '", key, "'", call. = FALSE)
  }
  vals <- strsplit(filter[[key]], "|", fixed = TRUE)[[1L]]
  if (is.numeric(cell_grid[[key]])) vals <- as.numeric(vals)
  cell_grid <- cell_grid[cell_grid[[key]] %in% vals, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)
cell_grid <- cell_grid[order(match(cell_grid$model, models), cell_grid$shape,
                             cell_grid$N, cell_grid$categories), , drop = FALSE]
rownames(cell_grid) <- NULL

# Threshold reference table (one row per shape x K actually used).
thr_keys <- unique(cell_grid[, c("shape", "categories")])
threshold_rows <- do.call(rbind, lapply(seq_len(nrow(thr_keys)), function(i) {
  sh <- thr_keys$shape[[i]]; K <- thr_keys$categories[[i]]
  taus <- li_thresholds(sh, K)
  props <- diff(stats::pnorm(c(-Inf, taus, Inf)))
  data.frame(shape = sh, categories = K,
             thresholds = paste(sprintf("%.3f", taus), collapse = ";"),
             proportions = paste(sprintf("%.3f", props), collapse = ";"),
             stringsAsFactors = FALSE)
}))
threshold_rows <- threshold_rows[order(threshold_rows$shape,
                                       threshold_rows$categories), ]
utils::write.csv(threshold_rows, file.path(results_dir, "thresholds.csv"),
                 row.names = FALSE)
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"),
                 row.names = FALSE)

cat(sprintf("li-2021-mixed: magmaan %s, %s, reps=%d, models={%s}, cells=%d%s\n",
            as.character(utils::packageVersion("magmaan")),
            R.version.string, args$reps, paste(models, collapse = ","),
            nrow(cell_grid), if (isTRUE(args$smoke)) " (smoke)" else ""))

all_params <- list(); all_chi2 <- list(); all_conv <- list()
all_parity <- vector("list", nrow(cell_grid))
pp <- 0L; cc <- 0L; vv <- 0L

t_global0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  mod <- model_objs[[cell$model]]
  taus <- li_thresholds(cell$shape, cell$categories)
  cat(sprintf("  cell %3d/%3d: model=%-3s cat=%d shape=%-4s N=%4d ",
              ci, nrow(cell_grid), cell$model, cell$categories, cell$shape,
              cell$N))
  dwls_ok <- 0L; ml_ok <- 0L
  t_cell0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    seed <- args$seed_base + ci * 10000L + rep_idx
    keep_parity <- isTRUE(args$lavaan_parity) && rep_idx == 1L
    out <- fit_one_rep(mod, taus, cell$N, seed, keep_parity = keep_parity)
    meta_cols <- data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                            model = cell$model, categories = cell$categories,
                            shape = cell$shape, N = cell$N,
                            stringsAsFactors = FALSE)
    if (!is.null(out$params)) {
      pp <- pp + 1L; all_params[[pp]] <- cbind(out$params, meta_cols)
    }
    if (!is.null(out$chi2)) {
      cc <- cc + 1L; all_chi2[[cc]] <- cbind(out$chi2, meta_cols)
    }
    vv <- vv + 1L; all_conv[[vv]] <- cbind(out$conv, meta_cols)
    dwls_ok <- dwls_ok +
      isTRUE(out$conv$converged[out$conv$estimator == "DWLS"])
    ml_ok <- ml_ok + isTRUE(out$conv$converged[out$conv$estimator == "MLR"])
    if (keep_parity) all_parity[[ci]] <- out$parity
  }
  t_cell1 <- proc.time()[["elapsed"]]
  cat(sprintf("DWLS_ok=%d MLR_ok=%d  (%.1fs)\n", dwls_ok, ml_ok,
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
# Parameter bias / SE bias / coverage aggregated by cell x estimator x family.
# Li collapses same-family parameters into one cell, so we aggregate across all
# parameters within a (cell, estimator, family) before computing relative bias.
summary_param <- if (!is.null(params_long)) {
  key <- with(params_long, paste(cell_idx, estimator, param_type, sep = "\r"))
  do.call(rbind, lapply(split(params_long, key), function(g) {
    # Relative bias / relative MSE averaged over the parameters in the family
    # (each parameter's bias is relative to its own true value).
    per <- split(g, g$param_name)
    rb <- vapply(per, function(h)
      (mean(h$est, na.rm = TRUE) - h$true[[1L]]) / h$true[[1L]], double(1))
    rmse <- vapply(per, function(h)
      mean(((h$est - h$true[[1L]]) / h$true[[1L]])^2, na.rm = TRUE), double(1))
    se_rb <- vapply(per, function(h) {
      sdt <- stats::sd(h$est, na.rm = TRUE)
      if (!is.finite(sdt) || sdt < 1e-12) return(NA_real_)
      (mean(h$se, na.rm = TRUE) - sdt) / sdt
    }, double(1))
    cover <- mean(abs(g$est - g$true) <= 1.96 * g$se, na.rm = TRUE)
    data.frame(
      cell_idx = g$cell_idx[[1L]], model = g$model[[1L]],
      categories = g$categories[[1L]], shape = g$shape[[1L]], N = g$N[[1L]],
      estimator = g$estimator[[1L]], param_type = g$param_type[[1L]],
      n_params = length(per), n_reps = length(unique(g$rep)),
      rel_bias = mean(rb, na.rm = TRUE),
      rmse_a = mean(rmse, na.rm = TRUE),
      se_rel_bias = mean(se_rb, na.rm = TRUE),
      coverage = cover, stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_param))
  utils::write.csv(summary_param, file.path(results_dir, "summary_param.csv"),
                   row.names = FALSE)

# chi-square rejection rate (Type I error; model is correctly specified).
summary_chi2 <- if (!is.null(chi2_long)) {
  key <- with(chi2_long, paste(cell_idx, estimator, stat_name, sep = "\r"))
  do.call(rbind, lapply(split(chi2_long, key), function(g) {
    p_vals <- stats::pchisq(g$statistic, df = g$df, lower.tail = FALSE)
    data.frame(
      cell_idx = g$cell_idx[[1L]], model = g$model[[1L]],
      categories = g$categories[[1L]], shape = g$shape[[1L]], N = g$N[[1L]],
      estimator = g$estimator[[1L]], stat_name = g$stat_name[[1L]],
      n_reps = nrow(g), mean_chi2 = mean(g$statistic, na.rm = TRUE),
      mean_df = mean(g$df, na.rm = TRUE),
      mean_trace = mean(g$trace, na.rm = TRUE),
      sd_trace = stats::sd(g$trace, na.rm = TRUE),
      mean_scale_c = mean(g$scale_c, na.rm = TRUE),
      sd_scale_c = stats::sd(g$scale_c, na.rm = TRUE),
      reject_05 = mean(p_vals < 0.05, na.rm = TRUE), stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_chi2))
  utils::write.csv(summary_chi2, file.path(results_dir, "summary_chi2.csv"),
                   row.names = FALSE)

summary_conv <- if (!is.null(conv_long)) {
  key <- with(conv_long, paste(cell_idx, estimator, sep = "\r"))
  do.call(rbind, lapply(split(conv_long, key), function(g) {
    data.frame(
      cell_idx = g$cell_idx[[1L]], model = g$model[[1L]],
      categories = g$categories[[1L]], shape = g$shape[[1L]], N = g$N[[1L]],
      estimator = g$estimator[[1L]], n_attempt = nrow(g),
      conv_rate = mean(g$converged),
      improper_rate = mean(g$improper[g$converged]), stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_conv))
  utils::write.csv(summary_conv, file.path(results_dir, "summary_conv.csv"),
                   row.names = FALSE)

# ── Metadata ─────────────────────────────────────────────────────────────────
meta <- metadata_frame(
  values = list(
    reps = args$reps, seed_base = args$seed_base,
    model = args$model, cells_filter = args$cells_filter %||% "",
    smoke = isTRUE(args$smoke), lavaan_parity = isTRUE(args$lavaan_parity),
    n_cells = nrow(cell_grid),
    n_param_rows = if (is.null(params_long)) 0L else nrow(params_long),
    n_chi2_rows = if (is.null(chi2_long)) 0L else nrow(chi2_long),
    total_seconds = sprintf("%.2f", t_global1 - t_global0)),
  packages = "magmaan")
write_csv(meta, file.path(results_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1fs — wrote results to %s\n",
            t_global1 - t_global0, results_dir))

# ── Lavaan parity (rep 1 of each cell) ───────────────────────────────────────
# DWLS vs lavaan WLSMV (standardized solution + scaled chi); MLR vs lavaan MLM
# (standardized solution). Compares loadings, structural paths, factor
# correlations on rep 1, plus the chi-square upper-tail p-value.
if (isTRUE(args$lavaan_parity)) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    cat("--lavaan-parity requested but lavaan is not installed; skipping.\n")
  } else {
    cat("running lavaan parity check (one fit per cell)...\n")
    rows <- list()
    add <- function(ci, cell, estimator, metric, mag, lv) {
      rows[[length(rows) + 1L]] <<- data.frame(
        cell_idx = ci, model = cell$model, categories = cell$categories,
        shape = cell$shape, N = cell$N, estimator = estimator, metric = metric,
        magmaan = mag, lavaan = lv, stringsAsFactors = FALSE)
    }
    # Compare the standardized estimate vectors of magmaan vs lavaan for a
    # parameter family, matched on param_name; reports the max abs diff.
    cmp_family <- function(ci, cell, estimator, pr, lv_std, types) {
      sub <- pr[pr$param_type %in% types, ]
      if (!nrow(sub)) return(invisible())
      lv_val <- lv_std$est.std[match(sub$param_name, lv_std$.name)]
      add(ci, cell, estimator, paste0("max_abs_diff_",
                                      paste(types, collapse = "+")),
          max(abs(sub$est - lv_val), na.rm = TRUE), 0)
    }
    # lavaan standardizedSolution rows tagged with a magmaan-style name.
    lv_named <- function(lv) {
      s <- lavaan::standardizedSolution(lv)
      sep <- ifelse(s$op == "=~", "", ifelse(s$op == "~~", "~~", "~"))
      s$.name <- ifelse(s$op == "=~", s$rhs, paste0(s$lhs, sep, s$rhs))
      s
    }
    for (ci in seq_len(nrow(cell_grid))) {
      cell <- as.list(cell_grid[ci, , drop = FALSE])
      cached <- all_parity[[ci]]
      if (is.null(cached)) next
      mod <- model_objs[[cell$model]]
      df <- cached$data
      load_types <- c("loading_cont", "loading_cat")
      path_types <- c("gamma", "beta")
      if (!is.null(cached$DWLS)) {
        lv <- tryCatch(lavaan::sem(mod$syntax, data = df,
                                   ordered = mod$cat_names, estimator = "WLSMV",
                                   std.lv = TRUE), error = function(e) e)
        if (!inherits(lv, "error")) {
          s <- lv_named(lv)
          cmp_family(ci, cell, "DWLS", cached$DWLS$params, s, load_types)
          if (cell$model == "sem")
            cmp_family(ci, cell, "DWLS", cached$DWLS$params, s, path_types)
          cmp_family(ci, cell, "DWLS", cached$DWLS$params, s, "factor_cor")
          tst <- lavaan::lavInspect(lv, "test")
          mv <- cached$DWLS$chi2[cached$DWLS$chi2$stat_name == "MV", ]
          add(ci, cell, "DWLS", "chi2_MV_pvalue",
              stats::pchisq(mv$statistic, mv$df, lower.tail = FALSE),
              stats::pchisq(tst[[length(tst)]]$stat, tst[[length(tst)]]$df,
                            lower.tail = FALSE))
        }
      }
      if (!is.null(cached$MLR)) {
        df_num <- as.data.frame(lapply(df, function(c)
          if (is.ordered(c)) as.numeric(as.character(c)) else as.numeric(c)))
        lv <- tryCatch(lavaan::sem(mod$syntax, data = df_num,
                                   estimator = "MLM", std.lv = TRUE),
                       error = function(e) e)
        if (!inherits(lv, "error")) {
          s <- lv_named(lv)
          cmp_family(ci, cell, "MLR", cached$MLR$params, s, load_types)
          if (cell$model == "sem")
            cmp_family(ci, cell, "MLR", cached$MLR$params, s, path_types)
          cmp_family(ci, cell, "MLR", cached$MLR$params, s, "factor_cor")
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
