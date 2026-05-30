#!/usr/bin/env Rscript
# Li (2016) replication — SEM with ordinal observed variables.
#
# Cheng-Hsien Li (2016), "The Performance of ML, DWLS, and ULS Estimation With
# Robust Corrections in Structural Equation Models With Ordinal Variables",
# Psychological Methods 21(3):369-387.
#
# Li's population model is the SAME five-factor structural model used in Li
# (2021) (experiment 16): two exogenous (xi1, xi2) and three endogenous
# (eta1, eta2, eta3) factors, structural matrices read off the paper's Figure 1
# and verified to give a standardized solution (unit latent variances):
#
#   B = [[0,0,0],[.3,0,0],[.2,.5,0]]   among endogenous eta
#   Gamma = [[.4,.6],[.4,.2],[.1,.1]]  xi -> eta
#   Phi = [[1,.3],[.3,1]] , Psi = diag(.336,.436,.379)
#
# The difference from experiment 16 is that here ALL 20 indicators are ordinal
# (4 per factor, standardized loadings .8/.7/.6/.5), and the design crosses
# three distribution shapes x four category counts x seven sample sizes (84
# cells) following Li's Figure 2 response-probability profiles.
#
# Li's three estimators are fit on the SAME categorized matrix per replicate:
#
#   DWLS — polychoric correlations, diagonal weight, robust
#          mean-and-variance-adjusted chi-square (T_MV) and SEs. lavaan WLSMV.
#   ULS  — polychoric correlations, identity weight, robust mean-and-variance
#          adjusted chi-square and SEs. lavaan ULSMV. DWLS and ULS share one
#          polychoric stats build per replicate (neither needs the full-WLS
#          NACOV inverse).
#   ML   — normal-theory ML treating the integer codes as continuous, with
#          Satorra-Bentler / mean-variance-adjusted chi-square and robust
#          (sandwich) SEs. lavaan MLM-family / Mplus MLR.
#
# The fitted model is correctly specified, so chi-square rejection rates are
# Type I error rates. Parameter estimates and SEs are reported in the
# standardized metric, split by family (loadings, inter-factor correlation,
# Gamma paths, Beta paths).
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--shape sym|slight|moderate|all]
#                            [--cells FILTER] [--seed-base S] [--smoke]
#                            [--lavaan-parity]
#
# `--cells FILTER` is a comma-separated list of `key=value` pairs over
# `categories` and `N` (e.g. `categories=4,N=500`) restricting the grid; each
# `key` may list `|`-separated values (e.g. `categories=4|7`). `--shape`
# subsets the distribution shapes.

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

SHAPES <- c("sym", "slight", "moderate")

parse_args <- function(args) {
  out <- list(reps = 10L, shape = "all", cells_filter = NULL,
              seed_base = 20260530L, smoke = FALSE, lavaan_parity = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] ",
          "[--shape sym|slight|moderate|all] [--cells FILTER] [--seed-base S] ",
          "[--smoke] [--lavaan-parity]\n", sep = "")
      cat("  --shape: which distribution shape(s) to run (default all)\n")
      cat("  --cells: comma-separated key=value over `categories`,`N` ",
          "(e.g. categories=4|7,N=500)\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") {
      i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) {
      out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--shape") {
      i <- i + 1L; out$shape <- args[[i]]
    } else if (startsWith(a, "--shape=")) {
      out$shape <- sub("^--shape=", "", a)
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
  if (!(out$shape %in% c(SHAPES, "all"))) {
    stop("--shape must be one of sym, slight, moderate, all", call. = FALSE)
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

project_root <- repo_root()
results_dir <- ensure_results_dir()

set_single_threaded_math()

require_pkg("magmaan")
core <- magmaan::magmaan_core

# ── Response-probability profiles (Li 2016, Fig 2; hardcoded, exact) ─────────
# Per-category response probabilities for each (shape, K). All 20 indicators in
# a cell share one profile, so one threshold vector per (shape, K). Thresholds
# are the standard-normal cutpoints tau_k = qnorm(cumsum(p))[1..K-1]; categories
# coded 1..K. The achieved skew/kurtosis are checked against Li's text below.
LI_PROBS <- list(
  sym = list(
    `4` = c(.10, .40, .40, .10),
    `5` = c(.10, .20, .40, .20, .10),
    `6` = c(.05, .16, .29, .29, .16, .05),
    `7` = c(.05, .12, .18, .30, .18, .12, .05)),
  slight = list(
    `4` = c(.05, .09, .52, .34),
    `5` = c(.04, .05, .21, .46, .24),
    `6` = c(.04, .05, .05, .36, .31, .19),
    `7` = c(.04, .05, .06, .12, .42, .22, .09)),
  moderate = list(
    `4` = c(.05, .09, .26, .60),
    `5` = c(.04, .06, .10, .32, .48),
    `6` = c(.04, .05, .06, .10, .33, .42),
    `7` = c(.04, .04, .05, .06, .10, .32, .39)))

# Li's reported marginal moments (text, p. 374): nominal targets per shape.
LI_MOMENT_REF <- list(
  sym      = c(skew =  0.000, kurt = -0.485),
  slight   = c(skew = -0.915, kurt =  0.820),
  moderate = c(skew = -1.385, kurt =  1.165))

li_thresholds <- function(shape, K) {
  p <- LI_PROBS[[shape]][[as.character(K)]]
  if (is.null(p)) stop("no Li probabilities for shape=", shape, " K=", K)
  stats::qnorm(cumsum(p)[-length(p)])
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

discretize <- function(y_cont, taus) {
  matrix(findInterval(y_cont, taus, left.open = TRUE) + 1L,
         nrow = nrow(y_cont), ncol = ncol(y_cont))
}

# ── Population model: all-ordinal five-factor SEM (Li 2016, Figure 1) ─────────
# Each factor has four ordinal indicators with standardized loadings
# .8/.7/.6/.5 (residual variances .36/.51/.64/.75). Latent order
# (f1,f2,f3,f4,f5) = (xi1, xi2, eta1, eta2, eta3). Structural matrices match
# experiment 16's build_sem() (read off the figure, verified standardized).
LOADINGS <- c(0.8, 0.7, 0.6, 0.5)

build_sem <- function() {
  fac <- c("f1", "f2", "f3", "f4", "f5")
  exog <- c("f1", "f2"); endo <- c("f3", "f4", "f5")
  ind <- do.call(rbind, lapply(seq_along(fac), function(k) {
    base <- (k - 1L) * 4L
    data.frame(
      name = sprintf("y%d", base + seq_len(4L)),
      factor = fac[[k]], lambda = LOADINGS, stringsAsFactors = FALSE)
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
    param_type = "loading", param_name = ind$name, true = ind$lambda,
    stringsAsFactors = FALSE)
  truth_path <- data.frame(param_type = paths$kind,
                           param_name = paths$param_name, true = paths$true,
                           stringsAsFactors = FALSE)
  truth_fcor <- data.frame(param_type = "factor_cor",
                           param_name = "f1~~f2", true = 0.3,
                           stringsAsFactors = FALSE)

  list(name = "sem", ind = ind, factors = fac,
       Sigma = Sigma, L = chol(Sigma), syntax = syntax,
       varnames = ind$name,
       truth = rbind(truth_load, truth_path, truth_fcor),
       exog_factors = exog)
}

# All-ordinal delta-parameterization spec (DWLS) and a continuous spec (ML);
# both std.lv (exogenous factor variances fixed to 1). The solution is
# standardized before comparing to Li's standardized population values.
attach_specs <- function(mod) {
  mod$spec_ord <- magmaan::model_spec(
    mod$syntax, ordered = mod$varnames,
    parameterization = "delta", std_lv = TRUE)
  mod$spec_cont <- magmaan::model_spec(mod$syntax, std_lv = TRUE)
  mod
}

# ── Sample-stats wrapper for the continuous ML path (mirrors exp 15/16) ──────
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

# ── chi-square trace-moment adjustments for continuous ML (mirrors exp 15/16) ─
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

# ── Parameter extraction + sign orientation (mirrors exp 16) ─────────────────
# Build long-format parameter rows for one estimator from a fitted partable and
# a standardized (theta, se) pair aligned to free-parameter indices. Under
# std.lv each factor's sign is unidentified; population loadings are all
# positive, so orient each factor's loadings positive and propagate the flip to
# the factor correlation and structural paths.
extract_params <- function(estimator, pt, th_std, se_std, mod) {
  truth <- mod$truth
  factors <- mod$factors
  free_val <- function(idx, v) if (length(idx)) v[pt$free[idx]] else numeric(0)

  li <- which(pt$op == "=~" & pt$free > 0L)
  load_name <- pt$rhs[li]; load_factor <- pt$lhs[li]
  load_est <- free_val(li, th_std); load_se <- free_val(li, se_std)

  flip <- setNames(rep(1, length(factors)), factors)
  for (f in factors) {
    idx <- which(load_factor == f)
    if (length(idx) && sum(load_est[idx], na.rm = TRUE) < 0) flip[[f]] <- -1
  }
  load_est <- load_est * flip[load_factor]

  rows <- list()
  load_true <- truth$true[match(load_name, truth$param_name)]
  rows[["load"]] <- data.frame(
    estimator = estimator, param_type = "loading", param_name = load_name,
    true = load_true, est = load_est, se = load_se, stringsAsFactors = FALSE)

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

# Categorical LS arm (DWLS or ULS): robust corrections + standardized solution
# + parameter extraction from a converged ordinal fit on shared stats `d`.
# Standardize_all puts categorical-indicator loadings on the unit-variance
# latent-response scale and rescales structural paths / endogenous-factor
# loadings by the model-implied latent SDs.
cat_arm_result <- function(estimator, f, d, mod) {
  rob <- tryCatch(core$robust_ordinal(f, d), error = function(e) e)
  vcov <- if (inherits(rob, "error")) NULL else rob$vcov
  sac <- if (is.null(vcov)) NULL else
    tryCatch(core$measures_standardize_all(f, vcov), error = function(e) NULL)
  th_std <- if (is.null(sac)) rep(NA_real_, length(f$theta)) else sac$theta
  se_std <- if (is.null(sac)) rep(NA_real_, length(f$theta)) else sac$se
  pr <- extract_params(estimator, f$partable, th_std, se_std, mod)
  chi2 <- if (!inherits(rob, "error")) data.frame(
    estimator = estimator, stat_name = c("SB", "MV"),
    statistic = c(rob$satorra_bentler$chi2_scaled, rob$mean_var_adjusted$chi2_adj),
    df = c(rob$satorra_bentler$df, rob$mean_var_adjusted$df_adj),
    stringsAsFactors = FALSE) else NULL
  list(params = pr, chi2 = chi2,
       improper = any(abs(pr$est[pr$param_type == "loading"]) > 1, na.rm = TRUE))
}

# ── Per-replicate pipeline ───────────────────────────────────────────────────
# Generate one categorized integer matrix and fit all three estimators on it.
# Each estimator contributes param rows, chi2 rows, and a convergence/improper
# record; an estimator that errors or fails to converge contributes only the
# conv record.
fit_one_rep <- function(mod, taus, N, seed, keep_parity = FALSE) {
  p <- nrow(mod$Sigma)
  set.seed(seed)
  z <- matrix(stats::rnorm(N * p), N, p)
  y <- z %*% mod$L
  colnames(y) <- mod$varnames
  X <- discretize(y, taus)
  colnames(X) <- mod$varnames
  df_ord <- as.data.frame(lapply(seq_len(p), function(j) ordered(X[, j])))
  names(df_ord) <- mod$varnames

  param_list <- list(); chi2_list <- list(); conv_list <- list()
  parity <- if (keep_parity) list(data = df_ord) else NULL

  # ---- Categorical LS arms: DWLS (diagonal weight) and ULS (identity weight) ----
  # Shared polychoric stats, computed once. full_wls_weight = FALSE skips the
  # dense NACOV inverse (O(m^3), singular at small N with 20 indicators); neither
  # DWLS nor ULS needs it — each uses NACOV + its own weight, and the robust
  # sandwich uses NACOV directly.
  cat_d <- tryCatch(core$data_ordinal_stats_from_df(df_ord, mod$spec_ord,
                                                    full_wls_weight = FALSE),
                    error = function(e) e)
  cat_arms <- list(list(name = "DWLS", fit = core$fit_dwls_ordinal),
                   list(name = "ULS",  fit = core$fit_uls_ordinal))
  for (arm in cat_arms) {
    ok <- FALSE; improper <- NA
    f <- if (inherits(cat_d, "error")) cat_d else tryCatch(
      arm$fit(mod$spec_ord, cat_d,
              control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8)),
      error = function(e) e)
    if (!inherits(f, "error") && isTRUE(f$converged)) {
      res <- cat_arm_result(arm$name, f, cat_d, mod)
      ok <- TRUE; improper <- res$improper
      param_list[[arm$name]] <- res$params
      if (!is.null(res$chi2)) chi2_list[[arm$name]] <- res$chi2
      if (keep_parity)
        parity[[arm$name]] <- list(params = res$params, chi2 = res$chi2)
    }
    conv_list[[arm$name]] <- data.frame(estimator = arm$name, converged = ok,
                                        improper = isTRUE(improper),
                                        stringsAsFactors = FALSE)
  }

  # ---- ML (continuous robust ML on the same integer matrix) ----
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
    robust_vcov <- if (is.null(pair)) NULL else pair$expected$vcov
    sac <- tryCatch(core$measures_standardize_all(ml, robust_vcov %||% {
      infoE <- core$inference_information_expected(ml)
      core$inference_vcov_fit(infoE, ml)
    }), error = function(e) NULL)
    th_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$theta
    se_std <- if (is.null(sac)) rep(NA_real_, length(ml$theta)) else sac$se
    pr <- extract_params("ML", ml$partable, th_std, se_std, mod)
    ml_improper <- any(abs(pr$est[pr$param_type == "loading"]) > 1,
                       na.rm = TRUE)
    ml_ok <- TRUE
    param_list[["ML"]] <- pr
    T_MV <- if (is.null(tm)) list(stat = NA_real_, df = NA_real_) else
      mv_from_moments(T_ML, tm$expected)
    T_SB <- if (is.null(tm)) list(stat = NA_real_, df = NA_integer_) else
      sb_from_moments(T_ML, tm$expected)
    chi2_list[["ML"]] <- data.frame(
      estimator = "ML", stat_name = c("naive", "SB", "MV"),
      statistic = c(T_ML, T_SB$stat, T_MV$stat),
      df = c(df_ML, T_SB$df, T_MV$df), stringsAsFactors = FALSE)
    if (keep_parity) parity$ML <- list(params = pr, chi2 = chi2_list[["ML"]])
  }
  conv_list[["ML"]] <- data.frame(estimator = "ML", converged = ml_ok,
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

shapes <- if (args$shape == "all") SHAPES else args$shape
mod <- attach_specs(build_sem())

cell_grid <- expand.grid(
  categories = 4:7, shape = shapes,
  N = c(200L, 300L, 400L, 500L, 750L, 1000L, 1500L),
  stringsAsFactors = FALSE)
if (isTRUE(args$smoke)) {
  cell_grid <- cell_grid[cell_grid$categories %in% c(4L, 7L) &
                           cell_grid$shape %in% c("sym", "moderate") &
                           cell_grid$N %in% c(200L, 1000L), , drop = FALSE]
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
cell_grid <- cell_grid[order(match(cell_grid$shape, SHAPES), cell_grid$N,
                             cell_grid$categories), , drop = FALSE]
rownames(cell_grid) <- NULL

# Threshold reference table (one row per shape x K), with the achieved marginal
# skew/kurtosis vs Li's reported nominal moments.
thr_keys <- unique(cell_grid[, c("shape", "categories")])
threshold_rows <- do.call(rbind, lapply(seq_len(nrow(thr_keys)), function(i) {
  sh <- thr_keys$shape[[i]]; K <- thr_keys$categories[[i]]
  taus <- li_thresholds(sh, K)
  props <- diff(stats::pnorm(c(-Inf, taus, Inf)))
  mom <- cat_moments(taus)
  ref <- LI_MOMENT_REF[[sh]]
  data.frame(shape = sh, categories = K,
             thresholds = paste(sprintf("%.3f", taus), collapse = ";"),
             proportions = paste(sprintf("%.3f", props), collapse = ";"),
             skew_achieved = mom[[1L]], kurt_achieved = mom[[2L]],
             skew_ref = unname(ref[["skew"]]), kurt_ref = unname(ref[["kurt"]]),
             stringsAsFactors = FALSE)
}))
threshold_rows <- threshold_rows[order(match(threshold_rows$shape, SHAPES),
                                       threshold_rows$categories), ]
utils::write.csv(threshold_rows, file.path(results_dir, "thresholds.csv"),
                 row.names = FALSE)
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"),
                 row.names = FALSE)

cat(sprintf("li-2016-ordinal: magmaan %s, %s, reps=%d, shapes={%s}, cells=%d%s\n",
            as.character(utils::packageVersion("magmaan")),
            R.version.string, args$reps, paste(shapes, collapse = ","),
            nrow(cell_grid), if (isTRUE(args$smoke)) " (smoke)" else ""))

all_params <- list(); all_chi2 <- list(); all_conv <- list()
all_parity <- vector("list", nrow(cell_grid))
pp <- 0L; cc <- 0L; vv <- 0L

t_global0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  taus <- li_thresholds(cell$shape, cell$categories)
  cat(sprintf("  cell %3d/%3d: cat=%d shape=%-8s N=%4d ",
              ci, nrow(cell_grid), cell$categories, cell$shape, cell$N))
  dwls_ok <- 0L; uls_ok <- 0L; ml_ok <- 0L
  t_cell0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    seed <- args$seed_base + ci * 10000L + rep_idx
    keep_parity <- isTRUE(args$lavaan_parity) && rep_idx == 1L
    out <- fit_one_rep(mod, taus, cell$N, seed, keep_parity = keep_parity)
    meta_cols <- data.frame(cell_idx = ci, rep = rep_idx, seed = seed,
                            categories = cell$categories, shape = cell$shape,
                            N = cell$N, stringsAsFactors = FALSE)
    if (!is.null(out$params)) {
      pp <- pp + 1L; all_params[[pp]] <- cbind(out$params, meta_cols)
    }
    if (!is.null(out$chi2)) {
      cc <- cc + 1L; all_chi2[[cc]] <- cbind(out$chi2, meta_cols)
    }
    vv <- vv + 1L; all_conv[[vv]] <- cbind(out$conv, meta_cols)
    dwls_ok <- dwls_ok +
      isTRUE(out$conv$converged[out$conv$estimator == "DWLS"])
    uls_ok <- uls_ok +
      isTRUE(out$conv$converged[out$conv$estimator == "ULS"])
    ml_ok <- ml_ok + isTRUE(out$conv$converged[out$conv$estimator == "ML"])
    if (keep_parity) all_parity[[ci]] <- out$parity
  }
  t_cell1 <- proc.time()[["elapsed"]]
  cat(sprintf("DWLS_ok=%d ULS_ok=%d ML_ok=%d  (%.1fs)\n", dwls_ok, uls_ok,
              ml_ok, t_cell1 - t_cell0))
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
# Li's evaluation criteria, aggregated by cell x estimator x family. Same-family
# parameters are collapsed into one cell, so we average each parameter's own
# relative bias / relative MSE / SE relative bias across the parameters in the
# family. (Li calls the relative-MSE criterion "RMSEA"; it is NOT the fit index.)
summary_param <- if (!is.null(params_long)) {
  key <- with(params_long, paste(cell_idx, estimator, param_type, sep = "\r"))
  do.call(rbind, lapply(split(params_long, key), function(g) {
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
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      shape = g$shape[[1L]], N = g$N[[1L]],
      estimator = g$estimator[[1L]], param_type = g$param_type[[1L]],
      n_params = length(per), n_reps = length(unique(g$rep)),
      rel_bias = mean(rb, na.rm = TRUE),
      rel_mse = mean(rmse, na.rm = TRUE),
      se_rel_bias = mean(se_rb, na.rm = TRUE),
      coverage = cover, stringsAsFactors = FALSE)
  }))
} else NULL
if (!is.null(summary_param))
  utils::write.csv(summary_param, file.path(results_dir, "summary_param.csv"),
                   row.names = FALSE)

# chi-square relative bias (over df) and rejection rate (Type I error; the
# fitted model is correctly specified).
summary_chi2 <- if (!is.null(chi2_long)) {
  key <- with(chi2_long, paste(cell_idx, estimator, stat_name, sep = "\r"))
  do.call(rbind, lapply(split(chi2_long, key), function(g) {
    p_vals <- stats::pchisq(g$statistic, df = g$df, lower.tail = FALSE)
    mean_chi2 <- mean(g$statistic, na.rm = TRUE)
    mean_df <- mean(g$df, na.rm = TRUE)
    data.frame(
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      shape = g$shape[[1L]], N = g$N[[1L]],
      estimator = g$estimator[[1L]], stat_name = g$stat_name[[1L]],
      n_reps = nrow(g), mean_chi2 = mean_chi2, mean_df = mean_df,
      chi2_rel_bias = if (is.finite(mean_df) && mean_df > 0)
        (mean_chi2 - mean_df) / mean_df else NA_real_,
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
      cell_idx = g$cell_idx[[1L]], categories = g$categories[[1L]],
      shape = g$shape[[1L]], N = g$N[[1L]],
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
    shape = args$shape, cells_filter = args$cells_filter %||% "",
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
# DWLS vs lavaan WLSMV (standardized solution + scaled chi); ML vs lavaan MLM
# (standardized solution). Compares loadings, structural paths, factor
# correlation on rep 1, plus the chi-square upper-tail p-value.
if (isTRUE(args$lavaan_parity)) {
  if (!requireNamespace("lavaan", quietly = TRUE)) {
    cat("--lavaan-parity requested but lavaan is not installed; skipping.\n")
  } else {
    cat("running lavaan parity check (one fit per cell)...\n")
    rows <- list()
    add <- function(ci, cell, estimator, metric, mag, lv) {
      rows[[length(rows) + 1L]] <<- data.frame(
        cell_idx = ci, categories = cell$categories, shape = cell$shape,
        N = cell$N, estimator = estimator, metric = metric,
        magmaan = mag, lavaan = lv, stringsAsFactors = FALSE)
    }
    cmp_family <- function(ci, cell, estimator, pr, lv_std, types) {
      sub <- pr[pr$param_type %in% types, ]
      if (!nrow(sub)) return(invisible())
      lv_val <- lv_std$est.std[match(sub$param_name, lv_std$.name)]
      add(ci, cell, estimator, paste0("max_abs_diff_",
                                      paste(types, collapse = "+")),
          max(abs(sub$est - lv_val), na.rm = TRUE), 0)
    }
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
      df <- cached$data
      path_types <- c("gamma", "beta")
      # DWLS vs lavaan WLSMV and ULS vs lavaan ULSMV (both polychoric LS).
      for (cl in list(list(arm = "DWLS", est = "WLSMV"),
                      list(arm = "ULS",  est = "ULSMV"))) {
        ca <- cached[[cl$arm]]
        if (is.null(ca)) next
        lv <- tryCatch(lavaan::sem(mod$syntax, data = df,
                                   ordered = mod$varnames, estimator = cl$est,
                                   std.lv = TRUE), error = function(e) e)
        if (!inherits(lv, "error")) {
          s <- lv_named(lv)
          cmp_family(ci, cell, cl$arm, ca$params, s, "loading")
          cmp_family(ci, cell, cl$arm, ca$params, s, path_types)
          cmp_family(ci, cell, cl$arm, ca$params, s, "factor_cor")
          tst <- lavaan::lavInspect(lv, "test")
          mv <- ca$chi2[ca$chi2$stat_name == "MV", ]
          add(ci, cell, cl$arm, "chi2_MV_pvalue",
              stats::pchisq(mv$statistic, mv$df, lower.tail = FALSE),
              stats::pchisq(tst[[length(tst)]]$stat, tst[[length(tst)]]$df,
                            lower.tail = FALSE))
        }
      }
      if (!is.null(cached$ML)) {
        df_num <- as.data.frame(lapply(df, function(c)
          as.numeric(as.character(c))))
        lv <- tryCatch(lavaan::sem(mod$syntax, data = df_num,
                                   estimator = "MLM", std.lv = TRUE),
                       error = function(e) e)
        if (!inherits(lv, "error")) {
          s <- lv_named(lv)
          cmp_family(ci, cell, "ML", cached$ML$params, s, "loading")
          cmp_family(ci, cell, "ML", cached$ML$params, s, path_types)
          cmp_family(ci, cell, "ML", cached$ML$params, s, "factor_cor")
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
