# Probe: can closed-form Guttman CFA make resampling-calibrated absolute GOF
# practical under non-normal data?
#
# The expensive analytic non-iterative GOF path builds J, M = I - Delta J, Gamma,
# and a weighted-chi-square spectrum. A bootstrap null reference only needs the
# raw discrepancy statistic on each draw:
#
#   T = N (vech(S) - vech(Sigma_hat))' V (vech(S) - vech(Sigma_hat)).
#
# So each bootstrap draw is just:
#   rebuild S_b -> closed-form map tau(S_b) -> direct T_b.
#
# This script compares:
#   chisq       naive central chi-square p-value,
#   nt_mix      analytic normal-theory weighted-mixture p-value,
#   emp_mix     analytic empirical-Gamma weighted-mixture p-value,
#   param_boot  Gaussian null bootstrap from Sigma_hat,
#   bstine      Bollen-Stine covariance-null row bootstrap.
#
# The Bollen-Stine transform recentres X and maps its ML covariance S to the
# fitted Sigma_hat, then row-resamples the transformed cases. It is the only
# resampling arm here that can preserve non-normal shape while imposing the
# covariance null.
#
# Pilot on 2026-07-09 (`--reps=200 --boot=99 --n=100 --p=9`):
# Bollen-Stine was near nominal under normal/t5 and a little high under skew
# (.035/.040/.075), with modest power (.195) against a residual covariance.
# It beat empirical-Gamma mixture on level/power balance in this tiny grid, but
# did not dominate NT/parametric bootstrap on power. Cost is the real hook:
# bootstrap draws use `T_only` (fit + direct statistic), which is close to the
# point-map cost and far below analytic empirical-Gamma spectrum cost.

suppressMessages(library(magmaan))
core <- magmaan::magmaan_core

parse_args <- function(args) {
  out <- list(reps = 120L, boot = 99L, seed = 20260709L, n = 100L,
              p = 9L, what = "both")
  for (arg in args) {
    if (grepl("^--reps=", arg)) out$reps <- as.integer(sub("^--reps=", "", arg))
    if (grepl("^--boot=", arg)) out$boot <- as.integer(sub("^--boot=", "", arg))
    if (grepl("^--seed=", arg)) out$seed <- as.integer(sub("^--seed=", "", arg))
    if (grepl("^--n=", arg)) out$n <- as.integer(sub("^--n=", "", arg))
    if (grepl("^--p=", arg)) out$p <- as.integer(sub("^--p=", "", arg))
    if (grepl("^--what=", arg)) out$what <- sub("^--what=", "", arg)
  }
  out
}

bench <- function(thunk, min_total = 0.25) {
  thunk()
  reps <- 2L
  repeat {
    t <- system.time(for (i in seq_len(reps)) thunk())[["elapsed"]]
    if (t >= min_total || reps > 2e4L) break
    reps <- reps * 2L
  }
  t / reps * 1000
}

cfa_syntax <- function(p) {
  sprintf("f =~ %s", paste0("x", seq_len(p), collapse = " + "))
}

draw_std <- function(n, dist) {
  if (identical(dist, "normal")) return(rnorm(n))
  if (identical(dist, "t5")) return(rt(n, df = 5) / sqrt(5 / 3))
  if (identical(dist, "chisq3")) {
    z <- rchisq(n, df = 3)
    return((z - 3) / sqrt(6))
  }
  stop("unknown dist: ", dist, call. = FALSE)
}

sim_onefactor <- function(n, p, load = 0.7, dist = "t5", rescov = 0) {
  stopifnot(p >= 3L)
  psi <- 1 - load^2
  if (rescov < 0 || rescov >= psi) {
    stop("rescov must be in [0, residual variance)", call. = FALSE)
  }
  f <- draw_std(n, dist)
  E <- matrix(draw_std(n * p, dist), n, p) * sqrt(psi)
  if (rescov > 0) {
    common <- draw_std(n, dist)
    E[, 1L] <- sqrt(psi - rescov) * draw_std(n, dist) + sqrt(rescov) * common
    E[, 2L] <- sqrt(psi - rescov) * draw_std(n, dist) + sqrt(rescov) * common
  }
  X <- load * f + E
  colnames(X) <- paste0("x", seq_len(p))
  X
}

ss_from_X <- function(proto, X) {
  proto$X <- list(X)
  proto$S <- list(cov(X) * (nrow(X) - 1L) / nrow(X))
  proto$mean <- list(colMeans(X))
  proto$nobs <- nrow(X)
  proto
}

fit_map <- function(pt, ss) {
  tryCatch(fit_noniterative_cfa(pt, ss, "guttman_gls_aligned", "standardized"),
           error = function(e) NULL)
}

sym_root <- function(S, inverse = FALSE, tol = 1e-10) {
  ee <- eigen((S + t(S)) / 2, symmetric = TRUE)
  vals <- pmax(ee$values, tol)
  scal <- if (inverse) 1 / sqrt(vals) else sqrt(vals)
  ee$vectors %*% diag(scal, length(scal)) %*% t(ee$vectors)
}

gof_uls_T <- function(fit, ss) {
  if (is.null(fit)) return(NA_real_)
  S <- ss$S[[1L]]
  Sig <- tryCatch(core$model_implied(fit)$sigma[[1L]], error = function(e) NULL)
  if (is.null(Sig)) return(NA_real_)
  R <- S - Sig
  n <- as.numeric(ss$nobs)[1L]
  n * (sum(diag(R)^2) + 2 * sum(R[lower.tri(R)]^2))
}

analytic_p <- function(fit, ss) {
  if (is.null(fit)) return(c(nt_mix = NA_real_, emp_mix = NA_real_))
  nt <- tryCatch(noniterative_cfa_inference(fit, "uls", "nt"),
                 error = function(e) NULL)
  emp <- tryCatch(noniterative_cfa_inference(fit, "uls", "empirical", ss),
                  error = function(e) NULL)
  c(nt_mix = if (is.null(nt)) NA_real_ else nt$p_mixture,
    emp_mix = if (is.null(emp)) NA_real_ else emp$p_mixture)
}

bstine_transform <- function(X, Sigma_hat) {
  Xc <- scale(X, center = TRUE, scale = FALSE)
  S <- crossprod(Xc) / nrow(Xc)
  Z <- Xc %*% sym_root(S, inverse = TRUE) %*% sym_root(Sigma_hat)
  colnames(Z) <- colnames(X)
  Z
}

boot_p <- function(Tobs, Tboot) {
  ok <- is.finite(Tboot)
  if (!is.finite(Tobs) || !any(ok)) return(NA_real_)
  (1 + sum(Tboot[ok] >= Tobs)) / (1 + sum(ok))
}

bootstrap_pvalues <- function(pt, proto, fit, ss, B) {
  Tobs <- gof_uls_T(fit, ss)
  if (is.null(fit) || !is.finite(Tobs)) {
    return(c(param_boot = NA_real_, bstine = NA_real_,
             param_ok = 0, bstine_ok = 0))
  }
  X <- ss$X[[1L]]
  n <- nrow(X)
  Sig <- core$model_implied(fit)$sigma[[1L]]
  Sig <- (Sig + t(Sig)) / 2
  Rchol <- tryCatch(chol(Sig), error = function(e) NULL)
  if (is.null(Rchol)) {
    return(c(param_boot = NA_real_, bstine = NA_real_,
             param_ok = 0, bstine_ok = 0))
  }
  Zbs <- bstine_transform(X, Sig)
  Tb_param <- rep(NA_real_, B)
  Tb_bs <- rep(NA_real_, B)
  for (b in seq_len(B)) {
    Xp <- matrix(rnorm(n * ncol(X)), n, ncol(X)) %*% Rchol
    colnames(Xp) <- colnames(X)
    ssp <- ss_from_X(proto, Xp)
    Tb_param[b] <- gof_uls_T(fit_map(pt, ssp), ssp)

    Xb <- Zbs[sample.int(n, n, replace = TRUE), , drop = FALSE]
    ssb <- ss_from_X(proto, Xb)
    Tb_bs[b] <- gof_uls_T(fit_map(pt, ssb), ssb)
  }
  c(param_boot = boot_p(Tobs, Tb_param),
    bstine = boot_p(Tobs, Tb_bs),
    param_ok = sum(is.finite(Tb_param)),
    bstine_ok = sum(is.finite(Tb_bs)))
}

one_rep <- function(seed, n, p, dist, rescov, B) {
  set.seed(seed)
  X <- sim_onefactor(n, p, dist = dist, rescov = rescov)
  dat <- as.data.frame(X)
  spec <- model_spec(cfa_syntax(p))
  proto <- df_to_data(dat, spec, missing = "error")
  ss <- ss_from_X(proto, X)
  fit <- fit_map(spec$partable, ss)
  if (is.null(fit)) return(NULL)
  Tobs <- gof_uls_T(fit, ss)
  df <- p * (p + 1L) / 2L - fit$npar
  ap <- analytic_p(fit, ss)
  bp <- bootstrap_pvalues(spec$partable, proto, fit, ss, B)
  c(chisq = stats::pchisq(Tobs, df = df, lower.tail = FALSE),
    ap, bp)
}

summarize_cell <- function(res) {
  res <- as.data.frame(do.call(rbind, res))
  pcols <- c("chisq", "nt_mix", "emp_mix", "param_boot", "bstine")
  rej <- vapply(res[pcols], function(x) mean(x <= 0.05, na.rm = TRUE), numeric(1))
  med <- vapply(res[pcols], function(x) median(x, na.rm = TRUE), numeric(1))
  ok <- c(param_ok = mean(res$param_ok, na.rm = TRUE),
          bstine_ok = mean(res$bstine_ok, na.rm = TRUE))
  list(reject = rej, median = med, ok = ok, n = nrow(res))
}

level_power_table <- function(reps, B, seed, n, p) {
  cells <- list(
    list(tag = "normal null", dist = "normal", rescov = 0),
    list(tag = "t5 null", dist = "t5", rescov = 0),
    list(tag = "chisq3 null", dist = "chisq3", rescov = 0),
    list(tag = "t5 rescov=.15", dist = "t5", rescov = 0.15)
  )
  ncores <- max(1L, parallel::detectCores() - 2L)
  cat(sprintf("reps=%d boot=%d n=%d p=%d cores=%d alpha=.05\n\n",
              reps, B, n, p, ncores))
  cat(sprintf("%-16s | %7s %7s %7s %7s %7s | %7s %7s\n",
              "cell", "chisq", "nt_mix", "emp", "param", "bstine",
              "p_ok", "bs_ok"))
  cat(strrep("-", 80), "\n")
  med_rows <- list()
  for (ci in seq_along(cells)) {
    cl <- cells[[ci]]
    res <- parallel::mclapply(seq_len(reps), function(i) {
      tryCatch(one_rep(seed + 10000L * ci + i, n, p, cl$dist, cl$rescov, B),
               error = function(e) NULL)
    }, mc.cores = ncores)
    sm <- summarize_cell(Filter(Negate(is.null), res))
    r <- sm$reject
    cat(sprintf("%-16s | %7.3f %7.3f %7.3f %7.3f %7.3f | %7.1f %7.1f   [%d]\n",
                cl$tag, r["chisq"], r["nt_mix"], r["emp_mix"],
                r["param_boot"], r["bstine"], sm$ok["param_ok"],
                sm$ok["bstine_ok"], sm$n))
    med_rows[[length(med_rows) + 1L]] <- c(cell = cl$tag, sm$median)
  }
  cat("\nrejection rate at alpha=.05. first three rows = LEVEL; last row = POWER.\n")
  cat("\nmedian p-values:\n")
  cat(sprintf("%-16s | %7s %7s %7s %7s %7s\n",
              "cell", "chisq", "nt_mix", "emp", "param", "bstine"))
  cat(strrep("-", 58), "\n")
  for (row in med_rows) {
    cat(sprintf("%-16s | %7.3f %7.3f %7.3f %7.3f %7.3f\n",
                row[["cell"]], as.numeric(row[["chisq"]]),
                as.numeric(row[["nt_mix"]]), as.numeric(row[["emp_mix"]]),
                as.numeric(row[["param_boot"]]), as.numeric(row[["bstine"]])))
  }
}

cost_table <- function(seed, n) {
  set.seed(seed)
  cat(sprintf("%-4s | %9s %10s %10s\n", "p", "point", "T_only", "analytic"))
  cat(strrep("-", 42), "\n")
  for (p in c(6L, 9L, 12L, 15L)) {
    X <- sim_onefactor(n, p, dist = "t5")
    dat <- as.data.frame(X)
    spec <- model_spec(cfa_syntax(p))
    proto <- df_to_data(dat, spec, missing = "error")
    ss <- ss_from_X(proto, X)
    point <- bench(function() fit_map(spec$partable, ss))
    tonly <- bench(function() {
      f <- fit_map(spec$partable, ss)
      gof_uls_T(f, ss)
    })
    analytic <- bench(function() {
      f <- fit_map(spec$partable, ss)
      noniterative_cfa_inference(f, "uls", "empirical", ss)
    }, min_total = 0.5)
    cat(sprintf("%-4d | %9.3f %10.3f %10.3f\n", p, point, tonly, analytic))
  }
  cat("\nms/call. T_only is fit + raw ULS GOF statistic; analytic is fit + empirical-Gamma mixture GOF.\n")
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (opts$what %in% c("cost", "both")) {
  cat("== Guttman GOF cost: T-only vs analytic spectrum ==\n")
  cost_table(opts$seed, opts$n)
  cat("\n")
}
if (opts$what %in% c("level", "both")) {
  cat("== Guttman GOF bootstrap calibration ==\n")
  level_power_table(opts$reps, opts$boot, opts$seed, opts$n, opts$p)
}
