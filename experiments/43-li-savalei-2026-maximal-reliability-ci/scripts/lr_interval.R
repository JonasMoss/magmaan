#!/usr/bin/env Rscript
# Profile likelihood-ratio interval for bifactor maximal reliability, normal-theory.
# Every estimator-centric interval (Wald, delta, Cornish-Fisher) inherits rho-hat's
# bias / skew / boundary pileup. Test inversion sidesteps that: the CI is
# {rho0 : LR(rho0) <= q}, LR(rho0) = N (F_con(rho0) - F_unc), and coverage of the
# population value needs only LR at the TRUE value (rho*_true in the CI iff
# LR(rho*_true) <= q). The profile LR is transformation-invariant (no logit-vs-Fisher
# choice), range-respecting (the constrained fit stays in [0,1]), and asymmetric by
# construction. We compare three calibrations of the threshold:
#   chi2        q = qchisq(.95, 1) = 3.841
#   bartlett    q * c, c = E[LR] estimated by the cell mean of LR(rho*_true)
#               (the empirical/oracle Bartlett factor; df=1 so the factor is E[LR])
# and report the coverage of each plus the inflation c and the empirical 95th
# percentile of LR (which would be 3.841 if chi^2_1 held exactly). For IG
# non-normal observed indicators, the df=1 fixed-misspecification scale is
# estimated as lambda = Var_observed-bread-sandwich(rho*) /
# Var_observed-bread-quadratic(rho*), and the robust statistic is LR / lambda;
# the "robust Bartlett" factor is the cell mean of that scaled statistic. The
# older expected-bread ratio is retained only as a diagnostic.
#
# Usage:
#   Rscript scripts/lr_interval.R [--reps 400] [--ns 50,100,200]
#                                 [--dists normal,ig] [--ig-skew 1]
#                                 [--ig-exk 4] [--ig-family fleishman]
#                                 [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/lr_interval.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/lr_interval.R", mustWork = FALSE)
  dirname(dirname(sp))
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))
source(file.path(exp_dir, "R", "lr_ml.R"))

parse_args <- function(a) {
  o <- list(reps = 400L, ns = c(50L, 100L, 200L), dists = "normal",
            ig_skew = 1.0, ig_exk = 4.0, ig_family = "fleishman",
            seed_base = 20260701L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (startsWith(x, "--reps=")) o$reps <- as.integer(sub("^--reps=", "", x))
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (startsWith(x, "--ns=")) o$ns <- as.integer(strsplit(sub("^--ns=", "", x), ",")[[1L]])
    else if (x == "--dists") { i <- i + 1L; o$dists <- a[[i]] }
    else if (startsWith(x, "--dists=")) o$dists <- sub("^--dists=", "", x)
    else if (x == "--ig-skew") { i <- i + 1L; o$ig_skew <- as.numeric(a[[i]]) }
    else if (startsWith(x, "--ig-skew=")) o$ig_skew <- as.numeric(sub("^--ig-skew=", "", x))
    else if (x == "--ig-exk") { i <- i + 1L; o$ig_exk <- as.numeric(a[[i]]) }
    else if (startsWith(x, "--ig-exk=")) o$ig_exk <- as.numeric(sub("^--ig-exk=", "", x))
    else if (x == "--ig-family") { i <- i + 1L; o$ig_family <- a[[i]] }
    else if (startsWith(x, "--ig-family=")) o$ig_family <- sub("^--ig-family=", "", x)
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  o$dists <- parse_csv_arg(o$dists)
  bad <- setdiff(o$dists, c("normal", "chisq", "ig"))
  if (length(bad)) stop("unknown dist(s): ", paste(bad, collapse = ", "), call. = FALSE)
  if (!o$ig_family %in% c("tukey_gh", "johnson", "fleishman")) {
    stop("unknown IG generator family: ", o$ig_family, call. = FALSE)
  }
  if (o$smoke) { o$reps <- 80L; o$ns <- c(50L) }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan"); require_pkg("nloptr")
suppressPackageStartupMessages(library(magmaan))
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

pop <- bifactor_population(per = 6L, gen = 0.6, grp = c(0.5, 0.4, 0.45))   # p18
spec <- magmaan::model_spec(pop$syntax, orthogonal = TRUE, std_lv = TRUE)
ml <- make_lr_ml(pop)
q1 <- stats::qchisq(0.95, 1)
tgt <- c(gen = pop$rho_gen, grp = pop$rho_grp1)

# Data blocks for one cell. IG uses magmaan's independent-generator simulator
# directly on the observed indicators, preserving pop$Sigma while imposing common
# marginal skewness/excess-kurtosis targets.
cell_draws <- function(pop, n, dist, seed0, reps) {
  if (dist == "ig") {
    batch <- magmaan::magmaan_core$sim_ig_batch(
      pop$Sigma, rep(cfg$ig_skew, pop$p), rep(cfg$ig_exk, pop$p),
      n = n, reps = reps, seed_base = seed0,
      root = "cholesky", generator_family = cfg$ig_family)
    return(lapply(batch$draws, function(X) {
      X <- as.data.frame(X)
      names(X) <- pop$ov
      X
    }))
  }
  lapply(seq_len(reps), function(r) {
    set.seed(seed0 + r)
    gen_data(pop, n, dist)
  })
}

observed_bread_vcov <- function(fit) {
  info <- tryCatch(magmaan::magmaan_core$inference_information_observed_analytic(fit),
                   error = function(e) NULL)
  if (!is.matrix(info)) {
    info <- tryCatch(magmaan::magmaan_core$inference_information_observed_fd(fit),
                     error = function(e) NULL)
  }
  if (!is.matrix(info)) return(NULL)
  solve((info + t(info)) / 2)
}

# LR(rho*_true) and the df=1 robust scales for both coefficients on one dataset,
# or NULL. `lambda_misspec` uses the observed-bread profile quadratic in the
# denominator, matching the fixed-misspecification profile-LRT reference law.
lr_rep <- function(dat, n) {
  S <- stats::cov(dat) * (n - 1) / n
  logdetS <- as.numeric(determinant(S, logarithm = TRUE)$modulus)
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"), error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  pt <- fit$partable
  rebuild <- make_rebuilder(pt, pop)
  Vm <- tryCatch(stats::vcov(fit, regime = "model", data = dat),
                 error = function(e) NULL)
  Vr <- tryCatch(stats::vcov(fit, regime = "robust", data = dat),
                 error = function(e) NULL)
  Vo <- tryCatch(observed_bread_vcov(fit), error = function(e) NULL)
  lambda_expected <- c(gen = NA_real_, grp = NA_real_)
  lambda_misspec <- c(gen = NA_real_, grp = NA_real_)
  if (is.matrix(Vr)) {
    for (w in c("gen", "grp")) {
      g <- grad_rho(pt$est, pt, rebuild, pop, w)
      vr <- as.numeric(crossprod(g, Vr %*% g))
      if (is.matrix(Vm)) {
        vm <- as.numeric(crossprod(g, Vm %*% g))
        if (is.finite(vm) && vm > 0 && is.finite(vr) && vr > 0) {
          lambda_expected[[w]] <- vr / vm
        }
      }
      if (is.matrix(Vo)) {
        vo <- as.numeric(crossprod(g, Vo %*% g))
        if (is.finite(vo) && vo > 0 && is.finite(vr) && vr > 0) {
          lambda_misspec[[w]] <- vr / vo
        }
      }
    }
  }
  m <- rebuild(pt$est)
  if (any(m$P <= 0)) return(NULL)                          # same admissible subset
  z0 <- ml$zeta_from(m$L, m$P)
  unc <- ml$fit_unc(S, logdetS, z0)
  out <- list()
  for (w in c("gen", "grp")) {
    con <- ml$fit_con(S, logdetS, unc$z, w, tgt[[w]])
    if (!is.null(con)) {
      out[[length(out) + 1L]] <- data.frame(
        coef = w,
        LR = max(0, n * (con$F - unc$F)),
        lambda_expected = lambda_expected[[w]],
        lambda_misspec = lambda_misspec[[w]],
        stringsAsFactors = FALSE)
    }
  }
  if (!length(out)) return(NULL)
  do.call(rbind, out)
}

q95 <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  stats::quantile(x, 0.95, names = FALSE)
}

one_cell <- function(n, dist, seed0) {
  LR <- list(gen = data.frame(LR = numeric(0), lambda_expected = numeric(0),
                              lambda_misspec = numeric(0)),
             grp = data.frame(LR = numeric(0), lambda_expected = numeric(0),
                              lambda_misspec = numeric(0)))
  draws <- cell_draws(pop, n, dist, seed0, cfg$reps)
  for (r in seq_along(draws)) {
    v <- lr_rep(draws[[r]], n)
    if (is.null(v)) next
    for (w in c("gen", "grp")) {
      ww <- v[v$coef == w & is.finite(v$LR),
              c("LR", "lambda_expected", "lambda_misspec"), drop = FALSE]
      if (nrow(ww)) LR[[w]] <- rbind(LR[[w]], ww)
    }
  }
  do.call(rbind, lapply(c("gen", "grp"), function(w) {
    x <- LR[[w]]
    exp_ok <- is.finite(x$LR) & is.finite(x$lambda_expected) & x$lambda_expected > 0
    exp_scaled <- x$LR[exp_ok] / x$lambda_expected[exp_ok]
    sc_ok <- is.finite(x$LR) & is.finite(x$lambda_misspec) & x$lambda_misspec > 0
    scaled <- x$LR[sc_ok] / x$lambda_misspec[sc_ok]
    c_bart <- mean(x$LR)
    c_robust_bart <- mean(scaled)
    data.frame(p = pop$p, coef = w, dist = dist, n = n, reps_ok = nrow(x),
      reps_expected_scaled = sum(exp_ok), reps_scaled = sum(sc_ok),
      rho_pop = tgt[[w]], mean_LR = c_bart, q95_LR = q95(x$LR),
      mean_lambda_expected = mean(x$lambda_expected[is.finite(x$lambda_expected)]),
      mean_LR_scaled_expected = mean(exp_scaled),
      cov_robust_expected = mean(exp_scaled <= q1),
      mean_lambda = mean(x$lambda_misspec[is.finite(x$lambda_misspec)]),
      mean_LR_scaled = c_robust_bart,
      q95_LR_scaled = q95(scaled),
      cov_chi2 = mean(x$LR <= q1), cov_bartlett = mean(x$LR <= q1 * c_bart),
      cov_robust = mean(scaled <= q1),
      cov_robust_bartlett = mean(scaled <= q1 * c_robust_bart),
      stringsAsFactors = FALSE)
  }))
}

all_cells <- expand.grid(dist = cfg$dists, n = cfg$ns, stringsAsFactors = FALSE)
all_rows <- do.call(rbind, lapply(seq_len(nrow(all_cells)), function(i) {
  dist <- all_cells$dist[i]
  n <- all_cells$n[i]
  message(sprintf("cell %d/%d: dist=%s n=%d (reps=%d)",
                  i, nrow(all_cells), dist, n, cfg$reps))
  one_cell(n, dist, cfg$seed_base + i * 100000L)
}))

write_csv(all_rows, file.path(res_dir, "lr_coverage.csv"))
write_metadata(
  file.path(res_dir, "lr_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
    dists = paste(cfg$dists, collapse = ","), ig_skew = cfg$ig_skew,
    ig_exk = cfg$ig_exk, ig_family = cfg$ig_family, seed_base = cfg$seed_base,
    model = "orthogonal std.lv bifactor, 6 indicators/subscale (p18)",
    statistic = "profile LR = N (F_con(rho*_true) - F_unc), chi^2_1 reference",
    bartlett = "threshold q*c, c = cell-mean LR (empirical Bartlett factor, df=1)",
    robust = "lambda = Var_observed_bread_sandwich(rho*) / Var_observed_bread_quadratic(rho*); robust statistic LR/lambda",
    smoke = cfg$smoke),
  packages = "magmaan, nloptr")
cat("\nProfile-LR coverage (nominal 0.95):\n")
print(all_rows[order(all_rows$coef, all_rows$n),
               c("coef","dist","n","reps_ok","mean_LR","mean_lambda_expected",
                 "mean_lambda","mean_LR_scaled","cov_chi2","cov_bartlett",
                 "cov_robust_expected","cov_robust","cov_robust_bartlett")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "lr_coverage.csv"), "\n", sep = "")
