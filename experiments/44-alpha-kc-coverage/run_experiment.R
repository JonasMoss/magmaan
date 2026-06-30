#!/usr/bin/env Rscript
# Two independent small-N defects of the distribution-free Wald interval for a
# covariance functional g(S), and the two corrections that compose to fix them:
#
#   (A) variance-of-the-variance: the ADF/influence SE se^2 = var(v)/n (v_k =
#       d_k' G d_k the per-case influence values) is itself a noisy estimate, so
#       the z interval undercovers. Kauermann & Carroll (2001): the undercoverage
#       is governed by the influence KURTOSIS; the fix is a t reference on
#       effective df f = 2n/(kappa_hat - 1), kappa_hat = mean(e^4)/mean(e^2)^2.
#   (B) skewness / boundedness: the raw estimate is asymmetric and range-bounded
#       (a correlation near +-1, a reliability near 1). A symmetric interval
#       cannot fix this; a variance-stabilizing transform does (Fisher's z for a
#       correlation, the logit for a reliability), with a delta-method SE on the
#       transformed scale and a back-transform of the endpoints.
#
# The 2x2 of corrections is {raw, transform} x {z, KC eff-df t}. We sweep the
# correlation over rho (skew grows with |rho|) so the two axes separate: the KC
# gap should be ~flat in rho (it is about kurtosis), the raw-scale gap should grow
# with rho (it is about skew), and the transform should flatten it. Cronbach's
# alpha (congeneric p=6, less boundary-pressed) rides along as the reliability
# companion with the logit transform. Both populations are tested under normal and
# contaminated-normal data with the functional's population value held fixed.

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

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Coverage of four intervals -- {raw, transform} x {Wald z, KC eff-df t} --\n",
    "for the Pearson correlation (swept over rho) and Cronbach's alpha, across\n",
    "small N and two data laws. Transform = Fisher z (correlation) / logit (alpha).\n\n",
    "Options:\n",
    "  --reps N         Replications per cell. Default 2000 (smoke 300).\n",
    "  --ns A,B,..      Sample sizes. Default 20,30,50,100,200.\n",
    "  --rhos A,B,..    Correlation values. Default 0,0.3,0.6,0.8,0.9.\n",
    "  --seed-base S    Base RNG seed. Default 20260630.\n",
    "  --results-dir D  Output directory. Default results.\n",
    "  --smoke          Fast path: reps=300, ns=20,100, rhos=0,0.9.\n",
    "  --help           Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  o <- list(reps = 2000L, ns = c(20L, 30L, 50L, 100L, 200L),
            rhos = c(0, 0.3, 0.6, 0.8, 0.9), seed_base = 20260630L,
            results_dir = "results", smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help") { usage(); quit(save = "no", status = 0L) }
    else if (a == "--reps") { i <- i + 1L; o$reps <- as.integer(args[[i]]) }
    else if (a == "--ns") { i <- i + 1L; o$ns <- as.integer(parse_csv_arg(args[[i]])) }
    else if (a == "--rhos") { i <- i + 1L; o$rhos <- parse_csv_numeric(args[[i]]) }
    else if (a == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(args[[i]]) }
    else if (a == "--results-dir") { i <- i + 1L; o$results_dir <- args[[i]] }
    else if (a == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 300L; o$ns <- c(20L, 100L); o$rhos <- c(0, 0.9) }
  o
}

# --- functionals: point estimate g(S) and gradient G = dg/dSigma ---------------
alpha_point <- function(S) {
  p <- nrow(S); (p / (p - 1)) * (1 - sum(diag(S)) / sum(S))
}
alpha_grad <- function(S) {                 # alpha = c(1 - a/b), a = tr S, b = 1'S1
  p <- nrow(S); cc <- p / (p - 1)
  a <- sum(diag(S)); b <- sum(S)
  (cc / b^2) * (a * matrix(1, p, p) - b * diag(p))
}
cor_point <- function(S) S[1, 2] / sqrt(S[1, 1] * S[2, 2])
cor_grad <- function(S) {                   # r = s12 / sqrt(s11 s22)
  r <- cor_point(S); s11 <- S[1, 1]; s22 <- S[2, 2]
  off <- 0.5 / sqrt(s11 * s22)
  matrix(c(-r / (2 * s11), off, off, -r / (2 * s22)), 2L, 2L)
}

# Variance-stabilizing transforms: link, inverse, and d link / d theta (for the
# delta-method SE on the transformed scale). Domain-clamped so a small-sample
# estimate outside (lo, hi) still yields finite endpoints.
logit_transform <- list(
  lo = 0, hi = 1,
  link  = function(x) stats::qlogis(pmin(pmax(x, 1e-6), 1 - 1e-6)),
  ilink = stats::plogis,
  dlink = function(x) { x <- pmin(pmax(x, 1e-6), 1 - 1e-6); 1 / (x * (1 - x)) })
fisher_transform <- list(
  lo = -1, hi = 1,
  link  = function(x) atanh(pmin(pmax(x, -1 + 1e-6), 1 - 1e-6)),
  ilink = tanh,
  dlink = function(x) { x <- pmin(pmax(x, -1 + 1e-6), 1 - 1e-6); 1 / (1 - x^2) })

# Point estimate, distribution-free (influence) SE, influence kurtosis, and the
# KC effective degrees of freedom from one n x p sample and a functional.
functional_inference <- function(X, point_fn, grad_fn) {
  n <- nrow(X)
  Xc <- sweep(X, 2L, colMeans(X))           # d_k = x_k - xbar
  S  <- crossprod(Xc) / (n - 1L)
  G  <- grad_fn(S)
  v  <- rowSums((Xc %*% G) * Xc)            # v_k = d_k' G d_k (the influence values)
  e  <- v - mean(v)
  m2 <- mean(e^2)
  se <- sqrt(stats::var(v) / n)             # ADF / influence-function SE
  kappa <- mean(e^4) / m2^2                 # influence kurtosis (3 under normality)
  f  <- if (is.finite(kappa) && kappa > 1) 2 * n / (kappa - 1) else n
  f  <- max(min(f, n), 1)                   # never sharper than n df, never below 1
  list(est = point_fn(S), se = se, kappa = kappa, df = f)
}

# Raw-scale and transformed-scale intervals, each with a normal (z) or KC (t_f)
# quantile. transform = NULL gives the raw interval.
make_ci <- function(est, se, qtl, transform) {
  if (is.null(transform)) return(c(est - qtl * se, est + qtl * se))
  l  <- transform$link(est); sl <- se * transform$dlink(est)
  c(transform$ilink(l - qtl * sl), transform$ilink(l + qtl * sl))
}

# --- data generation: both laws share the population covariance Sigma ----------
draw_mvn <- function(n, Sigma) {
  p <- nrow(Sigma); matrix(stats::rnorm(n * p), n, p) %*% chol(Sigma)
}
draw_contaminated <- function(n, Sigma, eps = 0.10, infl = 3) {
  Es <- (1 - eps) + eps * infl^2
  Z  <- draw_mvn(n, Sigma / Es)
  s  <- ifelse(stats::runif(n) < eps, infl^2, 1)
  Z * sqrt(s)
}
draw_data <- function(n, Sigma, law) {
  if (identical(law, "normal")) draw_mvn(n, Sigma) else draw_contaminated(n, Sigma)
}
congeneric_sigma <- function(lambda) { S <- tcrossprod(lambda); diag(S) <- 1; S }

# --- run -----------------------------------------------------------------------
cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()

lambda <- c(0.80, 0.70, 0.75, 0.65, 0.70, 0.60)   # alpha population: congeneric p = 6
designs <- list(list(quantity = "alpha", param = NA_real_,
                     Sigma = congeneric_sigma(lambda), point = alpha_point,
                     grad = alpha_grad, transform = logit_transform))
for (rho in cfg$rhos) {
  designs[[length(designs) + 1L]] <- list(
    quantity = "correlation", param = rho,
    Sigma = matrix(c(1, rho, rho, 1), 2L, 2L),
    point = cor_point, grad = cor_grad, transform = fisher_transform)
}
laws    <- c("normal", "contaminated")
methods <- c("wald", "kc", "twald", "tkc")        # raw/z, raw/t, transf/z, transf/t
level   <- 0.95

eval_cell <- function(des, n, law, reps, seed) {
  Sigma <- des$Sigma; true <- des$point(Sigma); tr <- des$transform
  set.seed(seed)
  est <- se <- df <- kap <- numeric(reps)
  hit <- mlo <- mhi <- wid <- matrix(0, reps, 4L, dimnames = list(NULL, methods))
  for (r in seq_len(reps)) {
    inf <- functional_inference(draw_data(n, Sigma, law), des$point, des$grad)
    est[r] <- inf$est; se[r] <- inf$se; df[r] <- inf$df; kap[r] <- inf$kappa
    z <- stats::qnorm(0.975); tf <- stats::qt(0.975, inf$df)
    ci <- list(wald  = make_ci(inf$est, inf$se, z,  NULL),
               kc    = make_ci(inf$est, inf$se, tf, NULL),
               twald = make_ci(inf$est, inf$se, z,  tr),
               tkc   = make_ci(inf$est, inf$se, tf, tr))
    for (mth in methods) {
      lo <- ci[[mth]][1]; hi <- ci[[mth]][2]
      hit[r, mth] <- is.finite(lo) && is.finite(hi) && true >= lo && true <= hi
      mlo[r, mth] <- is.finite(lo) && true < lo        # true below the interval
      mhi[r, mth] <- is.finite(hi) && true > hi        # true above the interval
      wid[r, mth] <- hi - lo
    }
  }
  common <- list(quantity = des$quantity, param = des$param, n = n, law = law,
                 reps = reps, true_value = true, mean_est = mean(est),
                 mc_sd = stats::sd(est), mean_se = mean(se),
                 mean_kappa = mean(kap), mean_df = mean(df))
  do.call(rbind, lapply(methods, function(mth)
    data.frame(c(common, list(method = mth, coverage = mean(hit[, mth]),
                              miss_lower = mean(mlo[, mth]), miss_upper = mean(mhi[, mth]),
                              mean_width = mean(wid[, mth]))),
               stringsAsFactors = FALSE)))
}

res_dir <- ensure_results_dir()
rows <- list(); k <- 0L
for (des in designs) for (law in laws) for (n in cfg$ns) {
  k <- k + 1L
  rows[[k]] <- eval_cell(des, n, law, cfg$reps, cfg$seed_base + k * 100003L)
  cv <- function(m) rows[[k]]$coverage[rows[[k]]$method == m]
  tag <- if (is.na(des$param)) "alpha" else sprintf("r=%.1f", des$param)
  cat(sprintf("[%-8s %-12s n=%4d] wald=%.3f kc=%.3f twald=%.3f tkc=%.3f (df %.0f)\n",
              tag, law, n, cv("wald"), cv("kc"), cv("twald"), cv("tkc"),
              rows[[k]]$mean_df[1]))
}
out <- do.call(rbind, rows)
out <- out[order(match(out$quantity, c("alpha", "correlation")), out$param,
                 match(out$law, laws), out$n, match(out$method, methods)), ]

write_csv(out, file.path(res_dir, "coverage.csv"))
write_metadata(file.path(res_dir, "metadata.csv"),
  values = list(reps = cfg$reps, ns = cfg$ns, rhos = cfg$rhos, laws = laws,
                level = level, methods = methods, alpha_lambda = lambda,
                true_alpha = alpha_point(congeneric_sigma(lambda)),
                seed_base = cfg$seed_base),
  packages = character())

cat(sprintf("\nwrote %s (%d rows)\n", file.path(res_dir, "coverage.csv"), nrow(out)))
