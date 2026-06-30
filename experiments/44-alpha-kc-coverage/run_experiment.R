#!/usr/bin/env Rscript
# Does the Kauermann & Carroll (2001) variance-of-the-variance correction fix the
# small-N undercoverage of the distribution-free (ADF) Wald interval for
# psychometric covariance functionals -- here Cronbach's alpha and the Pearson
# correlation -- versus a plain Wald interval?
#
# Any smooth covariance functional g(S) has a distribution-free (influence) SE,
#   se^2 = var(v)/n,   v_k = d_k' G d_k   (d_k = x_k - xbar,  G = dg/dSigma),
# i.e. v_k are the per-case influence values. Kauermann & Carroll prove the
# resulting Wald interval undercovers by an amount c_p * var(se^2)/se^4, and that
# var(se^2) is governed by the KURTOSIS of the influence values. The correction
# is an effective-df t reference,
#   f = 2 n / (kappa_hat - 1),   kappa_hat = mean(e^4)/mean(e^2)^2,  e = v - mean(v),
# the Satterthwaite realization of their quantile adjustment. The influence values
# are quadratic forms, so they are kurtotic even under normality; the ADF Wald
# interval therefore undercovers even on normal data, which is the cleanest
# demonstration. Two functionals share the machinery exactly:
#   alpha        = (p/(p-1)) (1 - tr(S)/1'S1)         (p = 6 congeneric)
#   correlation  = s12 / sqrt(s11 s22)                (p = 2, rho = 0.5)
# Both populations are tested under normal and contaminated-normal data with the
# functional's population value held fixed.

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
    "Coverage of the KC effective-df t interval vs the plain Wald (z) interval\n",
    "for Cronbach's alpha and the Pearson correlation, both built on the same\n",
    "distribution-free (influence) SE, across small N and two data laws.\n\n",
    "Options:\n",
    "  --reps N         Replications per cell. Default 2000 (smoke 300).\n",
    "  --ns A,B,..      Sample sizes. Default 20,30,50,100,200.\n",
    "  --seed-base S    Base RNG seed. Default 20260630.\n",
    "  --results-dir D  Output directory. Default results.\n",
    "  --smoke          Fast path: reps=300, ns=20,100.\n",
    "  --help           Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  o <- list(reps = 2000L, ns = c(20L, 30L, 50L, 100L, 200L),
            seed_base = 20260630L, results_dir = "results", smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help") { usage(); quit(save = "no", status = 0L) }
    else if (a == "--reps") { i <- i + 1L; o$reps <- as.integer(args[[i]]) }
    else if (a == "--ns") { i <- i + 1L; o$ns <- as.integer(parse_csv_arg(args[[i]])) }
    else if (a == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(args[[i]]) }
    else if (a == "--results-dir") { i <- i + 1L; o$results_dir <- args[[i]] }
    else if (a == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 300L; o$ns <- c(20L, 100L) }
  o
}

# --- the two functionals: point estimate g(S) and gradient G = dg/dSigma -------
# G is the symmetric matrix with tr(G H) = the directional derivative, so the
# influence value is v_k = d_k' G d_k. Off-diagonal entries carry the factor 1/2
# of the symmetric-perturbation convention (which is why alpha's all-ones J and
# correlation's 1/2 off-diagonal are correct).
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

ci_wald <- function(est, se, level = 0.95) {
  z <- stats::qnorm(1 - (1 - level) / 2); c(est - z * se, est + z * se)
}
ci_kc <- function(est, se, df, level = 0.95) {
  tq <- stats::qt(1 - (1 - level) / 2, df); c(est - tq * se, est + tq * se)
}

# --- data generation: both laws share the population covariance Sigma ----------
draw_mvn <- function(n, Sigma) {
  p <- nrow(Sigma); matrix(stats::rnorm(n * p), n, p) %*% chol(Sigma)
}
# Contaminated-normal scale mixture: X = sqrt(s) Z with s in {1, infl^2} at rate
# eps and Z ~ N(0, Sigma/E[s]); then Cov(X) = Sigma exactly, with finite moments
# (unlike multivariate-t, whose small-df influence kurtosis is infinite).
draw_contaminated <- function(n, Sigma, eps = 0.10, infl = 3) {
  Es <- (1 - eps) + eps * infl^2
  Z  <- draw_mvn(n, Sigma / Es)
  s  <- ifelse(stats::runif(n) < eps, infl^2, 1)
  Z * sqrt(s)
}
draw_data <- function(n, Sigma, law) {
  if (identical(law, "normal")) draw_mvn(n, Sigma) else draw_contaminated(n, Sigma)
}

congeneric_sigma <- function(lambda) {       # standardized one-factor population
  S <- tcrossprod(lambda); diag(S) <- 1; S
}

# --- run -----------------------------------------------------------------------
cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()

lambda <- c(0.80, 0.70, 0.75, 0.65, 0.70, 0.60)   # alpha population: congeneric p = 6
rho    <- 0.5                                     # correlation population: p = 2
quantities <- list(
  alpha = list(label = "Cronbach's alpha", Sigma = congeneric_sigma(lambda),
               point = alpha_point, grad = alpha_grad),
  correlation = list(label = "Pearson correlation",
                     Sigma = matrix(c(1, rho, rho, 1), 2L, 2L),
                     point = cor_point, grad = cor_grad))
laws  <- c("normal", "contaminated")
level <- 0.95

eval_cell <- function(q, n, law, reps, seed) {
  Sigma <- q$Sigma; true <- q$point(Sigma)
  set.seed(seed)
  est <- se <- df <- kap <- numeric(reps)
  hit_w <- hit_k <- lo_w <- hi_w <- lo_k <- hi_k <- wid_w <- wid_k <- numeric(reps)
  for (r in seq_len(reps)) {
    inf <- functional_inference(draw_data(n, Sigma, law), q$point, q$grad)
    est[r] <- inf$est; se[r] <- inf$se; df[r] <- inf$df; kap[r] <- inf$kappa
    cw <- ci_wald(inf$est, inf$se, level)
    ck <- ci_kc(inf$est, inf$se, inf$df, level)
    wid_w[r] <- cw[2] - cw[1]; wid_k[r] <- ck[2] - ck[1]
    hit_w[r] <- true >= cw[1] && true <= cw[2]
    hit_k[r] <- true >= ck[1] && true <= ck[2]
    lo_w[r] <- true < cw[1]; hi_w[r] <- true > cw[2]
    lo_k[r] <- true < ck[1]; hi_k[r] <- true > ck[2]
  }
  common <- list(quantity = names(quantities)[match(q$label, sapply(quantities, `[[`, "label"))],
                 n = n, law = law, reps = reps, true_value = true,
                 mean_est = mean(est), mc_sd = stats::sd(est), mean_se = mean(se),
                 mean_kappa = mean(kap))
  rbind(
    data.frame(c(common, list(method = "wald", coverage = mean(hit_w),
      mean_width = mean(wid_w), mean_df = NA_real_, miss_lower = mean(lo_w),
      miss_upper = mean(hi_w))), stringsAsFactors = FALSE),
    data.frame(c(common, list(method = "kc", coverage = mean(hit_k),
      mean_width = mean(wid_k), mean_df = mean(df), miss_lower = mean(lo_k),
      miss_upper = mean(hi_k))), stringsAsFactors = FALSE)
  )
}

res_dir <- ensure_results_dir()
rows <- list(); k <- 0L
for (qn in names(quantities)) for (law in laws) for (n in cfg$ns) {
  k <- k + 1L
  rows[[k]] <- eval_cell(quantities[[qn]], n, law, cfg$reps, cfg$seed_base + k * 100003L)
  cat(sprintf("[%-11s %-12s n=%4d] wald=%.3f  kc=%.3f  (df %.1f, kappa %.2f)\n",
              qn, law, n, rows[[k]]$coverage[1], rows[[k]]$coverage[2],
              rows[[k]]$mean_df[2], rows[[k]]$mean_kappa[1]))
}
out <- do.call(rbind, rows)
out <- out[order(match(out$quantity, names(quantities)), match(out$law, laws),
                 out$n, match(out$method, c("wald", "kc"))), ]

write_csv(out, file.path(res_dir, "coverage.csv"))
write_metadata(file.path(res_dir, "metadata.csv"),
  values = list(reps = cfg$reps, ns = cfg$ns, laws = laws, level = level,
                quantities = names(quantities), alpha_lambda = lambda, rho = rho,
                true_alpha = quantities$alpha$point(quantities$alpha$Sigma),
                true_correlation = rho, seed_base = cfg$seed_base),
  packages = character())

cat(sprintf("\nwrote %s (%d rows)\n", file.path(res_dir, "coverage.csv"), nrow(out)))
