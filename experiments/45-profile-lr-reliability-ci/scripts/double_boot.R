#!/usr/bin/env Rscript
# Smoky smoke for the iterated (double) bootstrap correction of the small-sample
# Bartlett factor c = E[T] of the profile-LR reliability interval. Motivation:
# the one-level null-model bootstrap recovers a STABLE ~76% of the true c across
# N (see coverage.R). A stable ratio is the signature of a MULTIPLICATIVE bias:
# the plug-in estimates c(theta_tilde), but theta_tilde fitted at small N is
# "cleaner" (less near-boundary mass) than the truth, so c(theta_tilde) < c(truth)
# by a roughly constant factor rho. The double bootstrap ESTIMATES rho from the
# data by nesting one more level and correcting.
#
# Per data sample (bifactor rho*, the hard case):
#   theta_tilde = constrained MLE imposing g = g_pop on the real sample.
#   outer  (B1): simulate from theta_tilde, fit -> theta_tilde*_b, record T.
#                mean T = c_plugin = c(theta_tilde)   (== the one-level null-boot)
#   inner  (B2): from each theta_tilde*_b, simulate and average T -> c(theta_tilde*_b).
#                mean over b = Estar = E*[c(theta_tilde*)]
#   corrections:
#     additive        c_add  = 2*c_plugin - Estar
#     multiplicative  c_mult = c_plugin^2 / Estar   (= c_plugin / (Estar/c_plugin))
# Target = the oracle c(theta_pop) (read from results/lawley_<pop>.csv if present).
# If c_mult ~ oracle and c_plugin ~ 0.76*oracle, the multiplicative double
# bootstrap closes the gap.
#
# Usage:
#   Rscript scripts/double_boot.R [--pop bf] [--n 50] [--samples 15]
#            [--B1 40] [--B2 25] [--seed-base S] [--cores K] [--smoke]

.here <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  if (length(fa)) dirname(dirname(normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)))
  else normalizePath(".", mustWork = TRUE)
}
exp_dir <- .here()
source(file.path(dirname(exp_dir), "_support", "R", "helpers.R"))
source(file.path(exp_dir, "R", "engine.R"))
source(file.path(exp_dir, "R", "functionals.R"))

parse_args <- function(a) {
  o <- list(pop = "bf", n = 50L, samples = 15L, B1 = 40L, B2 = 25L,
            seed_base = 20260701L,
            cores = max(1L, min(5L, parallel::detectCores() - 1L)), smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat(readLines(file.path(exp_dir, "scripts", "double_boot.R"))[2:38], sep = "\n"); quit(save = "no") }
    else if (x == "--pop")     { i <- i + 1L; o$pop <- a[[i]] }
    else if (x == "--n")       { i <- i + 1L; o$n <- as.integer(a[[i]]) }
    else if (x == "--samples") { i <- i + 1L; o$samples <- as.integer(a[[i]]) }
    else if (x == "--B1")      { i <- i + 1L; o$B1 <- as.integer(a[[i]]) }
    else if (x == "--B2")      { i <- i + 1L; o$B2 <- as.integer(a[[i]]) }
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--cores")   { i <- i + 1L; o$cores <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$samples <- 8L; o$B1 <- 30L; o$B2 <- 20L }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("nloptr"); require_pkg("MASS"); require_pkg("parallel")
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

make_pop_bf <- function(per, k, gen, grp) {
  p <- per * k; items <- split(seq_len(p), rep(seq_len(k), each = per))
  Lam <- matrix(0, p, k + 1L); Lam[, 1] <- gen
  for (j in seq_len(k)) Lam[items[[j]], j + 1L] <- grp[j]
  psi <- 1 - rowSums(Lam^2)
  list(kind = "bf", p = p, cols = c(list(seq_len(p)), items), Lam = Lam, psi = psi,
       Sigma = tcrossprod(Lam) + diag(psi))
}
pops <- list(bf = make_pop_bf(5L, 3L, 0.60, c(0.50, 0.40, 0.45)))
pop <- pops[[cfg$pop]]; if (is.null(pop)) stop("unknown --pop: ", cfg$pop)
p <- pop$p; eng <- make_engine(pop$cols, p)
gfun <- make_H(eng, tcol = 1L)                        # bifactor maximal reliability rho*
z_true <- eng$zeta_from(pop$Lam, pop$psi); g_pop <- eng$gval(gfun, z_true)

# oracle c(theta_pop) reference, if a prior lawley run recorded it
oref <- file.path(res_dir, sprintf("lawley_%s.csv", cfg$pop))
oracle_c <- if (file.exists(oref)) {
  lw <- read.csv(oref); r <- lw[lw$func == "rho" & lw$n == cfg$n, ]
  if (nrow(r)) r$oracle_ET[1] else NA_real_
} else NA_real_

## double bootstrap for one real data sample -------------------------------------
double_boot_one <- function(seed) {
  set.seed(seed)
  X <- MASS::mvrnorm(cfg$n, rep(0, p), pop$Sigma)
  S <- crossprod(sweep(X, 2, colMeans(X))) / cfg$n
  ld <- tryCatch(as.numeric(determinant(S, logarithm = TRUE)$modulus), error = function(e) NA_real_)
  if (is.na(ld)) return(NULL)
  unc <- tryCatch(eng$fit_unc(S, ld, z_true), error = function(e) NULL)
  if (is.null(unc) || any(eng$unpack(unc$z)$psi <= 0)) return(NULL)   # improper drop (match coverage.R)
  con <- eng$fit_con(S, ld, unc$z, gfun, g_pop)       # theta_tilde at the null
  if (is.null(con)) return(NULL)
  T_real <- max(0, cfg$n * (con$F - unc$F))           # coverage statistic T(g_pop) for THIS sample
  z_tilde <- con$z; Sig1 <- eng$sigma_of(z_tilde)
  L1 <- tryCatch(chol(Sig1), error = function(e) NULL); if (is.null(L1)) return(NULL)

  Ts_out <- numeric(0); c_inner <- numeric(0)
  for (b in seq_len(cfg$B1)) {
    Xb <- matrix(stats::rnorm(cfg$n * p), cfg$n, p) %*% L1
    Xb <- sweep(Xb, 2, colMeans(Xb)); Sb <- crossprod(Xb) / cfg$n
    ldb <- tryCatch(as.numeric(determinant(Sb, logarithm = TRUE)$modulus), error = function(e) NA_real_)
    if (is.na(ldb)) next
    ub <- tryCatch(eng$fit_unc(Sb, ldb, z_tilde), error = function(e) NULL); if (is.null(ub)) next
    cb <- eng$fit_con(Sb, ldb, ub$z, gfun, g_pop); if (is.null(cb)) next
    Ts_out <- c(Ts_out, max(0, cfg$n * (cb$F - ub$F)))         # outer T -> c_plugin
    ci <- boot_lr_mean(eng, eng$sigma_of(cb$z), cfg$n, cb$z, gfun, g_pop, cfg$B2, min_frac = 0.3)
    if (!is.null(ci)) c_inner <- c(c_inner, ci$c)             # inner c(theta_tilde*_b)
  }
  if (length(Ts_out) < 0.3 * cfg$B1 || length(c_inner) < 0.3 * cfg$B1) return(NULL)
  c_plugin <- mean(Ts_out); Estar <- mean(c_inner)
  data.frame(seed = seed, T_real = T_real, c_plugin = c_plugin, Estar = Estar,
             c_add = 2 * c_plugin - Estar, c_mult = c_plugin^2 / Estar,
             stringsAsFactors = FALSE)
}

message(sprintf("double-boot: pop=%s n=%d samples=%d B1=%d B2=%d cores=%d | oracle c=%.2f",
                cfg$pop, cfg$n, cfg$samples, cfg$B1, cfg$B2, cfg$cores, oracle_c))
seeds <- cfg$seed_base + 500000L + seq_len(cfg$samples)
t0 <- Sys.time()
rr <- parallel::mclapply(seeds, double_boot_one, mc.cores = cfg$cores, mc.preschedule = FALSE)
rr <- do.call(rbind, rr[vapply(rr, is.data.frame, logical(1))])
message(sprintf("  %d/%d samples usable, %.0fs", nrow(rr), cfg$samples,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))

write_csv(rr, file.path(res_dir, sprintf("double_boot_%s_n%d.csv", cfg$pop, cfg$n)))

q1 <- qchisq(0.95, 1)
# per-sample threshold scale, clamped at 1 (a correction never shrinks the chi^2 ref)
cov_at <- function(cscale) mean(rr$T_real <= pmax(1, cscale) * q1)
summ <- data.frame(
  quantity = c("oracle c (target)", "c_plugin (1-level null-boot)",
               "c_add (additive double)", "c_mult (multiplicative double)"),
  mean_c = c(oracle_c, mean(rr$c_plugin), mean(rr$c_add), mean(rr$c_mult)),
  c_vs_oracle = c(1, mean(rr$c_plugin), mean(rr$c_add), mean(rr$c_mult)) / oracle_c,
  coverage = c(mean(rr$T_real <= oracle_c * q1), cov_at(rr$c_plugin),
               cov_at(rr$c_add), cov_at(rr$c_mult)))
options(width = 200)
cat(sprintf("\nDouble-bootstrap correction (bifactor rho*, normal data; %d samples, nominal coverage 0.95):\n",
            nrow(rr)))
cat(sprintf("Uncorrected (chi^2_1) coverage: %.3f\n", mean(rr$T_real <= q1)))
print(summ, row.names = FALSE, digits = 3)
cat(sprintf("\nRecovery of oracle c (mean, tail-noisy): plug-in %.0f%% -> additive %.0f%% -> multiplicative %.0f%%\n",
            100 * mean(rr$c_plugin) / oracle_c, 100 * mean(rr$c_add) / oracle_c,
            100 * mean(rr$c_mult) / oracle_c))
cat("Wrote ", file.path(res_dir, sprintf("double_boot_%s_n%d.csv", cfg$pop, cfg$n)), "\n", sep = "")
