#!/usr/bin/env Rscript

# Advisory cross-check of magmaan's PLSIM generator against the covsim reference
# (Foldnes & Gronneberg piecewise-linear transforms, external/r_source/covsim).
# Outside CTest by design: it needs the covsim R sources under external/ plus
# tmvtnorm/MASS/Matrix/lavaan, none of which CI carries. Mirrors the structure of
# tests/checks/ig/pearson_draw_bench.R, including the graceful skip when covsim
# is absent.
#
# Headline metric. We feed magmaan's OWN fitted marginals and solved intermediate
# correlation into covsim's exact truncated-moment integrator get_cov() and check
# that the implied correlation equals the target. This validates magmaan's
# covariance integral + root-find against covsim's integrator WITHOUT being
# confounded by marginal-fit non-uniqueness: the piecewise-linear marginal for a
# given (skewness, excess kurtosis) is not unique, so the two engines generally
# fit different slopes and therefore different intermediate correlations even when
# both are correct. A raw intermediate-correlation comparison is thus not
# apples-to-apples; we still report it, flagged informational. The headline,
# using magmaan's own marginals on both sides of the integral, is clean and works
# even when covsim's rPLSIM cannot fit a shape at a low segment count.
#
# Run from the repo root (or via `just vs-covsim` from this directory):
#   Rscript tests/checks/plsim/plsim_vs_covsim.R --n=100000

suppressPackageStartupMessages(library(magmaan))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (length(hit) == 0L) return(default)
  sub(prefix, "", hit[[1L]], fixed = TRUE)
}

n        <- as.integer(get_arg("n", "100000"))
reps     <- as.integer(get_arg("reps", "1"))
nseg     <- as.integer(get_arg("num-segments", "4"))
seed     <- as.integer(get_arg("seed", "20260617"))
methods  <- strsplit(get_arg("methods",
  "rectangle,quadrature,hermite,hermite_then_quadrature,hermite_then_rectangle"),
  ",", fixed = TRUE)[[1L]]
headline <- "rectangle"

core <- magmaan::magmaan_core

# Advisory PASS/WARN tolerances.
tol <- list(
  gamma    = 1e-9,   # breakpoints: both engines use qnorm((1:k)/k)[1:(k-1)]
  mean     = 1e-6,   # realized marginal mean (should be 0)
  var      = 1e-6,   # realized marginal variance (should be 1)
  shape    = 5e-3,   # realized skew / excess kurtosis vs target
  headline = 3e-3,   # covsim get_cov(magmaan marginals, magmaan z) vs target (rectangle)
  achieved = 1e-5,   # magmaan self-reported achieved_corr vs target
  s_corr   = 0.035,  # sample correlation vs target (matches unit-test MC tol)
  s_skew   = 0.12,
  s_kurt   = 0.45
)
tag <- function(ok) if (isTRUE(ok)) "PASS" else "WARN"

# ---- locate + load the covsim reference (graceful skip if absent) ----
file_first <- function(cands) {
  for (p in cands) if (file.exists(p)) return(p)
  NA_character_
}
cov_dir <- file_first(c(
  "external/r_source/covsim/R",
  "../../../external/r_source/covsim/R"
))
have_pkg  <- function(p) requireNamespace(p, quietly = TRUE)
have_tmv  <- have_pkg("tmvtnorm")
have_rsim <- all(vapply(c("MASS", "Matrix", "lavaan"), have_pkg, logical(1)))

aux_path   <- if (!is.na(cov_dir)) file.path(cov_dir, "auxiliary.R") else NA_character_
plsim_path <- if (!is.na(cov_dir)) file.path(cov_dir, "PLSIM.R")     else NA_character_

have_getcov <- !is.na(aux_path)   && file.exists(aux_path)   && have_tmv
have_rplsim <- have_getcov && !is.na(plsim_path) && file.exists(plsim_path) && have_rsim
if (have_getcov) source(aux_path,   local = FALSE)
if (have_rplsim) source(plsim_path, local = FALSE)

cat(sprintf("PLSIM vs covsim | n=%d reps=%d num_segments=%d seed=%d\n",
            n, reps, nseg, seed))
cat(sprintf("methods: %s | headline: %s\n", paste(methods, collapse = ","), headline))
cat(sprintf("covsim get_cov available: %s | covsim rPLSIM available: %s\n\n",
            have_getcov, have_rplsim))
if (!have_getcov)
  cat(paste0("NOTE: covsim get_cov/tmvtnorm unavailable; reporting magmaan ",
             "self-consistency only.\n\n"))

# ---- helpers ----
sample_moments <- function(X) {
  mu <- colMeans(X)
  Xc <- sweep(X, 2L, mu)
  m2 <- colMeans(Xc^2); m3 <- colMeans(Xc^3); m4 <- colMeans(Xc^4)
  list(corr = stats::cor(X), skew = m3 / m2^1.5, exk = m4 / m2^2 - 3)
}
offdiag_maxabs <- function(A, B) {
  D <- abs(as.matrix(A) - as.matrix(B)); diag(D) <- 0; max(D)
}
# Max |covsim get_cov(magmaan marginals, magmaan intermediate rho) - target|.
# Each fitted marginal has unit variance, so get_cov returns the implied
# correlation; the residual is magmaan's covariance/root-find error measured by
# covsim's integrator.
headline_resid <- function(marg, zcorr, target) {
  p <- length(marg); mx <- 0
  for (i in 1:(p - 1)) for (j in (i + 1):p) {
    c_ij <- get_cov(marg[[i]]$a, marg[[i]]$b, marg[[i]]$gamma,
                    marg[[j]]$a, marg[[j]]$b, marg[[j]]$gamma, zcorr[i, j])
    mx <- max(mx, abs(c_ij - target[i, j]))
  }
  mx
}

# ---- grid: moderate shapes feasible at num_segments=4, PD correlation targets ----
# (Case 1 is covsim's own docstring example: skew=1, excess kurtosis=4.)
make_corr <- function(p, off) { R <- matrix(off, p, p); diag(R) <- 1; R }
grid <- list(
  list(name = "p2 sk1-ku4",   R = make_corr(2, 0.5),
       skew = c(1, 1),               exk = c(4, 4)),
  list(name = "p3 mixed",     R = make_corr(3, 0.4),
       skew = c(1, -1, 0.5),         exk = c(4, 3, 2)),
  list(name = "p2 neg-corr",  R = make_corr(2, -0.6),
       skew = c(0.8, -0.8),          exk = c(3, 3)),
  list(name = "p4 mixed-sign",
       R = rbind(c( 1.0, 0.5, 0.3, -0.2),
                 c( 0.5, 1.0, 0.4,  0.1),
                 c( 0.3, 0.4, 1.0,  0.3),
                 c(-0.2, 0.1, 0.3,  1.0)),
       skew = c(1, -0.8, 0.6, -0.5),  exk = c(4, 3, 2.5, 3.5))
)

warns <- 0L
bump <- function(ok) if (!isTRUE(ok)) warns <<- warns + 1L

for (case in grid) {
  p <- ncol(case$R)
  cat(sprintf("== %s (p=%d) ==\n", case$name, p))
  if (min(eigen(case$R, symmetric = TRUE, only.values = TRUE)$values) <= 0) {
    cat("  target correlation not PD; skipping case\n\n"); next
  }

  # magmaan calibrations (marginals identical across methods; covariance differs)
  cals <- list()
  for (m in methods) {
    cals[[m]] <- tryCatch(
      core$sim_plsim_calibrate(case$R, case$skew, case$exk,
                               method = m, num_segments = nseg, monotone = FALSE),
      error = function(e) {
        cat(sprintf("  magmaan calibrate [%s] failed: %s\n", m, conditionMessage(e)))
        NULL
      })
  }
  if (is.null(cals[[headline]])) { cat("  headline method failed; skipping case\n\n"); next }
  marg <- cals[[headline]]$marginals

  # Tier: marginal convention + realized moments (covsim moment formula).
  if (have_getcov) {
    cat("  marginal fit (covsim get_pl_moments on magmaan marginals):\n")
    for (j in seq_len(p)) {
      mom <- get_pl_moments(marg[[j]]$a, marg[[j]]$b, marg[[j]]$gamma)
      rmean <- mom[1]; rvar <- mom[2]
      rskew <- mom[3] / rvar^1.5; rexk <- mom[4] / rvar^2 - 3
      ok <- abs(rmean) < tol$mean && abs(rvar - 1) < tol$var &&
            abs(rskew - case$skew[j]) < tol$shape && abs(rexk - case$exk[j]) < tol$shape
      bump(ok)
      cat(sprintf(
        "    v%d: mean=% .2e var=%.6f skew=%.4f(t%.2f) exk=%.4f(t%.2f) [%s]\n",
        j, rmean, rvar, rskew, case$skew[j], rexk, case$exk[j], tag(ok)))
    }
  }

  # Tier: HEADLINE covariance cross-check (per method).
  if (have_getcov) {
    cat("  headline covariance cross-check (covsim get_cov on magmaan marginals+z, resid vs target):\n")
    for (m in methods) {
      if (is.null(cals[[m]])) next
      r <- headline_resid(cals[[m]]$marginals, cals[[m]]$intermediate_corr, case$R)
      if (m == headline) {
        ok <- r < tol$headline; bump(ok)
        cat(sprintf("    %-24s max_resid=%.3e [%s headline]\n", m, r, tag(ok)))
      } else {
        cat(sprintf("    %-24s max_resid=%.3e %s\n", m, r,
                    if (r < 1e-2) "[ok]" else "[info-large]"))
      }
    }
  }

  # Tier: magmaan self-reported achieved correlation vs target.
  ach <- offdiag_maxabs(cals[[headline]]$achieved_corr, case$R)
  ok <- ach < tol$achieved; bump(ok)
  cat(sprintf("  achieved corr (magmaan %s self-report vs target): max=%.3e [%s]\n",
              headline, ach, tag(ok)))

  # covsim reference run (optional; needed only for cross-engine comparisons).
  cov_res <- NULL
  if (have_rplsim) {
    cov_res <- tryCatch(
      suppressMessages(rPLSIM(N = n, sigma.target = case$R, skewness = case$skew,
                              excesskurtosis = case$exk, reps = reps,
                              numsegments = nseg, monot = FALSE, verbose = FALSE)),
      error = function(e) {
        cat(sprintf("  covsim rPLSIM failed (numsegments=%d): %s\n",
                    nseg, conditionMessage(e)))
        NULL
      })
  }
  if (!is.null(cov_res)) {
    gmax <- 0
    for (j in seq_len(p))
      gmax <- max(gmax, max(abs(marg[[j]]$gamma - cov_res$model$gamma[[j]])))
    okg <- gmax < tol$gamma; bump(okg)
    cat(sprintf("  breakpoints gamma (magmaan vs covsim): max=%.3e [%s]\n",
                gmax, tag(okg)))
    zc <- as.matrix(cov_res$model$z.corr)
    zinfo <- offdiag_maxabs(cals[[headline]]$intermediate_corr, zc)
    cat(sprintf(paste0("  raw z.corr (magmaan %s vs covsim) [info; confounded by ",
                       "marginal non-uniqueness]: max=%.3e\n"), headline, zinfo))
  }

  # Tier: sample moments (stochastic) vs target.
  drw <- tryCatch(
    core$sim_plsim_draw(cals[[headline]], n = n, reps = 1L, seed_base = seed),
    error = function(e) NULL)
  if (!is.null(drw)) {
    sm <- sample_moments(drw$draws[[1L]])
    sc <- offdiag_maxabs(sm$corr, case$R)
    ss <- max(abs(sm$skew - case$skew)); sk <- max(abs(sm$exk - case$exk))
    okm <- sc < tol$s_corr && ss < tol$s_skew && sk < tol$s_kurt; bump(okm)
    cat(sprintf("  sample moments magmaan (n=%d): corr=%.4f skew=%.4f exk=%.4f [%s]\n",
                n, sc, ss, sk, tag(okm)))
  }
  if (!is.null(cov_res)) {
    smc <- sample_moments(cov_res$samples[[1L]])
    scc <- offdiag_maxabs(smc$corr, case$R)
    ssc <- max(abs(smc$skew - case$skew)); skc <- max(abs(smc$exk - case$exk))
    cat(sprintf("  sample moments covsim  (n=%d): corr=%.4f skew=%.4f exk=%.4f\n",
                n, scc, ssc, skc))
  }
  cat("\n")
}

# ---- discrimination probe: prove the headline check has teeth ----
if (have_getcov) {
  case <- grid[[1L]]
  cal <- tryCatch(
    core$sim_plsim_calibrate(case$R, case$skew, case$exk,
                             method = headline, num_segments = nseg, monotone = FALSE),
    error = function(e) NULL)
  if (!is.null(cal)) {
    good <- headline_resid(cal$marginals, cal$intermediate_corr, case$R)
    z2 <- cal$intermediate_corr; z2[1, 2] <- z2[2, 1] <- z2[1, 2] + 0.05
    bad <- headline_resid(cal$marginals, z2, case$R)
    cat(sprintf(paste0("discrimination probe (%s): resid at solved z=%.3e, ",
                       "at z+0.05=%.3e (ratio %.0fx)\n"),
                case$name, good, bad, bad / max(good, 1e-12)))
  }
}

cat(sprintf("\nTOTAL WARN tiers: %d (advisory; exit 0)\n", warns))
