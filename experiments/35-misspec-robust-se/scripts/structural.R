#!/usr/bin/env Rscript
# Probe for the residual the observed-Hessian bread leaves behind: the influence
# of the data-dependent weight Ŵ = diag(Γ̂)⁻¹, which in magmaan is the EMPIRICAL
# two-stage sandwich (src/data/ordinal.cpp: NACOV = n·B_inv·INNER·B_inv'). The
# observed bread fixes only the Hessian; the meat still uses Δ'W, omitting the
# influence of the estimated variance Γ̂ -- an O(1)-under-misspecification term.
#
# The clean, tail-robust headline is the PER-REP ratio se_jack / se_obs: two SE
# estimators of the SAME focal parameter, so no fragile empirical SD enters.
#   - se_obs  = observed-Hessian sandwich, FIXED weight (omits the Γ̂ influence)
#   - se_jack = grouped (delete-d) jackknife: refits the whole pipeline -- incl.
#     Γ̂ and Ŵ -- on each delete-group, so it CAPTURES the weight influence. This
#     is the numerical infinitesimal jackknife.
# Prediction: ULS (W = I, fixed) ⇒ se_jack/se_obs ≈ 1 (no weight term); DWLS
# (W estimated) ⇒ se_jack/se_obs > 1 under misspecification (the omitted term),
# and ≈ 1 under the null. emp_sd (trimmed, with the trim count reported) is a
# secondary "which SE is closer to truth" check, not the headline -- the raw SD
# is heavy-tail-sensitive for this focal.
#
# Usage:
#   Rscript scripts/structural.R [--reps 200] [--ns 2500] [--jack 30]
#                                [--cross 0.4] [--beta 0.3] [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/structural.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
source(.support_helpers()); rm(.support_helpers)

parse_args <- function(a) {
  o <- list(reps = 200L, ns = c(2500L), cross = 0.4, beta = 0.3, jack = 30L,
            seed_base = 20260619L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--jack") { i <- i + 1L; o$jack <- as.integer(a[[i]]) }
    else if (x == "--cross") { i <- i + 1L; o$cross <- as.numeric(a[[i]]) }
    else if (x == "--beta") { i <- i + 1L; o$beta <- as.numeric(a[[i]]) }
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") { o$smoke <- TRUE }
    else stop("unknown arg: ", x)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 8L; o$ns <- c(800L); o$jack <- 8L }
  o
}
cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan"); suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]))
        else normalizePath("scripts/structural.R")
  rd <- file.path(dirname(dirname(sp)), "results")
  dir.create(rd, showWarnings = FALSE, recursive = TRUE); rd
}

ov <- paste0("y", 1:6); thr <- c(-0.8, 0, 0.8); loading <- 0.7
# Structural model: f2 ~ f1. The DGP gives y4 a cross-loading on f1 (omitted in
# the fit), contaminating the f2 measurement and so distorting the focal path
# f2~f1 -- a Stage-2 structural misspecification; measurement model correct.
spec <- magmaan::model_spec(
  "f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6\nf2 ~ f1",
  ordered = ov, parameterization = "delta")
focrow <- function(fit) {
  pt <- fit$partable; which(pt$lhs == "f2" & pt$op == "~" & pt$rhs == "f1")
}

gen <- function(n, cross, beta) {
  f1 <- rnorm(n)
  f2 <- beta * f1 + sqrt(1 - beta^2) * rnorm(n)
  rsd <- sqrt(1 - loading^2)
  y4sd <- sqrt(max(1e-3, 1 - cross^2 - loading^2 - 2 * cross * loading * beta))
  z <- cbind(loading * f1 + rnorm(n, sd = rsd), loading * f1 + rnorm(n, sd = rsd),
             loading * f1 + rnorm(n, sd = rsd),
             cross * f1 + loading * f2 + rnorm(n, sd = y4sd),
             loading * f2 + rnorm(n, sd = rsd), loading * f2 + rnorm(n, sd = rsd))
  d <- as.data.frame(lapply(1:6, function(j)
    ordered(cut(z[, j], c(-Inf, thr, Inf), labels = FALSE))))
  names(d) <- ov; d
}

fitter <- function(est) if (est == "ULS") core$fit_uls_ordinal else core$fit_dwls_ordinal

# Focal estimate for one data set; with want_se, the model (expected-bread) and
# observed-bread analytic SEs as well.
fit_focal <- function(d, est, want_se = FALSE) {
  st <- tryCatch(core$data_ordinal_stats_from_df(d, spec), error = function(e) NULL)
  if (is.null(st)) return(NULL)
  fit <- tryCatch(fitter(est)(spec, st), error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  r <- focrow(fit); if (!length(r)) return(NULL)
  fi <- fit$partable$free[r]; if (is.na(fi) || fi < 1) return(NULL)
  out <- list(est = fit$partable$est[r], fi = fi)
  if (want_se) {
    re <- tryCatch(core$robust_ordinal(fit, st, est, "expected"), error = function(e) NULL)
    ro <- tryCatch(core$robust_ordinal(fit, st, est, "observed"), error = function(e) NULL)
    if (is.null(re) || is.null(ro)) return(NULL)
    out$se_model <- re$se[fi]; out$se_obs <- ro$se[fi]
  }
  out
}

# Grouped (delete-d) jackknife SE of the focal estimate: refit on each of G
# leave-one-group-out subsamples, var = (G-1)/G · Σ (θ_(g) - θ̄)². Refitting
# rebuilds Γ̂ and Ŵ, so this captures the weight-estimation influence.
jack_se <- function(d, est, G) {
  n <- nrow(d); grp <- ((seq_len(n) - 1L) %% G) + 1L
  th <- vapply(seq_len(G), function(g) {
    f <- fit_focal(d[grp != g, , drop = FALSE], est)
    if (is.null(f)) NA_real_ else f$est
  }, numeric(1))
  th <- th[is.finite(th)]
  if (length(th) < G - 1L) return(NA_real_)
  sqrt((length(th) - 1) / length(th) * sum((th - mean(th))^2))
}

one_rep <- function(est, n, cross, beta, seed, do_jack) {
  set.seed(seed)
  d <- gen(n, cross, beta)
  base <- fit_focal(d, est, want_se = TRUE)
  if (is.null(base)) return(NULL)
  sj <- if (do_jack) jack_se(d, est, cfg$jack) else NA_real_
  data.frame(est = base$est, se_model = base$se_model, se_obs = base$se_obs,
             se_jack = sj)
}

# Trim only egregious outliers (|est - median| > 6·MAD): numerical-failure reps,
# not the bulk sampling distribution. Report how many were dropped.
trim_sd <- function(x) {
  m <- median(x); s <- mad(x)
  keep <- if (s > 0) abs(x - m) <= 6 * s else rep(TRUE, length(x))
  list(sd = sd(x[keep]), dropped = sum(!keep))
}

mid_n <- cfg$ns[(length(cfg$ns) + 1L) %/% 2L]
grid <- rbind(
  expand.grid(est = c("ULS", "DWLS"), n = cfg$ns, cell = "misspec",
              stringsAsFactors = FALSE),
  expand.grid(est = c("ULS", "DWLS"), n = mid_n, cell = "null",
              stringsAsFactors = FALSE))

run_cell <- function(est, n, cell, off) {
  cross <- if (cell == "null") 0.0 else cfg$cross
  do_jack <- cfg$jack > 0L
  rows <- lapply(seq_len(cfg$reps), function(r)
    one_rep(est, n, cross, cfg$beta, cfg$seed_base + off + r, do_jack))
  D <- do.call(rbind, Filter(Negate(is.null), rows))
  ts <- trim_sd(D$est)
  # Tail-robust headline ratios (per-rep, no emp_sd):
  r_obs_model <- mean(D$se_obs / D$se_model)
  jk <- D$se_jack[is.finite(D$se_jack)]
  D_jk <- D[is.finite(D$se_jack), ]
  r_jack_obs <- if (nrow(D_jk)) mean(D_jk$se_jack / D_jk$se_obs) else NA_real_
  data.frame(
    estimator = est, n = n, cell = cell, reps = nrow(D),
    emp_sd_trim = ts$sd, dropped = ts$dropped,
    mean_se_model = mean(D$se_model), mean_se_obs = mean(D$se_obs),
    mean_se_jack = if (length(jk)) mean(jk) else NA_real_,
    ratio_obs = mean(D$se_obs) / ts$sd,
    ratio_jack = if (length(jk)) mean(jk) / ts$sd else NA_real_,
    r_jack_obs = r_jack_obs, r_obs_model = r_obs_model,
    stringsAsFactors = FALSE)
}

summary <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]; run_cell(g$est, g$n, g$cell, i * 100000L)
}))
write_csv(summary, file.path(res_dir, "structural.csv"))
write_metadata(file.path(res_dir, "structural_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
                cross = cfg$cross, beta = cfg$beta, jack_groups = cfg$jack,
                seed_base = cfg$seed_base, estimators = "ULS, DWLS (ordinal delta)",
                focal = "f2~f1 structural path",
                headline = "r_jack_obs = mean per-rep se_jack/se_obs (tail-robust weight-term magnitude)",
                model = "two-factor + f2~f1; y4 cross-loads on f1 in DGP, omitted in fit",
                smoke = cfg$smoke),
  packages = "magmaan")
print(summary[, c("estimator", "n", "cell", "reps", "dropped",
                  "r_jack_obs", "ratio_obs", "ratio_jack", "r_obs_model")],
      row.names = FALSE, digits = 4)
cat("\nheadline: r_jack_obs = per-rep se_jack/se_obs (>1 ⇒ obs misses the weight term)\n")
cat("Wrote results to ", res_dir, "\n", sep = "")
