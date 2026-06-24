#!/usr/bin/env Rscript
# Coverage-vs-N sweep: the proof-of-concept that distinguishes OUR contribution
# (the complete estimated-weight infinitesimal jackknife) from Lai & Simoes
# (2023). Three analytic-normal SEs for the same focal ordinal-DWLS loading,
# scored by CI coverage of the pseudo-true theta* (nominal 0.95) across N, under
# the correct model and under an omitted cross-loading:
#   - expected bread  (Muthen 1997 Eq 17 = lavaan default = Lai Method II)
#   - observed bread   (Lai & Simoes 2023 new SE, Eq 36 = Lai Method III)
#   - complete IJ      (estimated-weight infinitesimal jackknife = the Wd term
#                       Lai left open; magmaan robust_ordinal_ij)
# Prediction (the headline): under misspecification the expected bread
# under-covers and does not improve with N; the observed bread recovers most but
# plateaus BELOW 0.95 for DWLS and does not converge with N (the weight Wd=Gd^-1
# is held fixed); the complete IJ converges to ~0.95. Under the correct model all
# three coincide near 0.95. ULS would make observed==IJ (W=I fixed); DWLS is the
# cell where the gap lives.
#
# No bootstrap here (that is the separate, expensive arm in scripts/bootstrap.R);
# all three SEs are analytic, so the N-sweep is cheap.
#
# Usage:
#   Rscript scripts/coverage_sweep.R [--reps 400] [--ns 600,1500,3000,6000]
#                                    [--cross 0.4] [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/coverage_sweep.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
source(.support_helpers()); rm(.support_helpers)

parse_args <- function(a) {
  o <- list(reps = 400L, ns = c(600L, 1500L, 3000L, 6000L), cross = 0.4,
            seed_base = 20260624L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (x == "--cross") { i <- i + 1L; o$cross <- as.numeric(a[[i]]) }
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") { o$smoke <- TRUE }
    else stop("unknown arg: ", x)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 20L; o$ns <- c(600L) }
  if (any(!is.finite(o$ns)) || any(o$ns < 120L)) stop("--ns values must be >= 120")
  if (!is.finite(o$reps) || o$reps < 2L) stop("--reps must be >= 2")
  o
}
cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan"); suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]))
        else normalizePath("scripts/coverage_sweep.R")
  rd <- file.path(dirname(dirname(sp)), "results")
  dir.create(rd, showWarnings = FALSE, recursive = TRUE); rd
}

# Same structural design as scripts/bootstrap.R: two-factor 6-indicator ordinal
# model; in the DGP y4 cross-loads on f1, the fitted model omits it. Focal: the
# distorted f2=~y5 loading.
ov <- paste0("y", 1:6); thr <- c(-0.8, 0, 0.8); loading <- 0.7; fcor <- 0.3
spec <- magmaan::model_spec("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6",
                            ordered = ov, parameterization = "delta")
focrow <- function(fit) { pt <- fit$partable; which(pt$lhs == "f2" & pt$op == "=~" & pt$rhs == "y5") }

gen <- function(n, cross) {
  f1 <- rnorm(n); f2 <- fcor * f1 + sqrt(1 - fcor^2) * rnorm(n)
  rsd <- sqrt(1 - loading^2); y4sd <- sqrt(max(1e-3, 1 - cross^2 - loading^2))
  z <- cbind(loading * f1 + rnorm(n, sd = rsd), loading * f1 + rnorm(n, sd = rsd),
             loading * f1 + rnorm(n, sd = rsd),
             cross * f1 + loading * f2 + rnorm(n, sd = y4sd),
             loading * f2 + rnorm(n, sd = rsd), loading * f2 + rnorm(n, sd = rsd))
  d <- as.data.frame(lapply(1:6, function(j) ordered(cut(z[, j], c(-Inf, thr, Inf), labels = FALSE))))
  names(d) <- ov; d
}

# Fit DWLS and return the focal estimate plus the three analytic SEs. NULL on any
# fit/SE failure (so a bad replicate is dropped, not silently mislabeled).
fit_focal <- function(d, want_se = FALSE) {
  st <- tryCatch(core$data_ordinal_stats_from_df(d, spec), error = function(e) NULL)
  if (is.null(st)) return(NULL)
  fit <- tryCatch(core$fit_dwls_ordinal(spec, st), error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  r <- focrow(fit); fi <- fit$partable$free[r]
  out <- list(est = fit$partable$est[r], fi = fi)
  if (want_se) {
    re <- tryCatch(core$robust_ordinal(fit, st, "DWLS", "expected"), error = function(e) NULL)
    ro <- tryCatch(core$robust_ordinal(fit, st, "DWLS", "observed"), error = function(e) NULL)
    rj <- tryCatch(core$robust_ordinal_ij(fit, st, "DWLS"), error = function(e) NULL)
    if (is.null(re) || is.null(ro) || is.null(rj)) return(NULL)
    out$se_exp <- re$se[fi]; out$se_obs <- ro$se[fi]; out$se_ij <- rj$se[fi]
  }
  out
}

# Pseudo-true theta* per cell (probability limit of the focal estimate), one
# huge-N fit per data-generating process.
theta_star_null <- fit_focal(gen(200000L, 0.0))$est
theta_star_mis  <- fit_focal(gen(200000L, cfg$cross))$est

qn <- qnorm(c(.025, .975))
one_rep <- function(cross, n, seed) {
  set.seed(seed)
  f <- fit_focal(gen(n, cross), want_se = TRUE)
  if (is.null(f)) return(NULL)
  data.frame(est = f$est, se_exp = f$se_exp, se_obs = f$se_obs, se_ij = f$se_ij)
}

cover <- function(est, se, theta_star) {
  lo <- est + qn[1] * se; hi <- est + qn[2] * se
  is.finite(se) && lo <= theta_star && theta_star <= hi
}

run_cell <- function(cell, cross, n, off, theta_star) {
  rows <- lapply(seq_len(cfg$reps), function(r) one_rep(cross, n, cfg$seed_base + off + r))
  D <- do.call(rbind, Filter(Negate(is.null), rows))
  data.frame(
    cell = cell, n_total = n, cross = cross, reps = nrow(D), theta_star = theta_star,
    emp_sd = sd(D$est),
    mean_se_exp = mean(D$se_exp), mean_se_obs = mean(D$se_obs), mean_se_ij = mean(D$se_ij),
    cov_exp = mean(mapply(cover, D$est, D$se_exp, MoreArgs = list(theta_star = theta_star))),
    cov_obs = mean(mapply(cover, D$est, D$se_obs, MoreArgs = list(theta_star = theta_star))),
    cov_ij  = mean(mapply(cover, D$est, D$se_ij,  MoreArgs = list(theta_star = theta_star))),
    stringsAsFactors = FALSE)
}

rows <- list()
for (k in seq_along(cfg$ns)) {
  n <- cfg$ns[k]
  off_null <- (k - 1L) * 100000L
  off_mis  <- 5000000L + (k - 1L) * 100000L
  rows[[length(rows) + 1L]] <- run_cell("null", 0.0, n, off_null, theta_star_null)
  rows[[length(rows) + 1L]] <- run_cell("misspec", cfg$cross, n, off_mis, theta_star_mis)
  cat(sprintf("  n=%-6d done (null + misspec)\n", n))
}
summary <- do.call(rbind, rows)
write_csv(summary, file.path(res_dir, "coverage_sweep.csv"))
write_metadata(file.path(res_dir, "coverage_sweep_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","), cross = cfg$cross,
                seed_base = cfg$seed_base,
                theta_star_null = theta_star_null, theta_star_misspec = theta_star_mis,
                estimator = "DWLS_ordinal", focal = "f2=~y5 raw loading",
                methods = "expected (lavaan/Lai-II), observed (Lai-III), complete-IJ (ours)",
                target = "coverage of pseudo-true theta* (nominal 0.95)", smoke = cfg$smoke),
  packages = "magmaan")
print(summary, row.names = FALSE, digits = 4)
cat("\nWrote results to ", res_dir, "\n", sep = "")
