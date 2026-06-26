#!/usr/bin/env Rscript
# Cornish-Fisher interval for bifactor maximal reliability, on the logit scale.
# The logit-scale second-order *point* correction (bias_correction.R) leaves one
# gap: the general factor at small N, whose sampling distribution piles against the
# ceiling at 1 -- a skewness a symmetric Wald interval cannot follow. CF corrects
# the quantiles for that skewness analytically (skewness from a single
# Hessian-vector product, no bootstrap; see R/maxrel.R cf_logit_interval). We
# compare three logit-scale intervals by coverage of the population rho*:
#   wald_uncorr   logit-Wald around rho-hat
#   wald_corr     logit-Wald around the second-order-corrected point
#   cf            Cornish-Fisher (bias b* + skew g1 in the pivot quantiles)
# and report the mean skewness g1 and the fraction of replicates where |g1| is past
# the CF monotonicity range (the expansion on thin ice).
#
# Usage:
#   Rscript scripts/cf_interval.R [--reps 600] [--ns 50,100,200]
#                                 [--pkeys p9,p18] [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/cf_interval.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/cf_interval.R", mustWork = FALSE)
  dirname(dirname(sp))
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))

parse_args <- function(a) {
  o <- list(reps = 600L, ns = c(50L, 100L, 200L), pkeys = c("p9", "p18"),
            seed_base = 20260630L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (startsWith(x, "--reps=")) o$reps <- as.integer(sub("^--reps=", "", x))
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (startsWith(x, "--ns=")) o$ns <- as.integer(strsplit(sub("^--ns=", "", x), ",")[[1L]])
    else if (x == "--pkeys") { i <- i + 1L; o$pkeys <- strsplit(a[[i]], ",")[[1L]] }
    else if (startsWith(x, "--pkeys=")) o$pkeys <- strsplit(sub("^--pkeys=", "", x), ",")[[1L]]
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 60L; o$ns <- c(50L); o$pkeys <- c("p18") }
  o
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))
res_dir <- file.path(exp_dir, "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

pops_all <- list(
  p9  = bifactor_population(per = 3L, gen = 0.6, grp = c(0.5, 0.4, 0.45)),
  p18 = bifactor_population(per = 6L, gen = 0.6, grp = c(0.5, 0.4, 0.45)))
pops <- pops_all[cfg$pkeys]
specs <- lapply(pops, function(pop)
  magmaan::model_spec(pop$syntax, orthogonal = TRUE, std_lv = TRUE))
dists <- c("normal", "chisq")
methods <- c("wald_uncorr", "wald_corr", "cf")
clamp01 <- function(x, eps = 1e-4) pmin(1 - eps, pmax(eps, x))
ci_logit_pt <- function(point, se) {
  p <- clamp01(point); eta <- stats::qlogis(p); se_eta <- se / (p * (1 - p))
  stats::plogis(c(eta - z975 * se_eta, eta + z975 * se_eta))
}

# Per coefficient: the three intervals plus g1 / nonmono. NULL on a failed fit.
one_rep <- function(spec, dat, pop) {
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  pt <- fit$partable; rebuild <- make_rebuilder(pt, pop)
  if (any(rebuild(pt$est)$P <= 0)) return(NULL)
  Vr <- tryCatch(stats::vcov(fit, regime = "robust", data = dat),
                 error = function(e) NULL)
  if (!is.matrix(Vr)) return(NULL)
  lapply(c(gen = "gen", grp = "grp"), function(w) {
    rho  <- rho_of_theta(pt$est, rebuild, pop, w)
    grad <- grad_rho(pt$est, pt, rebuild, pop, w)
    se   <- sqrt(max(0, as.numeric(crossprod(grad, Vr %*% grad))))
    cb_l <- curvature_bias(pt$est, pt, rebuild, pop, w, Vr, link = "logit")
    rho_c <- stats::plogis(stats::qlogis(clamp01(rho)) - cb_l)
    cf <- cf_logit_interval(pt$est, pt, rebuild, pop, w, Vr)
    if (is.null(cf)) return(NULL)
    list(intervals = list(wald_uncorr = ci_logit_pt(rho, se),
                          wald_corr   = ci_logit_pt(rho_c, se),
                          cf          = c(cf$lo, cf$hi)),
         gamma1 = cf$gamma1, nonmono = cf$nonmono)
  })
}

one_cell <- function(pkey, n, dist, seed0) {
  pop <- pops[[pkey]]; spec <- specs[[pkey]]
  tgt <- c(gen = pop$rho_gen, grp = pop$rho_grp1)
  acc <- list(); for (coef in c("gen","grp")) for (m in methods)
    acc[[paste(coef,m,sep=".")]] <- list(cov = 0L, width = 0)
  g1 <- c(gen = 0, grp = 0); nm <- c(gen = 0L, grp = 0L)
  ok <- c(gen = 0L, grp = 0L)
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    res <- one_rep(spec, gen_data(pop, n, dist), pop)
    if (is.null(res)) next
    for (coef in c("gen", "grp")) {
      v <- res[[coef]]; if (is.null(v)) next
      ok[coef] <- ok[coef] + 1L; g1[coef] <- g1[coef] + v$gamma1
      nm[coef] <- nm[coef] + as.integer(v$nonmono)
      for (m in methods) {
        ci <- v$intervals[[m]]
        a <- acc[[paste(coef,m,sep=".")]]
        a$cov <- a$cov + as.integer(is.finite(ci[1]) && ci[1] <= tgt[[coef]] && tgt[[coef]] <= ci[2])
        a$width <- a$width + (ci[2] - ci[1])
        acc[[paste(coef,m,sep=".")]] <- a
      }
    }
  }
  rows <- list()
  for (coef in c("gen","grp")) for (m in methods) {
    a <- acc[[paste(coef,m,sep=".")]]
    rows[[length(rows)+1L]] <- data.frame(
      p = pop$p, coef = coef, dist = dist, n = n, method = m, reps_ok = ok[coef],
      rho_pop = tgt[[coef]], coverage = a$cov / ok[coef], mean_width = a$width / ok[coef],
      mean_gamma1 = g1[coef] / ok[coef], frac_nonmono = nm[coef] / ok[coef],
      stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

grid <- expand.grid(pkey = names(pops), n = cfg$ns, dist = dists,
                    stringsAsFactors = FALSE)
all_rows <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  gg <- grid[i, ]
  message(sprintf("cell %d/%d: %s n=%d %s (reps=%d)",
                  i, nrow(grid), gg$pkey, gg$n, gg$dist, cfg$reps))
  one_cell(gg$pkey, gg$n, gg$dist, cfg$seed_base + i * 100000L)
}))

write_csv(all_rows, file.path(res_dir, "cf_coverage.csv"))
write_metadata(
  file.path(res_dir, "cf_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
                pkeys = paste(cfg$pkeys, collapse = ","), seed_base = cfg$seed_base,
                methods = "wald_uncorr, wald_corr (second-order point), cf (Cornish-Fisher, logit)",
                cf = "pivot q(z)=b*+z+(g1/6)(z^2-1), g1 analytic via Hessian-vector product",
                smoke = cfg$smoke),
  packages = "magmaan")
cat("\nCoverage by interval method (nominal 0.95):\n")
print(all_rows[order(all_rows$coef, all_rows$p, all_rows$dist, all_rows$n,
                     match(all_rows$method, methods)),
               c("coef","p","dist","n","method","coverage","mean_width","mean_gamma1","frac_nonmono")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "cf_coverage.csv"), "\n", sep = "")
