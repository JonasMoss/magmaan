#!/usr/bin/env Rscript
# Bootstrap arm of experiment 35. The analytic observed-bread ("robust" regime)
# SE for ordinal DWLS recovers most, but not all, of the finite-sample sampling
# variability under misspecification (the data-dependent Stage-2 weight Wd=Gd^-1
# contributes a higher-order term the analytic sandwich, fixed-weight, omits). A
# case-resampling bootstrap re-estimates Wd on every resample, so it should
# capture that term. We compare CI coverage of the pseudo-true value theta* for:
#   - analytic-normal, expected bread (Muthen 1997 / lavaan default)
#   - analytic-normal, observed bread (Lai & Simoes 2023 new SE; magmaan robust)
#   - bootstrap percentile
#   - BCa (bias-corrected and accelerated; jackknife acceleration)
#   - studentized (bootstrap-t), studentized by the observed-bread analytic SE
#
# Usage:
#   Rscript scripts/bootstrap.R [--reps 150] [--boot 499] [--n-total 600]
#                               [--cross 0.4] [--seed-base S] [--no-bca] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  script <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
            else normalizePath("scripts/bootstrap.R", mustWork = FALSE)
  # scripts/ is one level below the experiment dir
  file.path(dirname(dirname(dirname(script))), "_support", "R", "helpers.R")
}
source(.support_helpers()); rm(.support_helpers)

parse_args <- function(a) {
  o <- list(reps = 150L, boot = 499L, n_total = 600L, cross = 0.4,
            seed_base = 20260619L, bca = TRUE, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h","--help")) { cat("see header\n"); quit(save="no") }
    else if (x == "--reps") { i<-i+1L; o$reps <- as.integer(a[[i]]) }
    else if (x == "--boot") { i<-i+1L; o$boot <- as.integer(a[[i]]) }
    else if (x == "--n-total") { i<-i+1L; o$n_total <- as.integer(a[[i]]) }
    else if (x == "--cross") { i<-i+1L; o$cross <- as.numeric(a[[i]]) }
    else if (x == "--seed-base") { i<-i+1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--no-bca") { o$bca <- FALSE }
    else if (x == "--smoke") { o$smoke <- TRUE }
    else stop("unknown arg: ", x)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 6L; o$boot <- 99L; o$n_total <- 400L }
  o
}
cfg <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaan"); suppressPackageStartupMessages(library(magmaan))
core <- magmaan::magmaan_core
res_dir <- {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]))
        else normalizePath("scripts/bootstrap.R")
  rd <- file.path(dirname(dirname(sp)), "results")
  dir.create(rd, showWarnings = FALSE, recursive = TRUE); rd
}

ov <- paste0("y",1:6); thr <- c(-0.8,0,0.8); loading <- 0.7; fcor <- 0.3
spec <- magmaan::model_spec("f1 =~ y1 + y2 + y3\nf2 =~ y4 + y5 + y6",
                            ordered = ov, parameterization = "delta")
focrow <- function(fit){ pt<-fit$partable; which(pt$lhs=="f2"&pt$op=="=~"&pt$rhs=="y5") }

gen <- function(n, cross) {
  f1 <- rnorm(n); f2 <- fcor*f1 + sqrt(1-fcor^2)*rnorm(n)
  rsd <- sqrt(1-loading^2); y4sd <- sqrt(max(1e-3,1-cross^2-loading^2))
  z <- cbind(loading*f1+rnorm(n,sd=rsd), loading*f1+rnorm(n,sd=rsd), loading*f1+rnorm(n,sd=rsd),
             cross*f1+loading*f2+rnorm(n,sd=y4sd), loading*f2+rnorm(n,sd=rsd), loading*f2+rnorm(n,sd=rsd))
  d <- as.data.frame(lapply(1:6, function(j) ordered(cut(z[,j], c(-Inf,thr,Inf), labels=FALSE)))); names(d)<-ov; d
}
fit_focal <- function(d, want_se = FALSE) {
  st <- tryCatch(core$data_ordinal_stats_from_df(d, spec), error=function(e) NULL); if (is.null(st)) return(NULL)
  fit <- tryCatch(core$fit_dwls_ordinal(spec, st), error=function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  r <- focrow(fit); est <- fit$partable$est[r]
  out <- list(est = est, fi = fit$partable$free[r])
  if (want_se) {
    ro <- tryCatch(core$robust_ordinal(fit, st, "DWLS", "observed"), error=function(e) NULL)
    re <- tryCatch(core$robust_ordinal(fit, st, "DWLS", "expected"), error=function(e) NULL)
    if (is.null(ro) || is.null(re)) return(NULL)
    out$se_obs <- ro$se[out$fi]; out$se_exp <- re$se[out$fi]
  }
  out
}

# Pseudo-true value theta* is cell-specific (the probability limit of the focal
# estimate under each data-generating process), from one huge-N fit per cell.
theta_star_null <- fit_focal(gen(200000L, 0.0))$est
theta_star_mis  <- fit_focal(gen(200000L, cfg$cross))$est

qn <- qnorm(c(.025,.975))
one_rep <- function(cell, cross, seed, theta_star) {
  set.seed(seed)
  d <- gen(cfg$n_total, cross)
  n <- cfg$n_total
  base <- fit_focal(d, want_se = TRUE)
  if (is.null(base)) return(NULL)
  th <- base$est; se_o <- base$se_obs; se_e <- base$se_exp
  bo_est <- bo_t <- numeric(0)
  for (b in seq_len(cfg$boot)) {
    bf <- fit_focal(d[sample.int(n, replace=TRUE), , drop=FALSE], want_se = TRUE)
    if (is.null(bf)) next
    bo_est <- c(bo_est, bf$est); bo_t <- c(bo_t, (bf$est - th)/bf$se_obs)
  }
  if (length(bo_est) < 0.5*cfg$boot) return(NULL)
  ci_norm_o <- th + qn*se_o
  ci_norm_e <- th + qn*se_e
  ci_pct <- as.numeric(quantile(bo_est, c(.025,.975), names=FALSE))
  ci_stud <- th - rev(as.numeric(quantile(bo_t, c(.025,.975), names=FALSE)))*se_o
  ci_bca <- c(NA, NA)
  if (cfg$bca) {
    z0 <- qnorm(mean(bo_est < th))
    G <- min(50L, n); grp <- ((seq_len(n) - 1L) %% G) + 1L  # grouped (delete-d) jackknife
    jk <- vapply(seq_len(G), function(g){ f<-fit_focal(d[grp!=g,,drop=FALSE]); if(is.null(f)) NA_real_ else f$est }, numeric(1))
    jk <- jk[is.finite(jk)]; jbar <- mean(jk)
    a <- sum((jbar-jk)^3) / (6 * (sum((jbar-jk)^2))^1.5)
    if (is.finite(z0) && is.finite(a)) {
      adj <- function(z) pnorm(z0 + (z0+z)/(1 - a*(z0+z)))
      ci_bca <- as.numeric(quantile(bo_est, c(adj(qn[1]), adj(qn[2])), names=FALSE))
    }
  }
  cov <- function(ci) is.finite(ci[1]) && ci[1] <= theta_star && theta_star <= ci[2]
  data.frame(cell=cell, est=th, se_exp=se_e, se_obs=se_o, boot_se=sd(bo_est),
             cov_norm_exp=cov(ci_norm_e), cov_norm_obs=cov(ci_norm_o),
             cov_pct=cov(ci_pct), cov_bca=cov(ci_bca), cov_stud=cov(ci_stud),
             w_norm_obs=diff(ci_norm_o), w_pct=diff(ci_pct), w_bca=diff(ci_bca),
             w_stud=diff(ci_stud))
}

run_cell <- function(cell, cross, off, theta_star) {
  rows <- lapply(seq_len(cfg$reps), function(r) one_rep(cell, cross, cfg$seed_base+off+r, theta_star))
  D <- do.call(rbind, Filter(Negate(is.null), rows))
  data.frame(cell=cell, n_total=cfg$n_total, cross=cross, reps=nrow(D),
             theta_star=theta_star, emp_sd=sd(D$est),
             mean_se_exp=mean(D$se_exp), mean_se_obs=mean(D$se_obs), mean_boot_se=mean(D$boot_se),
             cov_norm_exp=mean(D$cov_norm_exp), cov_norm_obs=mean(D$cov_norm_obs),
             cov_pct=mean(D$cov_pct), cov_bca=mean(D$cov_bca), cov_stud=mean(D$cov_stud),
             stringsAsFactors=FALSE)
}

summary <- rbind(run_cell("null", 0.0, 0L, theta_star_null),
                 run_cell("misspec", cfg$cross, 2000000L, theta_star_mis))
write_csv(summary, file.path(res_dir, "bootstrap_coverage.csv"))
write_metadata(file.path(res_dir, "bootstrap_metadata.csv"),
  values = list(reps=cfg$reps, boot=cfg$boot, n_total=cfg$n_total, cross=cfg$cross,
                seed_base=cfg$seed_base, bca=cfg$bca,
                theta_star_null=theta_star_null, theta_star_misspec=theta_star_mis,
                estimator="DWLS_ordinal", focal="f2=~y5 raw loading",
                target="coverage of pseudo-true theta* (nominal 0.95)", smoke=cfg$smoke),
  packages="magmaan")
print(summary, row.names=FALSE, digits=4)
cat("\nWrote results to ", res_dir, "\n", sep="")
