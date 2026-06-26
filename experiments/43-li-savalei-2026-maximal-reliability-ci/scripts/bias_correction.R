#!/usr/bin/env Rscript
# After-the-fact bias correction for bifactor maximal reliability. run_experiment.R
# shows the binding error is the upward bias of rho-hat, not the SE. The bias is
# corrected on the FUNCTIONAL, not the parameters: rho* = g(theta) is strongly
# nonlinear, so the dominant O(1/n) bias is the curvature term (1/2) tr(g'' V)
# (Jensen), which parameter-level bias reduction would leave untouched. We correct
# rho-hat directly with the analytic second-order term and compare:
#   uncorrected     g(theta-hat)
#   second_order    g(theta-hat) - (1/2) tr(g'' V)
# Each gets a logit-scale interval with the robust SE, recentred on the corrected
# point (correction does not change the first-order variance). The bootstrap bias
# correction (2 rho-hat - mean over resamples) is the separate scripts/bootstrap.R
# (its basic/pivotal interval). We score coverage of the population rho*, the
# residual point bias, and width.
#
# Usage:
#   Rscript scripts/bias_correction.R [--reps 500] [--ns 50,100,200,500]
#                                     [--pkeys p9,p18] [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/bias_correction.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/bias_correction.R", mustWork = FALSE)
  dirname(dirname(sp))
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))

parse_args <- function(a) {
  o <- list(reps = 500L, ns = c(50L, 100L, 200L, 500L), pkeys = c("p9", "p18"),
            seed_base = 20260628L, smoke = FALSE)
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
methods <- c("uncorrected", "second_order")

clamp01 <- function(x, eps = 1e-4) pmin(1 - eps, pmax(eps, x))
ci_logit_pt <- function(point, se) {
  p <- clamp01(point); eta <- stats::qlogis(p); se_eta <- se / (p * (1 - p))
  stats::plogis(c(eta - z975 * se_eta, eta + z975 * se_eta))
}

# One replicate: per coefficient, uncorrected and second-order-corrected points
# plus the robust delta SE. NULL on a failed/improper fit.
one_rep <- function(spec, dat, pop) {
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  pt <- fit$partable
  rebuild <- make_rebuilder(pt, pop)
  if (any(rebuild(pt$est)$P <= 0)) return(NULL)
  Vr <- tryCatch(stats::vcov(fit, regime = "robust", data = dat),
                 error = function(e) NULL)
  if (!is.matrix(Vr)) return(NULL)
  lapply(c(gen = "gen", grp = "grp"), function(w) {
    rho  <- rho_of_theta(pt$est, rebuild, pop, w)
    cb   <- curvature_bias(pt$est, pt, rebuild, pop, w, Vr)
    grad <- grad_rho(pt$est, pt, rebuild, pop, w)
    se   <- sqrt(max(0, as.numeric(crossprod(grad, Vr %*% grad))))
    c(uncorrected = rho, second_order = rho - cb, curv = cb, se = se)
  })
}

one_cell <- function(pkey, n, dist, seed0) {
  pop <- pops[[pkey]]; spec <- specs[[pkey]]
  tgt <- c(gen = pop$rho_gen, grp = pop$rho_grp1)
  acc <- list(); for (coef in c("gen","grp")) for (m in methods)
    acc[[paste(coef,m,sep=".")]] <- list(cov = 0L, width = 0, est = 0)
  curv_sum <- c(gen = 0, grp = 0); ok <- 0L
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    res <- one_rep(spec, gen_data(pop, n, dist), pop)
    if (is.null(res)) next
    ok <- ok + 1L
    for (coef in c("gen", "grp")) {
      v <- res[[coef]]; se <- v[["se"]]; curv_sum[coef] <- curv_sum[coef] + v[["curv"]]
      for (m in methods) {
        pe <- v[[m]]; ci <- ci_logit_pt(pe, se)
        a <- acc[[paste(coef,m,sep=".")]]
        a$cov <- a$cov + as.integer(ci[1] <= tgt[[coef]] && tgt[[coef]] <= ci[2])
        a$width <- a$width + (ci[2] - ci[1]); a$est <- a$est + pe
        acc[[paste(coef,m,sep=".")]] <- a
      }
    }
  }
  rows <- list()
  for (coef in c("gen","grp")) for (m in methods) {
    a <- acc[[paste(coef,m,sep=".")]]
    rows[[length(rows)+1L]] <- data.frame(
      p = pop$p, coef = coef, dist = dist, n = n, method = m, reps_ok = ok,
      rho_pop = tgt[[coef]], mean_est = a$est / ok, bias = a$est / ok - tgt[[coef]],
      mean_curv = curv_sum[coef] / ok,
      coverage = a$cov / ok, mean_width = a$width / ok, stringsAsFactors = FALSE)
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

write_csv(all_rows, file.path(res_dir, "correction.csv"))
write_metadata(
  file.path(res_dir, "correction_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
                pkeys = paste(cfg$pkeys, collapse = ","), seed_base = cfg$seed_base,
                correction = "analytic second-order: rho-hat - 0.5 tr(g'' V_robust)",
                interval = "logit scale, robust SE, recentred on the corrected point",
                smoke = cfg$smoke),
  packages = "magmaan")
cat("\nResidual bias and coverage by correction:\n")
print(all_rows[order(all_rows$coef, all_rows$p, all_rows$dist, all_rows$n,
                     match(all_rows$method, methods)),
               c("coef","p","dist","n","method","bias","coverage","mean_width")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "correction.csv"), "\n", sep = "")
