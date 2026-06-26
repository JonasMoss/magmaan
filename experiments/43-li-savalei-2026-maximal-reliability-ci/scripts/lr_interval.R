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
# percentile of LR (which would be 3.841 if chi^2_1 held exactly). Normal data only:
# under non-normality the normal-theory LR is mis-scaled (Satorra-Bentler), a
# separate correction; see report.
#
# Usage:
#   Rscript scripts/lr_interval.R [--reps 400] [--ns 50,100,200]
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
  o <- list(reps = 400L, ns = c(50L, 100L, 200L), seed_base = 20260701L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (startsWith(x, "--reps=")) o$reps <- as.integer(sub("^--reps=", "", x))
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (startsWith(x, "--ns=")) o$ns <- as.integer(strsplit(sub("^--ns=", "", x), ",")[[1L]])
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
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

# LR(rho*_true) for both coefficients on one dataset, or NULL.
lr_rep <- function(dat, n) {
  S <- stats::cov(dat) * (n - 1) / n
  logdetS <- as.numeric(determinant(S, logarithm = TRUE)$modulus)
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"), error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  m <- make_rebuilder(fit$partable, pop)(fit$partable$est)
  if (any(m$P <= 0)) return(NULL)                          # same admissible subset
  z0 <- ml$zeta_from(m$L, m$P)
  unc <- ml$fit_unc(S, logdetS, z0)
  out <- c(gen = NA_real_, grp = NA_real_)
  for (w in c("gen", "grp")) {
    con <- ml$fit_con(S, logdetS, unc$z, w, tgt[[w]])
    if (!is.null(con)) out[w] <- max(0, n * (con$F - unc$F))
  }
  out
}

one_cell <- function(n, seed0) {
  LR <- list(gen = numeric(0), grp = numeric(0))
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    v <- lr_rep(gen_data(pop, n, "normal"), n)
    if (is.null(v)) next
    for (w in c("gen", "grp")) if (is.finite(v[[w]])) LR[[w]] <- c(LR[[w]], v[[w]])
  }
  do.call(rbind, lapply(c("gen", "grp"), function(w) {
    x <- LR[[w]]; c_bart <- mean(x)
    data.frame(p = pop$p, coef = w, dist = "normal", n = n, reps_ok = length(x),
      rho_pop = tgt[[w]], mean_LR = c_bart, q95_LR = stats::quantile(x, 0.95, names = FALSE),
      cov_chi2 = mean(x <= q1), cov_bartlett = mean(x <= q1 * c_bart),
      stringsAsFactors = FALSE)
  }))
}

all_rows <- do.call(rbind, lapply(seq_along(cfg$ns), function(i) {
  n <- cfg$ns[i]; message(sprintf("cell %d/%d: n=%d (reps=%d)", i, length(cfg$ns), n, cfg$reps))
  one_cell(n, cfg$seed_base + i * 100000L)
}))

write_csv(all_rows, file.path(res_dir, "lr_coverage.csv"))
write_metadata(
  file.path(res_dir, "lr_metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","), seed_base = cfg$seed_base,
    model = "orthogonal std.lv bifactor, 6 indicators/subscale (p18), normal data",
    statistic = "profile LR = N (F_con(rho*_true) - F_unc), chi^2_1 reference",
    bartlett = "threshold q*c, c = cell-mean LR (empirical Bartlett factor, df=1)",
    smoke = cfg$smoke),
  packages = "magmaan, nloptr")
cat("\nProfile-LR coverage (nominal 0.95):\n")
print(all_rows[order(all_rows$coef, all_rows$n),
               c("coef","n","reps_ok","mean_LR","q95_LR","cov_chi2","cov_bartlett")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "lr_coverage.csv"), "\n", sep = "")
