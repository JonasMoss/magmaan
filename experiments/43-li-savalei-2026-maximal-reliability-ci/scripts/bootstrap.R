#!/usr/bin/env Rscript
# Bootstrap-percentile reference for the maximal-reliability CIs. The separate,
# heavier arm (cf. run_experiment.R's analytic delta methods): per replicate, fit
# the bifactor model once for the point rho*, then resample rows B times, refit,
# and take the 2.5/97.5 percentiles of rho*_b. This is the distribution-free
# gold standard the analytic intervals are judged against; it should approach
# nominal coverage where the delta method fails (low rho*, small N, non-normal).
#
# Usage:
#   Rscript scripts/bootstrap.R [--reps 200] [--B 299] [--ns 50,200]
#                               [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/bootstrap.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/bootstrap.R", mustWork = FALSE)
  dirname(dirname(sp))
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))

parse_args <- function(a) {
  o <- list(reps = 150L, B = 199L, ns = c(50L, 200L),
            pkeys = c("p9", "p18"), seed_base = 20260627L, smoke = FALSE)
  i <- 1L
  while (i <= length(a)) {
    x <- a[[i]]
    if (x %in% c("-h", "--help")) { cat("see header\n"); quit(save = "no") }
    else if (x == "--reps") { i <- i + 1L; o$reps <- as.integer(a[[i]]) }
    else if (startsWith(x, "--reps=")) o$reps <- as.integer(sub("^--reps=", "", x))
    else if (x == "--B") { i <- i + 1L; o$B <- as.integer(a[[i]]) }
    else if (startsWith(x, "--B=")) o$B <- as.integer(sub("^--B=", "", x))
    else if (x == "--ns") { i <- i + 1L; o$ns <- as.integer(strsplit(a[[i]], ",")[[1L]]) }
    else if (startsWith(x, "--ns=")) o$ns <- as.integer(strsplit(sub("^--ns=", "", x), ",")[[1L]])
    # --pkeys restricts the model sizes (p9, p18); the small-N bifactor bootstrap
    # barely converges at p9, so a focused p18 run is the usable reference.
    else if (x == "--pkeys") { i <- i + 1L; o$pkeys <- strsplit(a[[i]], ",")[[1L]] }
    else if (startsWith(x, "--pkeys=")) o$pkeys <- strsplit(sub("^--pkeys=", "", x), ",")[[1L]]
    else if (x == "--seed-base") { i <- i + 1L; o$seed_base <- as.integer(a[[i]]) }
    else if (x == "--smoke") o$smoke <- TRUE
    else stop("unknown argument: ", x, call. = FALSE)
    i <- i + 1L
  }
  if (o$smoke) { o$reps <- 20L; o$B <- 99L; o$ns <- c(100L) }
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

# Point rho* (gen, grp) for one dataset; NULL on non-convergence/Heywood.
point_rho <- function(spec, dat, pop) {
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  rebuild <- make_rebuilder(fit$partable, pop)
  m <- rebuild(fit$partable$est)
  if (any(m$P <= 0)) return(NULL)
  c(gen = rho_olc(m$Sigma, m$L[, "G"]),
    grp = rho_olsc(m$Sigma, m$L[, "g1"], pop$items[[1]]))
}

boot_ci <- function(spec, dat, pop, B) {
  n <- nrow(dat)
  bs <- matrix(NA_real_, B, 2, dimnames = list(NULL, c("gen", "grp")))
  for (b in seq_len(B)) {
    db <- dat[sample.int(n, n, replace = TRUE), , drop = FALSE]
    pr <- point_rho(spec, db, pop)
    if (!is.null(pr)) bs[b, ] <- pr
  }
  apply(bs, 2, function(v) {
    v <- v[is.finite(v)]
    if (length(v) < B / 2) return(c(NA, NA))     # too many failed resamples
    stats::quantile(v, c(0.025, 0.975), names = FALSE)
  })
}

methods <- c("boot_perc", "boot_basic")

one_cell <- function(pkey, n, dist, seed0) {
  pop <- pops[[pkey]]; spec <- specs[[pkey]]
  tgt <- c(gen = pop$rho_gen, grp = pop$rho_grp1)
  acc <- list(); for (coef in c("gen","grp")) for (m in methods)
    acc[[paste(coef, m, sep = ".")]] <- list(cov = 0L, width = 0)
  ok <- 0L
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    dat <- gen_data(pop, n, dist)
    pr <- point_rho(spec, dat, pop)
    if (is.null(pr)) next
    ci <- boot_ci(spec, dat, pop, cfg$B)         # 2 x 2: rows q025/q975, cols coef
    if (any(is.na(ci))) next
    ok <- ok + 1L
    for (coef in c("gen", "grp")) {
      qlo <- ci[1, coef]; qhi <- ci[2, coef]
      # percentile, and the basic/pivotal interval that reflects through the
      # point estimate (so an upward-biased rho-hat shifts the interval DOWN).
      intervals <- list(
        boot_perc  = c(qlo, qhi),
        boot_basic = c(2 * pr[[coef]] - qhi, 2 * pr[[coef]] - qlo))
      for (m in methods) {
        lo <- intervals[[m]][1]; hi <- intervals[[m]][2]
        a <- acc[[paste(coef, m, sep = ".")]]
        a$cov <- a$cov + as.integer(lo <= tgt[[coef]] && tgt[[coef]] <= hi)
        a$width <- a$width + (hi - lo)
        acc[[paste(coef, m, sep = ".")]] <- a
      }
    }
  }
  rows <- list()
  for (coef in c("gen", "grp")) for (m in methods) {
    a <- acc[[paste(coef, m, sep = ".")]]
    rows[[length(rows) + 1L]] <- data.frame(
      p = pop$p, coef = coef, dist = dist, n = n, method = m,
      reps_ok = ok, rho_pop = tgt[[coef]],
      coverage = a$cov / ok, mean_width = a$width / ok,
      stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

grid <- expand.grid(pkey = names(pops), n = cfg$ns, dist = dists,
                    stringsAsFactors = FALSE)
all_rows <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]
  message(sprintf("boot cell %d/%d: %s n=%d %s (reps=%d B=%d)",
                  i, nrow(grid), g$pkey, g$n, g$dist, cfg$reps, cfg$B))
  one_cell(g$pkey, g$n, g$dist, cfg$seed_base + i * 100000L)
}))

write_csv(all_rows, file.path(res_dir, "bootstrap.csv"))
cat("\nBootstrap-percentile reference:\n")
print(all_rows[, c("p", "coef", "dist", "n", "coverage", "mean_width")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "bootstrap.csv"), "\n", sep = "")
