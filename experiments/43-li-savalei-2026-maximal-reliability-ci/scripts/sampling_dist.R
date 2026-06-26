#!/usr/bin/env Rscript
# Sampling distribution of the maximal-reliability estimate at small N, on three
# scales, to see which geometry is closest to normal (and hence the right scale
# for an interval). Maximal reliability is a squared correlation, rho* =
# corr^2(f, X_c), so the natural variance-stabilizing transform is Fisher's z on
# the correlation sqrt(rho*) (the factor determinacy), atanh(sqrt(rho*)) - not the
# logit on rho* itself, which is what the interval sweeps use. We collect rho-hat
# over many replicates and report skewness on the raw, logit, and Fisher scales.
#
# Usage:
#   Rscript scripts/sampling_dist.R [--reps 2000] [--ns 50,200]
#                                   [--pkeys p9,p18] [--seed-base S] [--smoke]

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/sampling_dist.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(sp))), "_support", "R", "helpers.R")
}
.exp_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE); fa <- grep("^--file=", a, value = TRUE)
  sp <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
        else normalizePath("scripts/sampling_dist.R", mustWork = FALSE)
  dirname(dirname(sp))
}
source(.support_helpers())
exp_dir <- .exp_dir()
source(file.path(exp_dir, "R", "maxrel.R"))

parse_args <- function(a) {
  o <- list(reps = 2000L, ns = c(50L, 200L), pkeys = c("p9", "p18"),
            seed_base = 20260629L, smoke = FALSE)
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
  if (o$smoke) { o$reps <- 300L; o$ns <- c(50L); o$pkeys <- c("p18") }
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

clamp01 <- function(x, eps = 1e-6) pmin(1 - eps, pmax(eps, x))
skewness <- function(x) { x <- x[is.finite(x)]; m <- mean(x)
  mean((x - m)^3) / (mean((x - m)^2))^1.5 }

point_rho <- function(spec, dat, pop) {
  fit <- tryCatch(magmaan::magmaan(spec, dat, estimator = "ML"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  rb <- make_rebuilder(fit$partable, pop); m <- rb(fit$partable$est)
  if (any(m$P <= 0)) return(NULL)
  c(gen = rho_olc(m$Sigma, m$L[, "G"]),
    grp = rho_olsc(m$Sigma, m$L[, "g1"], pop$items[[1]]))
}

draws <- list(); summ <- list()
grid <- expand.grid(pkey = names(pops), n = cfg$ns, dist = dists,
                    stringsAsFactors = FALSE)
for (i in seq_len(nrow(grid))) {
  gg <- grid[i, ]; pop <- pops[[gg$pkey]]; spec <- specs[[gg$pkey]]
  message(sprintf("cell %d/%d: %s n=%d %s (reps=%d)",
                  i, nrow(grid), gg$pkey, gg$n, gg$dist, cfg$reps))
  seed0 <- cfg$seed_base + i * 100000L
  rows <- vector("list", cfg$reps)
  for (r in seq_len(cfg$reps)) {
    set.seed(seed0 + r)
    pr <- point_rho(spec, gen_data(pop, gg$n, gg$dist), pop)
    if (!is.null(pr)) rows[[r]] <- data.frame(
      p = pop$p, dist = gg$dist, n = gg$n,
      gen = pr[["gen"]], grp = pr[["grp"]])
  }
  D <- do.call(rbind, rows)
  for (coef in c("gen", "grp")) {
    v <- D[[coef]]; tgt <- if (coef == "gen") pop$rho_gen else pop$rho_grp1
    draws[[length(draws) + 1L]] <- data.frame(
      p = pop$p, dist = gg$dist, n = gg$n, coef = coef, rho = v)
    summ[[length(summ) + 1L]] <- data.frame(
      p = pop$p, dist = gg$dist, n = gg$n, coef = coef, reps_ok = length(v),
      rho_pop = tgt, mean = mean(v),
      skew_raw    = skewness(v),
      skew_logit  = skewness(stats::qlogis(clamp01(v))),
      skew_fisher = skewness(atanh(sqrt(clamp01(v)))),
      stringsAsFactors = FALSE)
  }
}

write_csv(do.call(rbind, draws), file.path(res_dir, "sampling_draws.csv"))
S <- do.call(rbind, summ)
write_csv(S, file.path(res_dir, "sampling_skew.csv"))
cat("\nSkewness of rho-hat by scale (closer to 0 = more symmetric):\n")
print(S[order(S$coef, S$p, S$dist, S$n),
        c("coef","p","dist","n","mean","skew_raw","skew_logit","skew_fisher")],
      row.names = FALSE, digits = 3)
cat("\nWrote ", file.path(res_dir, "sampling_draws.csv"), " and sampling_skew.csv\n", sep = "")
