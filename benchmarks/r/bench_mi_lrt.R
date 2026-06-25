#!/usr/bin/env Rscript
# bench_mi_lrt.R — timing the LRT (refit-based) modification-index sweep against
# the one-step score-test sweep, for cross-group loading-invariance releases.
#
# This is the Milestone-1 cost gate for LRT-based modification indices (see
# docs/research/notes/lrt-modification-indices.tex). The LRT sweep refits one
# released model per candidate constraint and runs a nested Satorra-2000
# difference test; the one-step sweep (`score_tests`) evaluates a Lagrange-
# multiplier statistic at the constrained fit in a single fit. We time both
# across model size (n_items -> npar, #candidates) and sample size.
#
# This R harness measures the COLD-START refit cost (magmaan() computes its own
# start values). The warm-started refit (start each released fit from the
# constrained MLE) is the C++-core optimization whose speed-up this baseline is
# meant to quantify against, once that path exists.
#
# Usage:
#   Rscript benchmarks/r/bench_mi_lrt.R              # default grid
#   Rscript benchmarks/r/bench_mi_lrt.R 6,9,12 1000  # items grid, n_total

suppressMessages(library(magmaan))
source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

# ---- synthetic two-group one-factor CFA ------------------------------------
# loadings 0.6, unit-variance indicators, one planted non-invariant loading in
# the smaller group so the sweep has a real signal to localize.
gen_group <- function(n, loadings) {
  p <- length(loadings)
  f <- rnorm(n)
  rs <- sqrt(pmax(1 - loadings^2, 1e-6))
  X <- sapply(seq_len(p), function(j) loadings[j] * f + rnorm(n, 0, rs[j]))
  colnames(X) <- paste0("x", seq_len(p))
  as.data.frame(X)
}

make_data <- function(n_items, n_total, ratio = 0.25, seed = 1L) {
  set.seed(seed)
  nA <- max(round(n_total * ratio), 50L)
  nB <- n_total - nA
  loadA <- rep(0.6, n_items); loadA[3] <- 0.9      # x3 non-invariant in group A
  dA <- gen_group(nA, loadA); dA$g <- "A"
  dB <- gen_group(nB, rep(0.6, n_items)); dB$g <- "B"
  rbind(dA, dB)
}

cfa_syntax <- function(n_items) {
  paste0("f =~ ", paste0("x", seq_len(n_items), collapse = " + "))
}

# median elapsed seconds over `reps` runs of expr()
time_median <- function(expr, reps = 3L) {
  t <- vapply(seq_len(reps), function(i) system.time(expr())[["elapsed"]],
              numeric(1))
  stats::median(t)
}

# one cell of the sweep
bench_cell <- function(n_items, n_total, reps = 3L) {
  dat <- make_data(n_items, n_total)
  syn <- cfa_syntax(n_items)

  fit_anchor <- function()
    magmaan(syn, dat, estimator = "ML", groups = "g", group_equal = "loadings")
  anchor <- fit_anchor()

  ov <- if (is.list(anchor$ov_names)) anchor$ov_names[[1]] else anchor$ov_names
  glabs <- anchor$group_labels
  data_list <- lapply(glabs, function(g)
    as.matrix(dat[dat$g == g, ov, drop = FALSE]))

  # candidates: the non-marker loadings (x2..x_p); x1 is the fixed marker.
  cand_items <- paste0("f=~x", 2:n_items)

  # one-step arm: equality-release score tests at the constrained fit.
  t_onestep <- time_median(function() score_tests(anchor), reps)

  # LRT arm: per candidate, cold refit the released model + nested difference.
  refit_fevals <- numeric(0)
  lrt_once <- function() {
    for (tok in cand_items) {
      rel <- magmaan(syn, dat, estimator = "ML", groups = "g",
                     group_equal = "loadings", group_partial = tok)
      robust_nested_lrt(rel, anchor, data = data_list, method = "restriction_map")
    }
  }
  t_lrt_total <- time_median(lrt_once, reps)

  # one extra pass to record optimizer work per released refit
  for (tok in cand_items) {
    rel <- magmaan(syn, dat, estimator = "ML", groups = "g",
                   group_equal = "loadings", group_partial = tok)
    refit_fevals <- c(refit_fevals, rel$f_evals %||% NA_real_)
  }

  list(
    n_items        = n_items,
    n_total        = n_total,
    npar           = anchor$npar,
    n_candidates   = length(cand_items),
    anchor_fevals  = anchor$f_evals %||% NA_real_,
    refit_fevals_mean = mean(refit_fevals, na.rm = TRUE),
    t_anchor_fit   = time_median(fit_anchor, reps),
    t_onestep      = t_onestep,
    t_lrt_total    = t_lrt_total,
    t_lrt_per_cand = t_lrt_total / length(cand_items),
    lrt_over_onestep = t_lrt_total / t_onestep
  )
}

# ---- driver ----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
items_grid <- if (length(args) >= 1)
  as.integer(strsplit(args[[1]], ",")[[1]]) else c(6L, 9L, 12L, 15L)
n_total <- if (length(args) >= 2) as.integer(args[[2]]) else 1000L

cat(sprintf("LRT-MI timing gate | items={%s} n_total=%d\n",
            paste(items_grid, collapse = ","), n_total))
cat(strrep("-", 92), "\n")
cat(sprintf("%6s %6s %6s %12s %12s %14s %12s %10s\n",
            "items", "npar", "cand", "t_onestep_s", "t_lrt_s",
            "t_per_cand_s", "lrt/onestep", "refit_fev"))

rows <- lapply(items_grid, function(ni) {
  r <- bench_cell(ni, n_total)
  cat(sprintf("%6d %6d %6d %12.4f %12.4f %14.5f %12.1f %10.0f\n",
              r$n_items, r$npar, r$n_candidates, r$t_onestep, r$t_lrt_total,
              r$t_lrt_per_cand, r$lrt_over_onestep, r$refit_fevals_mean))
  r
})

result <- list(
  benchmark   = "mi_lrt",
  start       = "cold",
  n_total     = n_total,
  created_at  = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  magmaan_version = safe_package_version("magmaan"),
  cells       = rows
)
out_path <- file.path(bench_root(), "results",
  sprintf("mi_lrt_%s.json", format(Sys.time(), "%Y%m%dT%H%M%S", tz = "UTC")))
write_json_file(result, out_path)
cat(strrep("-", 92), "\n")
cat("wrote", out_path, "\n")
