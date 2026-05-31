#!/usr/bin/env Rscript
# Foldnes, Moss & Gronneberg (2024) — goodness-of-fit replication, magmaan.
#
# "Improved Goodness of Fit Procedures for Structural Equation Models",
# Structural Equation Modeling 32(1), 1-13.
#
# Two questions, one design (correctly-specified two-factor CFA, p in {10,20,40},
# data normal or non-normal):
#
#  (1) PROCEDURE PARITY. Does magmaan's penalized eigenvalue-block-averaging
#      family (pEBA / pOLS / SB / SS, x T_ML/T_RLS x biased/unbiased Gamma),
#      computed through fmg_pvalues(), reproduce the paper's own semTests
#      p-values fit-for-fit? This is the decisive correctness check.
#
#  (2) TYPE I ERROR. Under correct specification, do the recommended tests
#      (pEBA4_RLS, pOLS_RLS) hold the nominal 5% rejection rate where the
#      classical ML chi-square does not, as the model grows and the data turn
#      non-normal? A trimmed replication of the paper's rejection-rate tables.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--cells FILTER] [--seed-base S]
#                            [--semtests-parity] [--smoke]
#
# `--cells FILTER` restricts the grid, e.g. `p=10,N=400,dist=vm2`.
# `--semtests-parity` adds, on replicate 1 of each cell, a lavaan fit + a
#   semTests::pvalues() call compared against fmg_pvalues() (needs lavaan +
#   semTests). `--smoke` runs one tiny cell once.
#
# dist in {norm, ig1, ig2, pl1, pl2, vm1, vm2}; the "1" suffix is moderate
# non-normality (skew 2, kurt 7), "2" is severe (skew 3, kurt 21). IG and PL
# require covsim; norm and VM are self-contained.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "run_experiment.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "population.R"))

# The test family scored for rejection rates: classical baselines, the best
# traditional robust test (SB_UG_RLS), and the penalized recommendations.
TESTS <- c("std_ml", "std_rls", "sb_ug_rls", "ss_ug_rls",
           "peba2_rls", "peba4_rls", "peba6_rls", "pols2_rls",
           "peba4_ug_rls", "pols2_ug_rls")

parse_args <- function(args) {
  out <- list(reps = 5L, cells_filter = NULL, seed_base = 20240701L,
              semtests_parity = FALSE, smoke = FALSE)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--cells FILTER]",
          "[--seed-base S] [--semtests-parity] [--smoke]\n", sep = " ")
      cat("  --cells: comma-separated key=value over {p, N, dist}, e.g.",
          "p=10,N=400,dist=vm2\n", sep = " ")
      cat("  --semtests-parity: rep-1-per-cell magmaan-vs-semTests p-value check\n")
      cat("  dist in {norm, ig1, ig2, pl1, pl2, vm1, vm2}; *1=moderate, *2=severe\n")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(args[[i]])
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cells") { i <- i + 1L; out$cells_filter <- args[[i]]
    } else if (startsWith(a, "--cells=")) { out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(args[[i]])
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--semtests-parity") { out$semtests_parity <- TRUE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (!is.finite(out$reps) || out$reps < 1L) stop("--reps must be >= 1", call. = FALSE)
  out
}

parse_cells_filter <- function(s) {
  if (is.null(s) || !nzchar(s)) return(list())
  out <- list()
  for (p in strsplit(s, ",", fixed = TRUE)[[1L]]) {
    kv <- strsplit(p, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("bad --cells entry: ", p, call. = FALSE)
    out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
results_dir <- ensure_results_dir()
set_single_threaded_math()
require_pkg("magmaan")

# ── Cell grid ───────────────────────────────────────────────────────────────
# Default grid keeps norm + VM (self-contained) + IG (covsim, reliable). PL is
# reachable via `--cells dist=pl1|pl2` but kept out of the default run: covsim's
# rPLSIM is slow and often cannot fit the severe marginals.
cell_grid <- expand.grid(
  p    = c(10L, 20L, 40L),
  N    = c(400L, 800L),
  dist = c("norm", "vm1", "vm2", "ig2"),
  stringsAsFactors = FALSE)
if (isTRUE(args$smoke)) {
  cell_grid <- data.frame(p = 10L, N = 400L, dist = "vm2", stringsAsFactors = FALSE)
  args$reps <- 1L
}
for (key in names(parse_cells_filter(args$cells_filter))) {
  val <- parse_cells_filter(args$cells_filter)[[key]]
  if (!(key %in% names(cell_grid))) stop("--cells: unknown key '", key, "'", call. = FALSE)
  target <- if (is.numeric(cell_grid[[key]])) as.numeric(val) else val
  cell_grid <- cell_grid[cell_grid[[key]] == target, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)

needs_covsim <- any(dist_family(cell_grid$dist) %in% c("ig", "pl"))
if (needs_covsim && !requireNamespace("covsim", quietly = TRUE)) {
  stop("IG/PL cells need covsim; install it or restrict --cells to norm/vm*",
       call. = FALSE)
}

# ── Per-cell population + analysis spec (constant across reps) ───────────────
.spec_cache <- new.env(parent = emptyenv())
get_ctx <- function(p) {
  key <- as.character(p)
  if (!is.null(.spec_cache[[key]])) return(.spec_cache[[key]])
  ctx <- list(pop = build_population_2factor(p),
              spec = magmaan::model_spec(build_2factor_syntax(p)),
              syntax = build_2factor_syntax(p),
              varnames = paste0("x", seq_len(p)))
  .spec_cache[[key]] <- ctx
  ctx
}

fl_cache <- new.env(parent = emptyenv())
get_fl <- function(dist) {
  if (dist_family(dist) != "vm") return(NULL)
  if (!is.null(fl_cache[[dist]])) return(fl_cache[[dist]])
  mom <- dist_moments[[dist]]
  fl <- fleishman_coef(mom[[1L]], mom[[2L]])
  fl_cache[[dist]] <- fl
  fl
}

fit_magmaan <- function(ctx, X) {
  df_X <- as.data.frame(X)
  fit <- tryCatch(magmaan::magmaan(ctx$syntax, df_X, estimator = "ML"),
                  error = function(e) e)
  if (inherits(fit, "error")) return(list(ok = FALSE, error = conditionMessage(fit)))
  if (!isTRUE(fit$converged)) return(list(ok = FALSE, error = "not converged"))
  pv <- tryCatch(magmaan::fmg_pvalues(fit, df_X, tests = TESTS),
                 error = function(e) e)
  if (inherits(pv, "error")) return(list(ok = FALSE, error = conditionMessage(pv)))
  list(ok = TRUE, fit = fit, pvalues = pv, data = df_X)
}

# ── Run ─────────────────────────────────────────────────────────────────────
cat(sprintf("foldnes-moss-gronneberg-2024: magmaan %s, %s\n",
            as.character(utils::packageVersion("magmaan")), R.version.string))
cat(sprintf("  reps=%d, cells=%d%s\n", args$reps, nrow(cell_grid),
            if (isTRUE(args$semtests_parity)) ", semTests parity ON" else ""))

pval_rows <- list(); meta_rows <- list(); parity_rows <- list()
t0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  ctx <- get_ctx(cell$p)
  sampler <- make_cell_sampler(ctx$pop, cell$N, cell$dist, args$reps,
                               seed_base = args$seed_base + ci * 100000L,
                               fl = get_fl(cell$dist))
  cat(sprintf("  cell %2d/%2d: p=%3d N=%4d dist=%-4s ", ci, nrow(cell_grid),
              cell$p, cell$N, cell$dist))
  ok <- 0L; fail <- 0L; tc0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    X <- tryCatch(sampler(rep_idx), error = function(e) e)
    if (inherits(X, "error")) { fail <- fail + 1L; next }
    colnames(X) <- ctx$varnames
    res <- fit_magmaan(ctx, X)
    if (!isTRUE(res$ok)) { fail <- fail + 1L; next }
    ok <- ok + 1L
    pval_rows[[length(pval_rows) + 1L]] <- data.frame(
      cell_idx = ci, rep = rep_idx, p = cell$p, N = cell$N, dist = cell$dist,
      test = names(res$pvalues), p_value = as.numeric(res$pvalues),
      stringsAsFactors = FALSE)

    if (isTRUE(args$semtests_parity) && rep_idx == 1L &&
        requireNamespace("lavaan", quietly = TRUE) &&
        requireNamespace("semTests", quietly = TRUE)) {
      lv <- tryCatch(lavaan::cfa(ctx$syntax, data = res$data, estimator = "ML"),
                     error = function(e) e)
      if (!inherits(lv, "error")) {
        st <- tryCatch(as.numeric(semTests::pvalues(lv, tests = TESTS)),
                       error = function(e) rep(NA_real_, length(TESTS)))
        parity_rows[[length(parity_rows) + 1L]] <- data.frame(
          cell_idx = ci, p = cell$p, N = cell$N, dist = cell$dist,
          test = TESTS, magmaan = as.numeric(res$pvalues), semtests = st,
          abs_diff = abs(as.numeric(res$pvalues) - st), stringsAsFactors = FALSE)
      }
    }
  }
  tc1 <- proc.time()[["elapsed"]]
  cat(sprintf("ok=%d fail=%d (%.1fs)\n", ok, fail, tc1 - tc0))
  meta_rows[[ci]] <- data.frame(cell_idx = ci, p = cell$p, N = cell$N,
                                dist = cell$dist, rep_ok = ok, rep_fail = fail,
                                seconds = tc1 - tc0, stringsAsFactors = FALSE)
}
t1 <- proc.time()[["elapsed"]]

write_rows(pval_rows, file.path(results_dir, "pvalues.csv"))
write_rows(meta_rows, file.path(results_dir, "cell_meta.csv"))
utils::write.csv(cell_grid, file.path(results_dir, "cells.csv"), row.names = FALSE)

# ── Rejection-rate summary (alpha = .05) ────────────────────────────────────
pv_long <- if (length(pval_rows)) do.call(rbind, pval_rows) else NULL
if (!is.null(pv_long) && nrow(pv_long)) {
  grp <- split(pv_long, list(pv_long$cell_idx, pv_long$test), drop = TRUE)
  summary_rr <- do.call(rbind, lapply(grp, function(g) data.frame(
    cell_idx = g$cell_idx[[1L]], p = g$p[[1L]], N = g$N[[1L]], dist = g$dist[[1L]],
    test = g$test[[1L]], n_reps = nrow(g),
    reject_05 = mean(g$p_value < 0.05, na.rm = TRUE), stringsAsFactors = FALSE)))
  utils::write.csv(summary_rr, file.path(results_dir, "rejection_rates.csv"),
                   row.names = FALSE)
}

if (length(parity_rows)) {
  parity_df <- do.call(rbind, parity_rows)
  utils::write.csv(parity_df, file.path(results_dir, "semtests_parity.csv"),
                   row.names = FALSE)
  comp <- parity_df[is.finite(parity_df$abs_diff), ]
  cat(sprintf("semTests parity: %d rows; max |Δp| = %.3g\n",
              nrow(parity_df), if (nrow(comp)) max(comp$abs_diff) else NA_real_))
}

write_csv(metadata_frame(
  values = list(reps = args$reps, seed_base = args$seed_base,
                cells_filter = args$cells_filter,
                semtests_parity = isTRUE(args$semtests_parity),
                n_cells = nrow(cell_grid),
                total_seconds = sprintf("%.2f", t1 - t0)),
  packages = c("magmaan", "lavaan", "semTests", "covsim")),
  file.path(results_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1fs — results in %s\n", t1 - t0, results_dir))
