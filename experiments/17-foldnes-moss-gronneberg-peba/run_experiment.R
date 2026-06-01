#!/usr/bin/env Rscript
# Foldnes, Moss & Gronneberg (2024) — goodness-of-fit replication, magmaan.
#
# "Improved Goodness of Fit Procedures for Structural Equation Models",
# Structural Equation Modeling 32(1), 1-13.
#
# Two questions, one design (correctly-specified two-factor CFA, p in {10,20,40},
# data normal or non-normal):
#
#  (1) PROCEDURE PARITY. Does magmaan's eigenvalue-block-averaging family
#      (SB / SS / EBA / pEBA / pOLS / EBAd, x T_ML/T_RLS x biased/unbiased Gamma),
#      computed through fmg_pvalues(), reproduce the paper's own semTests
#      p-values fit-for-fit? This is the decisive correctness check.
#
#  (2) TYPE I ERROR. Under correct specification, do the robustified tests hold
#      the nominal 5% rejection rate where the classical ML chi-square does not,
#      as the model grows and the data turn non-normal? The full 84-condition
#      grid and the paper's 42-statistic battery; only the replicate count is
#      scaled down (--reps) from the paper's 3000. The Bollen-Stine bootstrap,
#      the paper's 43rd statistic, is the one test magmaan does not implement.
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
# non-normality (skew 2, kurt 7), "2" is severe (skew 3, kurt 21). All families
# are self-contained: VM via the Fleishman power method, IG/PL via magmaan's
# native simulators (no covsim).

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

# The paper's full test battery: 42 statistics = its 43 minus the Bollen-Stine
# bootstrap (the one family magmaan does not implement). Two normal-theory base
# tests (std) plus ten Gamma-dependent families, each crossed over base statistic
# (ML/RLS) x Gamma estimator (biased / unbiased "ug"): SB, SS, EBA2/4/6,
# pEBA2/4/6, pOLS (gamma=2), and ALL (= EBAd, singleton blocks).
.fmg_full_battery <- function() {
  fams <- c("sb", "ss", "eba2", "eba4", "eba6",
            "peba2", "peba4", "peba6", "pols2", "all")
  g <- expand.grid(fam = fams, ug = c("", "ug"), base = c("ml", "rls"),
                   stringsAsFactors = FALSE)
  gtests <- vapply(seq_len(nrow(g)), function(i)
    paste(c(g$fam[[i]], if (nzchar(g$ug[[i]])) g$ug[[i]], g$base[[i]]),
          collapse = "_"), character(1))
  c("std_ml", "std_rls", sort(gtests))
}
TESTS <- .fmg_full_battery()

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
core <- magmaan::magmaan_core

# Marginal moments per distribution label (moderate = *1: skew 2, kurt 7;
# severe = *2: skew 3, kurt 21). Passed to the sampler for the IG/PL families
# and used to solve the Fleishman coefficients for VM.
dist_moments <- list(
  norm = c(0, 0),
  vm1  = c(2, 7),  vm2 = c(3, 21),
  ig1  = c(2, 7),  ig2 = c(3, 21),
  pl1  = c(2, 7),  pl2 = c(3, 21))

# ── Cell grid ───────────────────────────────────────────────────────────────
# Full paper grid (3 x 4 x 7 = 84 conditions): p in {10,20,40}, the four sample
# sizes N in {400,800,1500,3000}, and norm + all six non-normal cells (VM via the
# self-contained Fleishman path, IG/PL via magmaan's native simulators, no
# covsim). The p=40 / N=3000 cells are heavy; scale with --reps and --cells.
cell_grid <- expand.grid(
  p    = c(10L, 20L, 40L),
  N    = c(400L, 800L, 1500L, 3000L),
  dist = c("norm", "vm1", "vm2", "ig1", "ig2", "pl1", "pl2"),
  stringsAsFactors = FALSE)
if (isTRUE(args$smoke)) {
  cell_grid <- data.frame(p = 10L, N = 400L, dist = "ig2", stringsAsFactors = FALSE)
  args$reps <- 1L
}
for (key in names(parse_cells_filter(args$cells_filter))) {
  val <- parse_cells_filter(args$cells_filter)[[key]]
  if (!(key %in% names(cell_grid))) stop("--cells: unknown key '", key, "'", call. = FALSE)
  target <- if (is.numeric(cell_grid[[key]])) as.numeric(val) else val
  cell_grid <- cell_grid[cell_grid[[key]] == target, , drop = FALSE]
}
if (!nrow(cell_grid)) stop("no cells selected", call. = FALSE)

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
# IG/PL calibration is population-only (no N), so reuse it across sample sizes
# sharing the same (family, p, dist).
.sim_cal_cache <- new.env(parent = emptyenv())
t0 <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  ctx <- get_ctx(cell$p)
  fam <- dist_family(cell$dist)
  cal_key <- if (fam %in% c("ig", "pl")) paste(fam, cell$p, cell$dist, sep = ":") else NULL
  cached_cal <- if (!is.null(cal_key) && exists(cal_key, envir = .sim_cal_cache,
                                                inherits = FALSE)) {
    get(cal_key, envir = .sim_cal_cache)
  } else NULL
  sampler <- make_cell_sampler(ctx$pop, cell$N, cell$dist, args$reps,
                               seed_base = args$seed_base + ci * 100000L,
                               moments = dist_moments, fl = get_fl(cell$dist),
                               core = core, sim_calibration = cached_cal)
  if (!is.null(cal_key) && is.null(cached_cal) && !is.null(sampler$calibration)) {
    assign(cal_key, sampler$calibration, envir = .sim_cal_cache)
  }
  cat(sprintf("  cell %2d/%2d: p=%3d N=%4d dist=%-4s ", ci, nrow(cell_grid),
              cell$p, cell$N, cell$dist))
  ok <- 0L; fail <- 0L; tc0 <- proc.time()[["elapsed"]]
  for (rep_idx in seq_len(args$reps)) {
    X <- tryCatch(sampler$draw(rep_idx), error = function(e) e)
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
                                setup_seconds = sampler$setup_seconds,
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

# Achieved VM marginal moments on one large draw, for report-side verification
# that the Fleishman path hits the (skew, kurt) targets.
vm_dists <- unique(cell_grid$dist)[dist_family(unique(cell_grid$dist)) == "vm"]
if (length(vm_dists)) {
  moment_rows <- lapply(vm_dists, function(d) {
    tgt <- dist_moments[[d]]; fl <- get_fl(d)
    set.seed(0xF1E15)
    Z <- stats::rnorm(2e5L)
    Y <- fl$a + fl$b * Z + fl$c * Z^2 + fl$d * Z^3
    data.frame(dist = d, skew_target = tgt[[1L]], kurt_target = tgt[[2L]],
               skew_achieved = mean((Y - mean(Y))^3) / stats::sd(Y)^3,
               kurt_achieved = mean((Y - mean(Y))^4) / stats::var(Y)^2 - 3,
               fleishman_resid = fl$resid, stringsAsFactors = FALSE)
  })
  utils::write.csv(do.call(rbind, moment_rows),
                   file.path(results_dir, "moments.csv"), row.names = FALSE)
}

write_csv(metadata_frame(
  values = list(reps = args$reps, seed_base = args$seed_base,
                cells_filter = args$cells_filter,
                semtests_parity = isTRUE(args$semtests_parity),
                n_cells = nrow(cell_grid),
                total_seconds = sprintf("%.2f", t1 - t0)),
  packages = c("magmaan", "lavaan", "semTests")),
  file.path(results_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1fs — results in %s\n", t1 - t0, results_dir))
