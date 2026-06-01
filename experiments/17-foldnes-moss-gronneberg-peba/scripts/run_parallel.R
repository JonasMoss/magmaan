#!/usr/bin/env Rscript
# Parallel driver for the Foldnes-Moss-Gronneberg (2024) goodness-of-fit grid.
#
# Same question and same per-cell computation as ../run_experiment.R, but forks
# the 84 cells across workers with parallel::mclapply (heaviest cells first,
# dynamic scheduling) and chunks the replicate loop so memory stays bounded even
# for the p=40/N=3000 IG/PL cells (whose batched draws would otherwise hold
# ~3 GB each at 3000 reps). It accumulates rejection counts rather than every
# p-value, and writes the same results/rejection_rates.csv + metadata.csv the
# report reads. No semTests parity here (that is the serial runner's job).
#
# Usage:
#   Rscript scripts/run_parallel.R [--reps N] [--cores K] [--chunk C]
#                                  [--seed-base S] [--cells FILTER]
# Defaults: reps=100, cores=4, chunk=250.

suppressWarnings(suppressMessages(library(magmaan)))
suppressWarnings(suppressMessages(library(parallel)))

.support_helpers <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  script <- if (length(fa)) normalizePath(sub("^--file=", "", fa[[1L]]), mustWork = TRUE)
            else normalizePath("scripts/run_parallel.R", mustWork = FALSE)
  file.path(dirname(dirname(dirname(script))), "_support", "R", "helpers.R")
}
source(.support_helpers()); rm(.support_helpers)

# This script lives in scripts/, so the experiment root is two levels up; point
# population.R and results/ there explicitly (helpers resolve relative to here).
exp_root <- dirname(dirname(normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]))))
source(file.path(exp_root, "R", "population.R"))
results_out <- file.path(exp_root, "results")
dir.create(results_out, recursive = TRUE, showWarnings = FALSE)

# ── args ────────────────────────────────────────────────────────────────────
parse <- function(args) {
  out <- list(reps = 100L, cores = 4L, chunk = 250L, seed_base = 20240701L, cells = NULL)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    grab <- function() { i <<- i + 1L; args[[i]] }
    if (a == "--reps") out$reps <- as.integer(grab())
    else if (a == "--cores") out$cores <- as.integer(grab())
    else if (a == "--chunk") out$chunk <- as.integer(grab())
    else if (a == "--seed-base") out$seed_base <- as.integer(grab())
    else if (a == "--cells") out$cells <- grab()
    else if (startsWith(a, "--reps=")) out$reps <- as.integer(sub("^--reps=", "", a))
    else if (startsWith(a, "--cores=")) out$cores <- as.integer(sub("^--cores=", "", a))
    else if (startsWith(a, "--chunk=")) out$chunk <- as.integer(sub("^--chunk=", "", a))
    else if (startsWith(a, "--seed-base=")) out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    else if (startsWith(a, "--cells=")) out$cells <- sub("^--cells=", "", a)
    else stop("unknown arg: ", a, call. = FALSE)
    i <- i + 1L
  }
  out
}
args <- parse(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
core <- magmaan::magmaan_core

# Full 42-statistic battery (paper's 43 minus Bollen-Stine), mirroring run_experiment.R.
fmg_battery <- function() {
  fams <- c("sb", "ss", "eba2", "eba4", "eba6", "peba2", "peba4", "peba6", "pols2", "all")
  g <- expand.grid(fam = fams, ug = c("", "ug"), base = c("ml", "rls"), stringsAsFactors = FALSE)
  gt <- vapply(seq_len(nrow(g)), function(i)
    paste(c(g$fam[[i]], if (nzchar(g$ug[[i]])) g$ug[[i]], g$base[[i]]), collapse = "_"), character(1))
  c("std_ml", "std_rls", sort(gt))
}
TESTS <- fmg_battery()
dist_moments <- list(norm = c(0, 0), vm1 = c(2, 7), vm2 = c(3, 21),
                     ig1 = c(2, 7), ig2 = c(3, 21), pl1 = c(2, 7), pl2 = c(3, 21))

cell_grid <- expand.grid(p = c(10L, 20L, 40L), N = c(400L, 800L, 1500L, 3000L),
                         dist = c("norm", "vm1", "vm2", "ig1", "ig2", "pl1", "pl2"),
                         stringsAsFactors = FALSE)
if (!is.null(args$cells)) for (kv in strsplit(args$cells, ",", fixed = TRUE)[[1L]]) {
  p <- strsplit(kv, "=", fixed = TRUE)[[1L]]
  tgt <- if (is.numeric(cell_grid[[p[[1L]]]])) as.numeric(p[[2L]]) else p[[2L]]
  cell_grid <- cell_grid[cell_grid[[p[[1L]]]] == tgt, , drop = FALSE]
}
# Heaviest first (p, then N) so big cells start early under dynamic scheduling.
cell_grid <- cell_grid[order(-cell_grid$p, -cell_grid$N), , drop = FALSE]
cells <- split(cell_grid, seq_len(nrow(cell_grid)))

# ── one cell: calibrate once, draw in rep-chunks, count rejections ───────────
spec_cache <- new.env(parent = emptyenv())
get_spec <- function(p) {
  k <- as.character(p)
  if (is.null(spec_cache[[k]])) spec_cache[[k]] <-
    list(spec = magmaan::model_spec(build_2factor_syntax(p)),
         syntax = build_2factor_syntax(p), vn = paste0("x", seq_len(p)))
  spec_cache[[k]]
}

run_cell <- function(cell, ci) {
  t0 <- proc.time()[["elapsed"]]
  pop <- build_population_2factor(cell$p); ctx <- get_spec(cell$p)
  fl <- if (dist_family(cell$dist) == "vm")
    fleishman_coef(dist_moments[[cell$dist]][[1L]], dist_moments[[cell$dist]][[2L]]) else NULL
  # Seed keyed on cell identity, not loop/worker order, so the run is
  # reproducible regardless of --cores or scheduling.
  dist_idx <- match(cell$dist, names(dist_moments))
  seed0 <- args$seed_base + cell$p * 1000000L + cell$N * 100L + dist_idx * 7L
  cal <- NULL; reject <- setNames(numeric(length(TESTS)), TESTS); ok <- 0L; fail <- 0L
  done <- 0L
  while (done < args$reps) {
    k <- min(args$chunk, args$reps - done)
    samp <- tryCatch(make_cell_sampler(pop, cell$N, cell$dist, k, seed0 + done,
                                       moments = dist_moments, fl = fl, core = core,
                                       sim_calibration = cal), error = function(e) e)
    if (inherits(samp, "error")) { fail <- fail + k; done <- done + k; next }
    if (is.null(cal)) cal <- samp$calibration   # reuse calibration across chunks
    for (i in seq_len(k)) {
      X <- tryCatch(samp$draw(i), error = function(e) e)
      if (inherits(X, "error")) { fail <- fail + 1L; next }
      colnames(X) <- ctx$vn; df <- as.data.frame(X)
      fit <- tryCatch(magmaan::magmaan(ctx$syntax, df, estimator = "ML"), error = function(e) e)
      if (inherits(fit, "error") || !isTRUE(fit$converged)) { fail <- fail + 1L; next }
      pv <- tryCatch(magmaan::fmg_pvalues(fit, df, tests = TESTS), error = function(e) e)
      if (inherits(pv, "error")) { fail <- fail + 1L; next }
      reject <- reject + (as.numeric(pv) < 0.05); ok <- ok + 1L
    }
    done <- done + k
  }
  list(p = cell$p, N = cell$N, dist = cell$dist, ci = ci,
       reject = reject, ok = ok, fail = fail, seconds = proc.time()[["elapsed"]] - t0)
}

cat(sprintf("parallel FMG-2024: magmaan %s | cells=%d reps=%d cores=%d chunk=%d\n",
            as.character(utils::packageVersion("magmaan")), length(cells),
            args$reps, args$cores, args$chunk))
wall0 <- proc.time()[["elapsed"]]
res <- mclapply(seq_along(cells), function(j) run_cell(cells[[j]], j),
                mc.cores = args$cores, mc.preschedule = FALSE)
wall <- proc.time()[["elapsed"]] - wall0

bad <- vapply(res, function(r) inherits(r, "try-error") || is.null(r$ok), logical(1))
if (any(bad)) stop(sum(bad), " cells failed in workers", call. = FALSE)

rr <- do.call(rbind, lapply(res, function(r) data.frame(
  cell_idx = r$ci, p = r$p, N = r$N, dist = r$dist, test = TESTS,
  n_reps = r$ok, reject_05 = if (r$ok) r$reject / r$ok else NA_real_,
  stringsAsFactors = FALSE)))
utils::write.csv(rr, file.path(results_out, "rejection_rates.csv"), row.names = FALSE)

meta <- do.call(rbind, lapply(res, function(r) data.frame(
  cell_idx = r$ci, p = r$p, N = r$N, dist = r$dist, rep_ok = r$ok,
  rep_fail = r$fail, seconds = r$seconds, stringsAsFactors = FALSE)))
utils::write.csv(meta, file.path(results_out, "cell_meta.csv"), row.names = FALSE)
write_csv(metadata_frame(
  values = list(reps = args$reps, cores = args$cores, chunk = args$chunk,
                seed_base = args$seed_base, n_cells = length(cells),
                wall_seconds = sprintf("%.1f", wall),
                cpu_seconds = sprintf("%.1f", sum(meta$seconds)),
                speedup = sprintf("%.2f", sum(meta$seconds) / wall)),
  packages = c("magmaan")), file.path(results_out, "metadata.csv"))

cat(sprintf("done: wall=%.1fs  cpu=%.1fs  speedup=%.2fx  total ok=%d fail=%d\n",
            wall, sum(meta$seconds), sum(meta$seconds) / wall,
            sum(meta$rep_ok), sum(meta$rep_fail)))
cat(sprintf("results -> %s\n", results_out))
