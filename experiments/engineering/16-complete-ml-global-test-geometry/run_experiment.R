#!/usr/bin/env Rscript
# Information geometry of the complete-data ML global tests.
#
# Decision this informs: which information matrices magmaan's default policy
# uses in the global score and likelihood-ratio tests for complete-data ML
# (project/design/r-interface-vision.md). Under a correct model every arm is
# asymptotically valid, so the question is finite-sample size: does observed
# information (sensitivity, metric, or the LR spectrum) calibrate better or
# worse than expected information, with SB and PEBA4?
#
# Design: the Foldnes-Moss-Gronneberg (2024) correctly specified two-factor
# CFA (R/population.R), p in {10, 20, 40}, normal and severe VM/IG/PL data,
# N in {200, 400, 1000} (p = 40 only at N >= 400). Each replicate fits ML once
# and evaluates every arm in R/arms.R on that fit.
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--reps-p40 N] [--workers K]
#                            [--run NAME] [--smoke] [--summarize]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "..", "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "population.R"))
source(experiment_path("R", "arms.R"))

parse_args <- function(args) {
  out <- list(reps = 500L, reps_p40 = 250L, workers = 4L, run = "full",
              smoke = FALSE, summarize = FALSE, seed_base = 20260925L)
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    val <- function() { i <<- i + 1L; args[[i]] }
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--reps-p40 N] [--workers K]",
          "[--seed-base S] [--run NAME] [--smoke]\n",
          "  --reps       replicates per p = 10/20 cell (default 500)\n",
          "  --reps-p40   replicates per p = 40 cell (default 250)\n",
          "  --workers    forked workers, one BLAS thread each (default 4)\n",
          "  --run        results subdirectory (default full)\n",
          "  --smoke      two tiny cells, 4 reps each, into results/smoke\n",
          "  --summarize  rebuild rejection_rates.csv and overall.csv from arms.csv\n",
          "Full grid: 32 cells, 14,000 fits, about 28 min on 4 workers.\n")
      quit(save = "no", status = 0L)
    }
    else if (a == "--seed-base") out$seed_base <- as.integer(val())
    else if (a == "--reps") out$reps <- as.integer(val())
    else if (a == "--reps-p40") out$reps_p40 <- as.integer(val())
    else if (a == "--workers") out$workers <- as.integer(val())
    else if (a == "--run") out$run <- val()
    else if (a == "--smoke") out$smoke <- TRUE
    else if (a == "--summarize") out$summarize <- TRUE
    else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out
}

# Rejection rates at alpha = .05 by cell, arm and calibration, plus an overall
# summary. Cells in which an arm has no usable fits are excluded from that arm's
# summary and show up in its usable share.
summarize_run <- function(run_dir) {
  arms <- utils::read.csv(file.path(run_dir, "arms.csv"), stringsAsFactors = FALSE)
  arms <- arms[!arms$arm %in% c("fit", "draw"), , drop = FALSE]
  long <- do.call(rbind, lapply(c("std", "sb", "peba4"), function(cal) {
    data.frame(arms[, c("cell", "p", "N", "dist", "rep", "arm")],
               calibration = cal, p_value = arms[[paste0("p_", cal)]],
               stringsAsFactors = FALSE)
  }))
  groups <- split(long, list(long$cell, long$arm, long$calibration), drop = TRUE)
  rates <- do.call(rbind, lapply(groups, function(g) {
    ok <- is.finite(g$p_value)
    data.frame(cell = g$cell[[1L]], p = g$p[[1L]], N = g$N[[1L]], dist = g$dist[[1L]],
               arm = g$arm[[1L]], calibration = g$calibration[[1L]],
               usable = sum(ok), total = nrow(g),
               reject_05 = if (any(ok)) mean(g$p_value[ok] < 0.05) else NA_real_,
               stringsAsFactors = FALSE)
  }))
  rates <- rates[order(rates$cell, rates$arm, rates$calibration), ]
  write_csv(rates, file.path(run_dir, "rejection_rates.csv"))
  overall <- do.call(rbind, lapply(split(rates, list(rates$arm, rates$calibration), drop = TRUE),
    function(g) {
      r <- g$reject_05[is.finite(g$reject_05)]
      data.frame(arm = g$arm[[1L]], calibration = g$calibration[[1L]],
                 mean_abs_error = mean(abs(r - 0.05)), max_rejection = max(r),
                 min_rejection = min(r), cells_without_fits = sum(!is.finite(g$reject_05)),
                 usable_share = sum(g$usable) / sum(g$total))
    }))
  overall <- overall[order(overall$mean_abs_error), ]
  write_csv(overall, file.path(run_dir, "overall.csv"))
  print(overall, row.names = FALSE, digits = 3)
  invisible(overall)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
set_single_threaded_math()
require_pkg("magmaanlab")
core <- magmaanlab::magmaan_core

dist_moments <- list(norm = c(0, 0), vm2 = c(3, 21), ig2 = c(3, 21), pl2 = c(3, 21))
cell_grid <- rbind(
  expand.grid(p = c(10L, 20L), N = c(200L, 400L, 1000L),
              dist = names(dist_moments), stringsAsFactors = FALSE),
  expand.grid(p = 40L, N = c(400L, 1000L),
              dist = names(dist_moments), stringsAsFactors = FALSE))
cell_grid$reps <- ifelse(cell_grid$p == 40L, args$reps_p40, args$reps)
if (args$smoke) {
  cell_grid <- data.frame(p = c(10L, 20L), N = c(200L, 400L), dist = c("vm2", "ig2"),
                          reps = 4L, stringsAsFactors = FALSE)
  args$run <- "smoke"
}
cell_grid$cell <- seq_len(nrow(cell_grid))

run_dir <- file.path(ensure_results_dir(), args$run)
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
arms_path <- file.path(run_dir, "arms.csv")
if (args$summarize) {
  if (!file.exists(arms_path)) stop("no ", arms_path, "; run the simulation first", call. = FALSE)
  summarize_run(run_dir)
  cat("\nsummaries rewritten in", run_dir, "\n")
  quit(save = "no", status = 0L)
}
if (file.exists(arms_path)) file.remove(arms_path)

fl_for <- function(dist) {
  if (dist_family(dist) != "vm") return(NULL)
  m <- dist_moments[[dist]]
  fleishman_coef(m[[1L]], m[[2L]])
}

cal_cache <- new.env(parent = emptyenv())
cat(sprintf("complete-ML global test geometry: magmaanlab %s, %d cells, %d workers\n",
            as.character(utils::packageVersion("magmaanlab")), nrow(cell_grid),
            args$workers))
t_start <- proc.time()[["elapsed"]]
meta_rows <- list()
for (ci in seq_len(nrow(cell_grid))) {
  cell <- as.list(cell_grid[ci, , drop = FALSE])
  pop <- build_population_2factor(cell$p)
  syntax <- build_2factor_syntax(cell$p)
  varnames <- paste0("x", seq_len(cell$p))
  cal_key <- paste(cell$p, cell$dist)
  sampler <- make_cell_sampler(
    pop, cell$N, cell$dist, cell$reps,
    seed_base = args$seed_base + ci * 100000L, moments = dist_moments,
    fl = fl_for(cell$dist), core = core,
    sim_calibration = cal_cache[[cal_key]])
  if (!is.null(sampler$calibration)) cal_cache[[cal_key]] <- sampler$calibration
  t0 <- proc.time()[["elapsed"]]
  one <- function(i) {
    X <- tryCatch(sampler$draw(i), error = function(e) e)
    if (inherits(X, "error")) return(data.frame(rep = i, arm = "draw", error = conditionMessage(X)))
    colnames(X) <- varnames
    fit <- tryCatch(magmaanlab::fit_model(syntax, as.data.frame(X), estimator = "ML"),
                    error = function(e) e)
    if (inherits(fit, "error") || !isTRUE(fit$converged)) {
      msg <- if (inherits(fit, "error")) conditionMessage(fit) else "not converged"
      return(data.frame(rep = i, arm = "fit", error = msg))
    }
    out <- fit_arms(fit, X)
    out$rep <- i
    out
  }
  res <- parallel::mclapply(seq_len(cell$reps), one, mc.cores = args$workers,
                            mc.preschedule = TRUE)
  res <- do.call(rbind, lapply(res, function(r) {
    if (inherits(r, "try-error")) return(NULL)
    full <- data.frame(rep = r$rep, arm = r$arm,
                       statistic = r$statistic %||% NA_real_, df = r$df %||% NA_integer_,
                       p_std = r$p_std %||% NA_real_, p_sb = r$p_sb %||% NA_real_,
                       p_peba4 = r$p_peba4 %||% NA_real_,
                       n_negative = r$n_negative %||% NA_integer_,
                       error = r$error %||% NA_character_, stringsAsFactors = FALSE)
    full
  }))
  res <- cbind(cell = ci, p = cell$p, N = cell$N, dist = cell$dist, res)
  append_csv(res, arms_path)
  secs <- proc.time()[["elapsed"]] - t0
  n_fit <- length(unique(res$rep[!res$arm %in% c("fit", "draw")]))
  elapsed <- proc.time()[["elapsed"]] - t_start
  done_share <- sum(cell_grid$reps[seq_len(ci)] * cell_grid$p[seq_len(ci)]^3) /
    sum(cell_grid$reps * cell_grid$p^3)
  cat(sprintf("  cell %2d/%d p=%2d N=%4d %-4s fits=%d/%d (%.0fs; ETA %.1f min)\n",
              ci, nrow(cell_grid), cell$p, cell$N, cell$dist, n_fit, cell$reps, secs,
              elapsed * (1 / done_share - 1) / 60))
  meta_rows[[ci]] <- data.frame(cell = ci, p = cell$p, N = cell$N, dist = cell$dist,
                                reps = cell$reps, fits = n_fit, seconds = secs)
}
t_total <- proc.time()[["elapsed"]] - t_start
write_rows(meta_rows, file.path(run_dir, "cells.csv"))
summarize_run(run_dir)

write_csv(metadata_frame(
  values = list(reps = args$reps, reps_p40 = args$reps_p40, workers = args$workers,
                seed_base = args$seed_base, n_cells = nrow(cell_grid),
                total_seconds = sprintf("%.1f", t_total)),
  packages = "magmaanlab"), file.path(run_dir, "metadata.csv"))
cat(sprintf("\ndone in %.1f min; results in %s\n", t_total / 60, run_dir))
