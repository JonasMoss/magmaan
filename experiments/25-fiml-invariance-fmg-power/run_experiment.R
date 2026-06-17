#!/usr/bin/env Rscript
# Measurement-invariance difference tests under non-normal incomplete data:
# FMG/pEBA vs the MLR/SB default, under FIML and two-stage ML (ML2S).
#
# See report.qmd for the full design. This runner executes the p=6 one-factor
# two-group invariance ladder (exp-21/23 base model) over the experiment-17
# non-normal families (norm, vm1/2, ig1/2, pl1/2 -- magmaan's native VM/IG/PL
# simulators), FIML + ML2S, MCAR/MAR, harvesting the full eigenvalue battery on
# every constrained model and the Satorra-2000 difference test on every clean
# ladder step. The Brace & Savalei two-factor p in {8,16,30} populations are the
# build-out (population.R gates p != 6).
#
# OVERNIGHT-SAFE: cells run in parallel; each cell writes its rows to
# results/raw/cell_*.csv the moment it finishes, and the final fits.csv is
# concatenated from those files -- so a crash or early kill keeps every completed
# cell. Type-I (null) cells run FIRST, so the highest-value signal is guaranteed
# even if the violation cells do not all finish.
#
# Usage:
#   Rscript run_experiment.R --help
#   Rscript run_experiment.R --smoke
#   Rscript run_experiment.R --reps 3000 --cores 10        # the overnight job

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else normalizePath("run_experiment.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
source(file.path(dirname(.support_helpers()), "missingness.R"))
rm(.support_helpers)
source(experiment_path("R", "population.R"))
source(experiment_path("R", "generators.R"))
source(experiment_path("R", "tests.R"))

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 3000L, seed_base = 20260616L,
              mechs = c("MCAR", "MAR"), rates = c(0.15, 0.30),
              dists = c("norm", "vm1", "vm2", "ig1", "ig2", "pl1", "pl2"),
              conds = c("null", "weak", "strong", "strict"),
              ratios = c("1:1", "1:3"),
              cores = max(parallel::detectCores() - 2L, 1L), smoke = FALSE)
  i <- 1L; take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--cores N] [--seed-base S]\n",
          "       [--mechs MCAR,MAR] [--rates 0.15,0.30]\n",
          "       [--dists norm,vm1,vm2,ig1,ig2,pl1,pl2] [--smoke]\n\n",
          "  Two-group p=6 invariance ladder; FIML + ML2S; MLR/SB vs the FMG\n",
          "  eigenvalue battery; difference tests via Satorra-2000. Non-normal\n",
          "  families are magmaan's native VM/IG/PL. Cells run in parallel,\n",
          "  null (Type-I) first, with per-cell checkpoints in results/raw/.\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--cores") { i <- i + 1L; out$cores <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--cores=")) { out$cores <- as.integer(sub("^--cores=", "", a))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(take(i - 1L))
    } else if (a == "--mechs") { i <- i + 1L; out$mechs <- parse_csv_arg(take(i - 1L))
    } else if (a == "--rates") { i <- i + 1L; out$rates <- parse_csv_numeric(take(i - 1L))
    } else if (a == "--dists") { i <- i + 1L; out$dists <- parse_csv_arg(take(i - 1L))
    } else if (a == "--conds") { i <- i + 1L; out$conds <- parse_csv_arg(take(i - 1L))
    } else if (a == "--ratios") { i <- i + 1L; out$ratios <- parse_csv_arg(take(i - 1L))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))

# ---- design grid (full intended factorial, for discussion / cells.csv) -------

design_grid <- function() {
  expand.grid(
    p = c(8L, 16L, 30L), n_total = c(100L, 200L, 500L, 1000L),
    ratio = c("1:1", "1:3"), dist = c("norm", "vm2", "ig2", "pl2"),
    mech = c("complete", "MCAR", "MAR"), rate = c(0.15, 0.30),
    cond = c("null", "weak", "strong", "strict"),
    delta = c(0.20), k = c(1L, 2L), stringsAsFactors = FALSE)
}

# ---- executable cells (p=6 model). null (Type-I) cells FIRST. ----------------

executable_cells <- function(cfg) {
  miss <- rbind(data.frame(mech = "complete", rate = 0.0, stringsAsFactors = FALSE),
                expand.grid(mech = cfg$mechs, rate = cfg$rates,
                            stringsAsFactors = FALSE))
  rows <- list()
  conds <- intersect(c("null", "weak", "strong", "strict"), cfg$conds)  # null first
  for (cd in conds)
    for (rt in cfg$ratios)
      for (d in cfg$dists)
        for (j in seq_len(nrow(miss)))
          rows[[length(rows) + 1L]] <- data.frame(
            p = 6L, n_total = 400L, ratio = rt, dist = d,
            mech = miss$mech[j], rate = miss$rate[j],
            cond = cd, delta = 0.30, k = 1L, stringsAsFactors = FALSE)
  do.call(rbind, rows)
}

n_per_group <- function(n_total, ratio) {
  if (ratio == "1:1") rep(n_total %/% 2L, 2L) else c(n_total %/% 4L, n_total - n_total %/% 4L)
}
violate_of <- function(cond, delta, k) {
  if (cond == "null") NULL else list(rung = cond, delta = delta, k = as.integer(k))
}
truth_of <- function(cond, rung) {
  if (cond == "null") "h0" else if (cond == rung) "power" else "contaminated"
}

# ---- run one cell (build sampler, loop reps, checkpoint to results/raw/) ------

run_cell <- function(cell, reps, seed_base, raw_dir) {
  pop <- build_population(cell$p)
  ns  <- n_per_group(cell$n_total, cell$ratio)
  vio <- violate_of(cell$cond, cell$delta, cell$k)
  sampler <- build_cell_sampler(pop, cell$dist, vio, ns, reps = reps,
                                seed_base = seed_base,
                                core = magmaan::magmaan_core,
                                knobs = default_gen_knobs())
  acc <- list(); ok <- 0L
  for (r in seq_len(reps)) {
    rows <- tryCatch(run_one_rep(pop, sampler, r, cell$mech, cell$rate,
                                 mask_seed = seed_base + 100000L + r),
                     error = function(e) NULL)
    if (is.null(rows)) next
    ok <- ok + 1L
    rows$cond <- cell$cond; rows$mech <- cell$mech; rows$rate <- cell$rate
    rows$p <- cell$p; rows$n_total <- cell$n_total; rows$ratio <- cell$ratio
    rows$dist <- cell$dist; rows$delta <- cell$delta; rows$k <- cell$k
    rows$truth <- vapply(rows$rung, truth_of, character(1), cond = cell$cond)
    acc[[length(acc) + 1L]] <- rows
  }
  out <- if (length(acc)) do.call(rbind, c(acc, list(make.row.names = FALSE))) else NULL
  if (!is.null(out)) {
    key <- sprintf("%s_%s_%02d_%s_%s", cell$dist, cell$mech,
                   as.integer(round(cell$rate * 100)), cell$cond,
                   gsub(":", "", cell$ratio))
    write.csv(out, file.path(raw_dir, paste0("cell_", key, ".csv")), row.names = FALSE)
  }
  list(rows = out, converged = ok)
}

# ---- smoke: per-stage timing for one replicate (shows the shared-EM reuse) ----
# Builds one masked p=6 dataset, the single shared saturated EM, then times the
# four fits / three GOF batteries / three nested tests per estimator. With the
# EM threaded in, the FIML nested step is cheap (it reuses the EM instead of
# rebuilding it); at p=6 every stage is fast, but the breakdown makes a future
# regression visible.
smoke_stage_timing <- function() {
  pop <- build_population(6L)
  sampler <- build_cell_sampler(pop, "norm", NULL, c(50L, 50L), reps = 2,
                                seed_base = 1L, core = magmaan::magmaan_core,
                                knobs = default_gen_knobs())
  df <- sampler$draw(1L)
  df <- apply_missingness(df, pop$ov, "MCAR", 0.30, seed = 7L)$df
  tm <- function(e) { t <- proc.time()[["elapsed"]]; force(e)
                      1000 * (proc.time()[["elapsed"]] - t) }
  spec0 <- magmaan::model_spec(invariance_syntax("configural", pop$ov),
                               group = "school", group_labels = c("A", "B"),
                               meanstructure = TRUE)
  em <- magmaan::magmaan_core$estimate_saturated_em_moments(
    magmaan::df_to_fiml_data(df, spec0))
  message("per-replicate stage timing (p=6, 50/group, one shared EM):")
  for (est in c("FIML", "ML2S")) {
    t_fit <- tm(fits <- lapply(c("configural", "metric", "scalar", "strict"),
                               fit_level, df = df, pop = pop, estimator = est,
                               em = em))
    names(fits) <- c("configural", "metric", "scalar", "strict")
    if (any(vapply(fits, is.null, logical(1)))) {
      message(sprintf("  [%-4s] a fit failed (skipping timing)", est)); next
    }
    t_gof <- tm(for (lvl in c("metric", "scalar", "strict"))
                  gof_rows(fits[[lvl]], est, lvl, add_mlr = (est == "FIML")))
    t_nst <- tm(for (pr in ladder_pairs())
                  nested_rows(fits[[pr$h1]], fits[[pr$h0]], est, pr$step))
    message(sprintf("  [%-4s] 4 fits %5.0f ms | 3 GOF %5.0f ms | 3 nested %5.0f ms",
                    est, t_fit, t_gof, t_nst))
  }
}

# ---- main --------------------------------------------------------------------

suppressMessages(library(magmaan))
if (exists("set_single_threaded_math")) set_single_threaded_math()

# The smoke writes to an isolated throwaway dir so it neither rescans the (large)
# overnight raw checkpoints in results/raw/ nor overwrites the real results/
# summaries. The real run uses results/ as before.
res_dir <- if (cfg$smoke) file.path(tempdir(), "exp25-smoke-results") else "results"
raw_dir <- file.path(res_dir, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(design_grid(), file.path(res_dir, "cells.csv"), row.names = FALSE)

cells <- executable_cells(cfg)
if (cfg$smoke) {
  # Fast pipeline + schema + reuse check (seconds, not minutes): a null and a
  # power cell, tiny n/reps, run synchronously, exercising both estimators, all
  # rungs, every GOF battery, and the nested tests -- including the shared-EM
  # reuse plumbing. First print the per-replicate stage breakdown.
  smoke_stage_timing()
  cells <- cells[cells$dist == "norm" & cells$mech == "MCAR" &
                   cells$rate == 0.30 & cells$ratio == "1:1" &
                   cells$cond %in% c("null", "weak"), ]
  cells$n_total <- 100L
  cfg$reps  <- 3L
  cfg$cores <- 1L
}
message(sprintf("%d cells x %d reps on %d cores; null-first; checkpoints -> %s",
                nrow(cells), cfg$reps, cfg$cores, raw_dir))

t0 <- Sys.time()
runner <- function(i) {
  cl <- cells[i, ]
  r <- tryCatch(run_cell(cl, cfg$reps, cfg$seed_base + i * 1000L, raw_dir),
                error = function(e) {
                  message(sprintf("cell %d FAILED: %s", i, conditionMessage(e)))
                  list(rows = NULL, converged = 0L)
                })
  message(sprintf("[%d/%d] %-4s %-8s %.2f %-7s -> %d ok", i, nrow(cells),
                  cl$dist, cl$mech, cl$rate, cl$cond, r$converged))
  r$converged
}
idx <- seq_len(nrow(cells))
if (cfg$cores > 1L) {
  parallel::mclapply(idx, runner, mc.cores = cfg$cores, mc.preschedule = FALSE)
} else lapply(idx, runner)

# Aggregate per-cell from the checkpoints (memory-safe: each cell is ~240k rows,
# so never build one monolithic fits.csv). Each raw file is one design cell, so
# its rejection summary is computed independently and the small summaries stack.
raws <- list.files(raw_dir, pattern = "^cell_.*\\.csv$", full.names = TRUE)
if (!length(raws)) stop("no cells completed", call. = FALSE)
agg_cell <- function(f) {
  d <- read.csv(f, stringsAsFactors = FALSE)
  a <- aggregate(p_value ~ cond + mech + rate + dist + ratio + estimator + rung +
                   outcome + method + h1_information + truth,
                 data = d, FUN = function(p) mean(p < 0.05, na.rm = TRUE),
                 na.action = na.pass)
  names(a)[names(a) == "p_value"] <- "reject"
  n <- aggregate(p_value ~ cond + mech + rate + dist + ratio + estimator + rung +
                   outcome + method + h1_information + truth,
                 data = d, FUN = function(p) sum(!is.na(p)), na.action = na.pass)
  a$n_reps <- n$p_value
  a
}
summ <- do.call(rbind, lapply(raws, agg_cell))
write.csv(summ, file.path(res_dir, "summary_rejection.csv"), row.names = FALSE)

if (cfg$smoke) {
  raw_all <- do.call(rbind, lapply(raws, read.csv, stringsAsFactors = FALSE))
  req  <- c("estimator", "rung", "outcome", "method", "p_value", "cond", "truth")
  miss <- setdiff(req, names(raw_all))
  key  <- raw_all$method %in% c("naive", "SB", "pEBA4")
  checks <- c(
    columns      = length(miss) == 0L,
    estimators   = all(c("FIML", "ML2S") %in% raw_all$estimator),
    outcomes     = all(c("gof", "nested") %in% raw_all$outcome),
    both_conds   = all(c("h0", "power") %in% raw_all$truth),
    pval_finite  = any(key) && all(is.finite(raw_all$p_value[key])))
  for (nm in names(checks))
    message(sprintf("  smoke check %-12s : %s", nm, checks[[nm]]))
  if (!all(checks)) {
    if (length(miss)) message("    missing columns: ", paste(miss, collapse = ", "))
    stop("SMOKE FAILED", call. = FALSE)
  }
  message("SMOKE PASS")
}

meta <- data.frame(
  key = c("reps", "cores", "seed_base", "estimators", "model", "dists",
          "conds", "ratios", "cells_completed", "note", "elapsed_min"),
  value = c(cfg$reps, cfg$cores, cfg$seed_base, "FIML,ML2S", "p6-1factor",
            paste(cfg$dists, collapse = "+"), paste(cfg$conds, collapse = "+"),
            paste(cfg$ratios, collapse = "+"), length(raws),
            "p=6 base (B&S p8/16/30 = build-out, todo); nested = weak+strict only (metric->scalar non-nesting, todo); pEBA-on-difference harvested via fmg_nested for FIML+ML2S",
            round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)),
  stringsAsFactors = FALSE)
write.csv(meta, file.path(res_dir, "metadata.csv"), row.names = FALSE)

message(sprintf("done. %d cells aggregated -> %s/summary_rejection.csv (per-cell raw in %s)",
                length(raws), res_dir, raw_dir))
