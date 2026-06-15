#!/usr/bin/env Rscript
# FIML goodness-of-fit and nested tests under non-normality: FMG/pEBA vs the MLR
# default.
#
# Question: under heavy non-normality + MCAR, does the Foldnes-Moss-Gronneberg
# eigenvalue family (SS / pEBA / penalized-all) improve FIML test calibration and
# power over the dominant applied default, MLR (the Yuan-Bentler mean-scaled
# test, lavaan's `estimator = "MLR", missing = "ml"`)? Both MLR and the FMG "SB"
# method are mean-only scalings, so under a spread-out UGamma spectrum (skewed,
# non-elliptical IG/PL data) they should miscalibrate together while the
# spectrum-matching FMG transforms hold nominal level.
#
# One model: two-group, one-factor, six-indicator CFA, mean structure,
# scalar-invariant population (sibling of experiment 21). Grid: truth {h0,
# gof_power, nested_power} x distribution {norm, vm1/2, ig1/2, pl1/2} x
# missingness {complete, MCAR 0.15}, unequal groups N = (n, 0.7n). The
# goodness-of-fit table reads the metric-model fit (h0 = Type I, gof_power =
# power); the nested table reads configural vs metric.
#
# Outputs (curated under results/; raw per-replicate under results/raw/):
#   gof_fits.csv          per-rep goodness-of-fit p-values (raw/)
#   nested_fits.csv       per-rep nested p-values (raw/)
#   summary_rejection.csv rejection rate by outcome x truth x dist x miss x method
#   parity.csv            magmaan vs lavaan MLR: scaled chi-square / p / spectrum
#   cells.csv             the design grid
#   metadata.csv          arguments, seeds, package versions
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]
#       [--truths CSV] [--dists CSV] [--miss CSV] [--cells FILTER]
#       [--lavaan-parity] [--smoke]

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
source(experiment_path("R", "population.R"))
source(experiment_path("R", "generators.R"))
source(experiment_path("R", "tests.R"))
source(experiment_path("R", "oracle.R"))

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 300L, n = 500L, seed_base = 20260615L,
              truths = c("h0", "gof_power", "nested_power"),
              dists = c("norm", "vm1", "vm2", "ig1", "ig2", "pl1", "pl2"),
              miss = c(0.0, 0.15), cells_filter = NULL,
              lavaan_parity = FALSE, smoke = FALSE)
  i <- 1L
  take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]\n",
          "       [--truths h0,gof_power,nested_power] [--dists norm,vm2,ig2,pl2]\n",
          "       [--miss 0,0.15] [--cells truth=h0,dist=pl2,miss=0.15]\n",
          "       [--lavaan-parity] [--smoke]\n\n", sep = "")
      cat("  Two-group 1-factor invariance ladder fit under FIML; compares the\n",
          "  MLR (Yuan-Bentler) default against the FMG eigenvalue family.\n",
          "  dist: norm, vm1/vm2, ig1/ig2, pl1/pl2 (*1 = skew 2 kurt 7,\n",
          "  *2 = skew 3 kurt 21). --lavaan-parity adds the lavaan MLR oracle\n",
          "  (rep 1 of each h0 cell). --smoke runs the single hardest cell.\n",
          sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--n") { i <- i + 1L; out$n <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--n=")) { out$n <- as.integer(sub("^--n=", "", a))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--truths") { i <- i + 1L; out$truths <- parse_csv_arg(take(i - 1L))
    } else if (startsWith(a, "--truths=")) { out$truths <- parse_csv_arg(sub("^--truths=", "", a))
    } else if (a == "--dists") { i <- i + 1L; out$dists <- parse_csv_arg(take(i - 1L))
    } else if (startsWith(a, "--dists=")) { out$dists <- parse_csv_arg(sub("^--dists=", "", a))
    } else if (a == "--miss") { i <- i + 1L; out$miss <- parse_csv_numeric(take(i - 1L))
    } else if (startsWith(a, "--miss=")) { out$miss <- parse_csv_numeric(sub("^--miss=", "", a))
    } else if (a == "--cells") { i <- i + 1L; out$cells_filter <- take(i - 1L)
    } else if (startsWith(a, "--cells=")) { out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--lavaan-parity") { out$lavaan_parity <- TRUE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- max(out$reps %/% 30L, 8L)
    out$cells_filter <- out$cells_filter %||% "truth=h0,dist=pl2,miss=0.15"
    out$lavaan_parity <- TRUE
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))

apply_cells_filter <- function(grid, filter) {
  if (is.null(filter)) return(grid)
  for (kv in parse_csv_arg(filter)) {
    parts <- strsplit(kv, "=", fixed = TRUE)[[1L]]
    key <- parts[[1L]]; val <- parts[[2L]]
    if (!key %in% names(grid)) stop("unknown cell key: ", key, call. = FALSE)
    grid <- grid[as.character(grid[[key]]) == val, , drop = FALSE]
  }
  grid
}

suppressMessages(library(magmaan))
set_single_threaded_math()
core <- magmaan::magmaan_core
have_lav <- cfg$lavaan_parity && requireNamespace("lavaan", quietly = TRUE)
if (cfg$lavaan_parity && !have_lav) {
  message("note: --lavaan-parity requested but lavaan is not installed; skipping the oracle.")
}

pop <- build_invariance_population()
n_groups <- c(cfg$n, round(0.7 * cfg$n))
alpha <- 0.05

grid <- expand.grid(truth = cfg$truths, dist = cfg$dists, miss = cfg$miss,
                    stringsAsFactors = FALSE)
grid <- apply_cells_filter(grid, cfg$cells_filter)
if (!nrow(grid)) stop("the cell filter selected no cells", call. = FALSE)
cat(sprintf("simulation: %d cell(s) x %d reps (N = %d, %d)\n",
            nrow(grid), cfg$reps, n_groups[1L], n_groups[2L]))

# ---- main loop ---------------------------------------------------------------
# The non-normal draw depends only on (dist, truth) through the implied moments;
# missingness is injected afterwards. Cache the per-(dist,truth) sampler so its
# IG/PL calibration is paid once and reused across missingness levels.

sampler_cache <- new.env(parent = emptyenv())
get_sampler <- function(dist, truth) {
  key <- paste(dist, truth, sep = ":")
  if (!is.null(sampler_cache[[key]])) return(sampler_cache[[key]])
  s <- build_cell_sampler(pop, dist, violate_for_truth(truth), n_groups,
                          cfg$reps, cfg$seed_base, core = core)
  assign(key, s, envir = sampler_cache)
  s
}

gof_rows <- list(); nested_rows <- list(); gk <- nk <- 0L
t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(grid))) {
  cell <- grid[ci, ]
  sampler <- get_sampler(cell$dist, cell$truth)
  n_ok <- 0L
  for (rep in seq_len(cfg$reps)) {
    mcar_seed <- cfg$seed_base + ci * 100000L + rep
    res <- run_one_rep(pop, sampler, rep, cell$miss, mcar_seed)
    if (is.null(res)) next
    n_ok <- n_ok + 1L
    tag <- data.frame(truth = cell$truth, dist = cell$dist, miss = cell$miss,
                      rep = rep, stringsAsFactors = FALSE)
    gk <- gk + 1L; gof_rows[[gk]]    <- cbind(tag, res$gof)
    nk <- nk + 1L; nested_rows[[nk]] <- cbind(tag, res$nested)
  }
  cat(sprintf("  %-12s %-4s miss=%.2f  %d/%d ok\n",
              cell$truth, cell$dist, cell$miss, n_ok, cfg$reps))
}
gof_df    <- do.call(rbind, gof_rows)
nested_df <- do.call(rbind, nested_rows)

# ---- aggregate rejection rates ----------------------------------------------

aggregate_rejection <- function(d) {
  if (is.null(d) || !nrow(d)) return(data.frame())
  grp <- split(d, list(d$outcome, d$truth, d$dist, d$miss, d$method), drop = TRUE)
  rows <- lapply(grp, function(g) data.frame(
    outcome = g$outcome[1L], truth = g$truth[1L], dist = g$dist[1L],
    miss = g$miss[1L], method = g$method[1L], n_rep = nrow(g),
    reject = rejection_rate(g$p_value, alpha), stringsAsFactors = FALSE))
  do.call(rbind, rows)
}
summary_df <- rbind(aggregate_rejection(gof_df), aggregate_rejection(nested_df))

# ---- lavaan MLR parity oracle (rep 1, h0 cells, both levels) ----------------

parity_rows <- list(); pk <- 0L
if (have_lav) {
  cat("lavaan parity: MLR scaled chi-square / p-value / UGamma (h0 cells, rep 1)\n")
  par_grid <- unique(grid[grid$truth == "h0", c("dist", "miss")])
  for (pi in seq_len(nrow(par_grid))) {
    pc <- par_grid[pi, ]
    ci <- which(grid$truth == "h0" & grid$dist == pc$dist & grid$miss == pc$miss)[1L]
    sampler <- get_sampler(pc$dist, "h0")
    df <- sampler$draw(1L)
    if (pc$miss > 0) {
      set.seed(cfg$seed_base + ci * 100000L + 1L)
      df <- inject_mcar(df, pop$ov, pc$miss)
    }
    for (lev in invariance_levels()) {
      fit <- fit_fiml_level(lev, df)
      if (is.null(fit)) next
      g <- fmg_gof(fit); mlr <- mlr_test(fit)
      if (is.null(g) || is.null(mlr)) next
      for (r in parity_rows_for_level(g, mlr, df, lev, pc$miss, pc$dist, "h0")) {
        pk <- pk + 1L; parity_rows[[pk]] <- r
      }
    }
  }
}
parity_df <- if (pk) do.call(rbind, parity_rows) else data.frame()
elapsed <- proc.time()[["elapsed"]] - t_start

# ---- write outputs ----------------------------------------------------------

raw_dir <- file.path(results_dir(), "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(grid, file.path(results_dir(), "cells.csv"))
write_csv(gof_df, file.path(raw_dir, "gof_fits.csv"))
write_csv(nested_df, file.path(raw_dir, "nested_fits.csv"))
write_csv(summary_df, file.path(results_dir(), "summary_rejection.csv"))
if (nrow(parity_df)) write_csv(parity_df, file.path(results_dir(), "parity.csv"))
write_metadata(
  file.path(results_dir(), "metadata.csv"),
  values = list(reps = cfg$reps, n_group_a = n_groups[1L], n_group_b = n_groups[2L],
                seed_base = cfg$seed_base, lavaan_parity = have_lav,
                cells_filter = cfg$cells_filter %||% "",
                truths = paste(cfg$truths, collapse = ","),
                dists = paste(cfg$dists, collapse = ","),
                miss = paste(cfg$miss, collapse = ","),
                estimation = "FIML",
                competitors = "naive,MLR,SB,SS,pEBA2,pEBA4,pEBA6,pall,all",
                elapsed_sec = round(elapsed, 1)),
  packages = c("magmaan", "lavaan"))

cat(sprintf("\nDone in %.1fs. Wrote results to: %s\n", elapsed, results_dir()))
if (nrow(parity_df)) {
  cat("\nParity (max |abs_diff| by metric):\n")
  print(aggregate(abs_diff ~ metric, parity_df, function(x) max(x, na.rm = TRUE)),
        row.names = FALSE)
}
if (nrow(summary_df)) {
  cat("\nGoodness-of-fit rejection (this run):\n")
  gof_sum <- summary_df[summary_df$outcome == "gof", ]
  print(gof_sum[order(gof_sum$dist, gof_sum$truth, gof_sum$method),
                c("truth", "dist", "miss", "method", "n_rep", "reject")],
        row.names = FALSE)
}
