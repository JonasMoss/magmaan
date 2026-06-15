#!/usr/bin/env Rscript
# FIML goodness-of-fit and nested tests under non-normality: FMG/pEBA vs the MLR
# default.
#
# Question: under non-normality + missing data, does the Foldnes-Moss-Gronneberg
# eigenvalue family (SS / EBA / pEBA / pOLS / penalized-all) improve FIML test
# calibration and power over the dominant applied default, MLR (the Yuan-Bentler
# mean-scaled test, lavaan's `estimator="MLR", missing="ml"`)? MLR and the FMG
# "SB" method are mean-only scalings, so under a spread-out UGamma spectrum
# (skewed, non-elliptical data) they miscalibrate while the spectrum-matching
# transforms hold level.
#
# Two regimes (see the report): under MCAR, normal-theory FIML is consistent for
# any distribution, so this is a clean test of the reference-law correction.
# Under MAR + non-normality FIML is itself biased (the Gaussian likelihood
# imputes missing values with linear conditional means), so the chi-square is
# non-central even under a true model and NO reference-law correction can fully
# restore level; the run stores the base statistic and the UGamma trace so the
# report can show mean(T) vs trace(UGamma) and attribute MAR over-rejection to
# estimator bias rather than the test.
#
# One model: two-group, one-factor, six-indicator CFA, mean structure,
# scalar-invariant population (sibling of experiment 21; no big-model arms).
# Grid: truth {h0, gof_power, nested_power} x distribution {norm, vm1/2, ig1/2,
# pl1/2} x missingness {complete, MCAR, MAR} x rate {.15, .30}, unequal groups
# N = (n, 0.7n). Missingness uses the Savalei-Bentler 2005 generators from
# experiments/_support.
#
# Outputs (curated under results/; raw per-replicate under results/raw/):
#   gof_fits.csv          per-rep GOF p-values + base stat / df / trace / rate
#   nested_fits.csv       per-rep nested p-values + difference stat / df / trace
#   spectra.rds           per-rep UGamma spectra (sufficient stats; no rerun)
#   summary_rejection.csv rejection rate by outcome x truth x dist x mech x rate x method
#   noncentrality.csv     h0 mean(base) vs mean(trace) -- the FIML-bias diagnostic
#   parity.csv            magmaan vs lavaan MLR: scaled chi-square / p / spectrum
#   cells.csv             the design grid
#   metadata.csv          arguments, seeds, package versions
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]
#       [--truths CSV] [--dists CSV] [--rates CSV] [--mechs CSV]
#       [--cells FILTER] [--lavaan-parity] [--smoke]

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
source(file.path(dirname(.support_helpers()), "missingness.R"))
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
              rates = c(0.15, 0.30), mechs = c("MCAR", "MAR"),
              cells_filter = NULL, lavaan_parity = FALSE, smoke = FALSE)
  i <- 1L
  take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--n N] [--seed-base S]\n",
          "       [--truths h0,gof_power,nested_power] [--dists norm,vm2,ig2,pl2]\n",
          "       [--rates 0.15,0.30] [--mechs MCAR,MAR]\n",
          "       [--cells truth=h0,dist=pl2,mech=MAR,rate=0.3]\n",
          "       [--lavaan-parity] [--smoke]\n\n", sep = "")
      cat("  Two-group 1-factor invariance ladder under FIML; MLR (Yuan-Bentler)\n",
          "  default vs the FMG eigenvalue battery. Missingness via Savalei-\n",
          "  Bentler 2005 MCAR/MAR; 'complete' is always included. dist: norm,\n",
          "  vm1/2, ig1/2, pl1/2 (*1 skew2 kurt7, *2 skew3 kurt21).\n", sep = "")
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
    } else if (a == "--rates") { i <- i + 1L; out$rates <- parse_csv_numeric(take(i - 1L))
    } else if (startsWith(a, "--rates=")) { out$rates <- parse_csv_numeric(sub("^--rates=", "", a))
    } else if (a == "--mechs") { i <- i + 1L; out$mechs <- parse_csv_arg(take(i - 1L))
    } else if (startsWith(a, "--mechs=")) { out$mechs <- parse_csv_arg(sub("^--mechs=", "", a))
    } else if (a == "--cells") { i <- i + 1L; out$cells_filter <- take(i - 1L)
    } else if (startsWith(a, "--cells=")) { out$cells_filter <- sub("^--cells=", "", a)
    } else if (a == "--lavaan-parity") { out$lavaan_parity <- TRUE
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  if (out$smoke) {
    out$reps <- max(out$reps %/% 30L, 8L)
    out$cells_filter <- out$cells_filter %||% "truth=h0,dist=pl2,mech=MAR,rate=0.3"
    out$lavaan_parity <- TRUE
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))

# Missingness conditions: complete (rate 0) plus each requested mechanism x rate.
miss_conditions <- function(mechs, rates) {
  out <- data.frame(mech = "complete", rate = 0.0, stringsAsFactors = FALSE)
  for (m in mechs) for (r in rates) {
    out <- rbind(out, data.frame(mech = m, rate = r, stringsAsFactors = FALSE))
  }
  out
}

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

mc <- miss_conditions(cfg$mechs, cfg$rates)
base_cells <- expand.grid(truth = cfg$truths, dist = cfg$dists,
                          stringsAsFactors = FALSE)
grid <- do.call(rbind, lapply(seq_len(nrow(base_cells)), function(i)
  data.frame(truth = base_cells$truth[i], dist = base_cells$dist[i],
             mech = mc$mech, rate = mc$rate, stringsAsFactors = FALSE)))
grid <- apply_cells_filter(grid, cfg$cells_filter)
if (!nrow(grid)) stop("the cell filter selected no cells", call. = FALSE)
rownames(grid) <- NULL
cat(sprintf("simulation: %d cell(s) x %d reps (N = %d, %d)\n",
            nrow(grid), cfg$reps, n_groups[1L], n_groups[2L]))

# ---- main loop ---------------------------------------------------------------
# The non-normal draw depends only on (dist, truth); missingness is applied
# afterwards. Cache the per-(dist,truth) sampler so its IG/PL calibration is
# paid once and reused across mechanisms and rates.

sampler_cache <- new.env(parent = emptyenv())
get_sampler <- function(dist, truth) {
  key <- paste(dist, truth, sep = ":")
  if (!is.null(sampler_cache[[key]])) return(sampler_cache[[key]])
  s <- build_cell_sampler(pop, dist, violate_for_truth(truth), n_groups,
                          cfg$reps, cfg$seed_base, core = core)
  assign(key, s, envir = sampler_cache)
  s
}
mask_seed_for <- function(ci, rep) cfg$seed_base + ci * 100000L + rep

gof_rows <- list(); nested_rows <- list(); spectra <- list()
gk <- nk <- 0L
t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(grid))) {
  cell <- grid[ci, ]
  sampler <- get_sampler(cell$dist, cell$truth)
  n_ok <- 0L
  for (rep in seq_len(cfg$reps)) {
    res <- run_one_rep(pop, sampler, rep, cell$mech, cell$rate,
                       mask_seed_for(ci, rep))
    if (is.null(res)) next
    n_ok <- n_ok + 1L
    tag <- data.frame(truth = cell$truth, dist = cell$dist, mech = cell$mech,
                      rate = cell$rate, rep = rep, stringsAsFactors = FALSE)
    gk <- gk + 1L; gof_rows[[gk]]    <- cbind(tag, res$gof)
    nk <- nk + 1L; nested_rows[[nk]] <- cbind(tag, res$nested)
    key <- paste(cell$truth, cell$dist, cell$mech, cell$rate, rep, sep = "|")
    spectra[[key]] <- list(gof = res$gof_spectrum, nested = res$nested_spectrum)
  }
  cat(sprintf("  %-12s %-4s %-8s rate=%.2f  %d/%d ok\n",
              cell$truth, cell$dist, cell$mech, cell$rate, n_ok, cfg$reps))
}
gof_df    <- do.call(rbind, gof_rows)
nested_df <- do.call(rbind, nested_rows)

# ---- aggregate: rejection + the FIML-bias (non-centrality) diagnostic --------

aggregate_rejection <- function(d) {
  if (is.null(d) || !nrow(d)) return(data.frame())
  grp <- split(d, list(d$outcome, d$truth, d$dist, d$mech, d$rate, d$method),
               drop = TRUE)
  do.call(rbind, lapply(grp, function(g) data.frame(
    outcome = g$outcome[1L], truth = g$truth[1L], dist = g$dist[1L],
    mech = g$mech[1L], rate = g$rate[1L], method = g$method[1L],
    n_rep = nrow(g), reject = rejection_rate(g$p_value, alpha),
    stringsAsFactors = FALSE)))
}
summary_df <- rbind(aggregate_rejection(gof_df), aggregate_rejection(nested_df))

# Under h0 with a consistent estimator, E[base_stat] = trace(UGamma). Excess of
# the realized mean over the trace is the non-centrality induced by FIML bias
# (expected ~0 under MCAR, >0 under MAR + non-normality). One row per rep, so
# restrict to a single method to avoid counting the shared base stat repeatedly.
noncentrality_df <- local({
  d <- gof_df[gof_df$truth == "h0" & gof_df$method == "naive", ]
  if (!nrow(d)) return(data.frame())
  grp <- split(d, list(d$dist, d$mech, d$rate), drop = TRUE)
  do.call(rbind, lapply(grp, function(g) data.frame(
    dist = g$dist[1L], mech = g$mech[1L], rate = g$rate[1L], df = g$df[1L],
    n_rep = nrow(g), mean_base = mean(g$base_stat), mean_trace = mean(g$trace),
    ncp_hat = mean(g$base_stat) - mean(g$trace),
    realized_rate = mean(g$realized_rate), stringsAsFactors = FALSE)))
})

# ---- lavaan MLR parity oracle (rep 1, h0 cells, both levels) ----------------

parity_rows <- list(); pk <- 0L
if (have_lav) {
  cat("lavaan parity: MLR scaled chi-square / p-value / UGamma (h0 cells, rep 1)\n")
  par_grid <- unique(grid[grid$truth == "h0", c("dist", "mech", "rate")])
  for (pi in seq_len(nrow(par_grid))) {
    pc <- par_grid[pi, ]
    ci <- which(grid$truth == "h0" & grid$dist == pc$dist &
                grid$mech == pc$mech & grid$rate == pc$rate)[1L]
    sampler <- get_sampler(pc$dist, "h0")
    mm <- apply_missingness(sampler$draw(1L), pop$ov, pc$mech, pc$rate,
                            seed = mask_seed_for(ci, 1L))
    miss_lab <- if (pc$rate > 0) pc$rate else 0
    for (lev in invariance_levels()) {
      fit <- fit_fiml_level(lev, mm$df)
      if (is.null(fit)) next
      g <- fmg_gof(fit); mlr <- mlr_test(fit)
      if (is.null(g) || is.null(mlr)) next
      for (r in parity_rows_for_level(g, mlr, mm$df, lev, miss_lab, pc$dist,
                                      "h0")) {
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
saveRDS(spectra, file.path(raw_dir, "spectra.rds"))
write_csv(summary_df, file.path(results_dir(), "summary_rejection.csv"))
if (nrow(noncentrality_df)) {
  write_csv(noncentrality_df, file.path(results_dir(), "noncentrality.csv"))
}
if (nrow(parity_df)) write_csv(parity_df, file.path(results_dir(), "parity.csv"))
write_metadata(
  file.path(results_dir(), "metadata.csv"),
  values = list(reps = cfg$reps, n_group_a = n_groups[1L], n_group_b = n_groups[2L],
                seed_base = cfg$seed_base, lavaan_parity = have_lav,
                cells_filter = cfg$cells_filter %||% "",
                truths = paste(cfg$truths, collapse = ","),
                dists = paste(cfg$dists, collapse = ","),
                mechs = paste(cfg$mechs, collapse = ","),
                rates = paste(cfg$rates, collapse = ","),
                missingness = "Savalei-Bentler 2005 MCAR/MAR (experiments/_support)",
                estimation = "FIML",
                competitors = "naive,MLR,SB,SS,SF,EBA2/4/6,pEBA2/4/6,pall,pOLS,all",
                elapsed_sec = round(elapsed, 1)),
  packages = c("magmaan", "lavaan"))

cat(sprintf("\nDone in %.1fs. Wrote results to: %s\n", elapsed, results_dir()))
if (nrow(parity_df)) {
  cat("\nParity (max |abs_diff| by metric):\n")
  print(aggregate(abs_diff ~ metric, parity_df, function(x) max(x, na.rm = TRUE)),
        row.names = FALSE)
}
if (nrow(noncentrality_df)) {
  cat("\nFIML-bias diagnostic (h0: mean base stat vs UGamma trace):\n")
  print(noncentrality_df[order(noncentrality_df$mech, noncentrality_df$dist),
                         c("dist", "mech", "rate", "mean_base", "mean_trace",
                           "ncp_hat")], row.names = FALSE)
}
