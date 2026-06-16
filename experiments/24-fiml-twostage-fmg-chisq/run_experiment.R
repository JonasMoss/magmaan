#!/usr/bin/env Rscript
# Goodness-of-fit chi-square calibration for FIML and two-stage ML (ML2S) under
# non-normal incomplete data: do the Foldnes-Moss-Gronneberg full-spectrum
# statistics (all/pall/pEBA/pOLS) calibrate the FIML and two-stage tests better
# than the Savalei low-moment corrections (mean-scaled T_RML, scaled-shifted,
# scaled-F) catalogued by Savalei & Rosseel (2022)?
#
# Type-I only. One correct single-group one-factor six-indicator CFA with a mean
# structure, so every goodness-of-fit rejection is a Type-I error. No
# misspecified arm, no nested test, no power. Both estimators see the SAME
# generated-and-masked data each replicate (paired). The whole statistic battery
# is a transform of one (base chi-square, df, UGamma spectrum) triple per
# estimator, so it costs one fmg_tests() call each.
#
# Grid: distribution {norm, vm1/2, ig1/2, pl1/2} x N x missingness {complete,
# MCAR, MAR} x rate. Non-normal families and Savalei-Bentler 2005 MCAR/MAR
# generators are experiment 17's / experiments/_support's. Under MCAR both
# estimators are consistent, so rejection != alpha is a pure reference-law
# (tail-approximation) failure -- the clean horse race. Under MAR + non-normality
# both estimators share the same Gaussian-imputation point bias, so the base
# statistic drifts above the reference-law mean (trace UGamma) for both and no
# statistic restores level; the noncentrality table reports that drift.
#
# Outputs (curated under results/; raw per-replicate under results/raw/):
#   gof_fits.csv          per-rep p-value + base stat / df / trace, by estimator x method
#   spectra.rds           per-rep UGamma spectra (sufficient stats; no rerun)
#   summary_rejection.csv rejection rate by estimator x method x dist x N x mech x rate
#   noncentrality.csv     mean(base) vs mean(trace) per estimator x cell (the bias diagnostic)
#   parity.csv            magmaan vs lavaan: FIML MLR/YB + lavaan two.stage
#   cells.csv             the design grid
#   metadata.csv          arguments, seeds, package versions
#
# Usage:
#   Rscript run_experiment.R [--reps N] [--ns CSV] [--seed-base S]
#       [--dists CSV] [--rates CSV] [--mechs CSV] [--cells FILTER]
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
source(file.path(dirname(.support_helpers()), "missingness.R"))
rm(.support_helpers)
source(experiment_path("R", "population.R"))
source(experiment_path("R", "generators.R"))
source(experiment_path("R", "tests.R"))
source(experiment_path("R", "oracle.R"))

# ---- arguments ---------------------------------------------------------------

parse_args <- function(args) {
  out <- list(reps = 500L, ns = c(150L, 300L, 600L), seed_base = 20260616L,
              dists = c("norm", "vm2", "ig2", "pl2"),
              rates = c(0.15, 0.30), mechs = c("MCAR", "MAR"),
              cells_filter = NULL, lavaan_parity = FALSE, smoke = FALSE)
  i <- 1L
  take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--ns 150,300,600]\n",
          "       [--seed-base S] [--dists norm,vm2,ig2,pl2]\n",
          "       [--rates 0.15,0.30] [--mechs MCAR,MAR]\n",
          "       [--cells dist=pl2,mech=MAR,rate=0.3,n=300]\n",
          "       [--lavaan-parity] [--smoke]\n\n", sep = "")
      cat("  Single-group correct one-factor CFA; FIML and two-stage ML (ML2S)\n",
          "  goodness-of-fit Type-I across the FMG eigenvalue battery vs the\n",
          "  Savalei low-moment corrections. Missingness via Savalei-Bentler\n",
          "  2005 MCAR/MAR; 'complete' is always included. dist: norm, vm1/2,\n",
          "  ig1/2, pl1/2 (*1 skew2 kurt7, *2 skew3 kurt21).\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--ns") { i <- i + 1L; out$ns <- as.integer(parse_csv_numeric(take(i - 1L)))
    } else if (startsWith(a, "--ns=")) { out$ns <- as.integer(parse_csv_numeric(sub("^--ns=", "", a)))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--seed-base=")) { out$seed_base <- as.integer(sub("^--seed-base=", "", a))
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
    out$reps <- max(out$reps %/% 50L, 6L)
    out$ns <- out$ns[1L]
    out$cells_filter <- out$cells_filter %||% "dist=pl2,mech=MAR,rate=0.3"
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

pop <- build_population()
spec <- magmaan::model_spec(model_syntax(), meanstructure = TRUE)
alpha <- 0.05

mc <- miss_conditions(cfg$mechs, cfg$rates)
base_cells <- expand.grid(dist = cfg$dists, n = cfg$ns, stringsAsFactors = FALSE)
grid <- do.call(rbind, lapply(seq_len(nrow(base_cells)), function(i)
  data.frame(dist = base_cells$dist[i], n = base_cells$n[i],
             mech = mc$mech, rate = mc$rate, stringsAsFactors = FALSE)))
grid <- apply_cells_filter(grid, cfg$cells_filter)
if (!nrow(grid)) stop("the cell filter selected no cells", call. = FALSE)
rownames(grid) <- NULL
cat(sprintf("simulation: %d cell(s) x %d reps (N in {%s})\n",
            nrow(grid), cfg$reps, paste(sort(unique(grid$n)), collapse = ", ")))

# ---- main loop ---------------------------------------------------------------
# The non-normal draw depends only on (dist, n); missingness is applied
# afterwards. Cache the per-(dist,n) sampler so IG/PL calibration is paid once
# and reused across mechanisms and rates.

sampler_cache <- new.env(parent = emptyenv())
get_sampler <- function(dist, n) {
  key <- paste(dist, n, sep = ":")
  if (!is.null(sampler_cache[[key]])) return(sampler_cache[[key]])
  s <- build_cell_sampler(pop, dist, n, cfg$reps, cfg$seed_base, core = core)
  assign(key, s, envir = sampler_cache)
  s
}
mask_seed_for <- function(ci, rep) cfg$seed_base + ci * 100000L + rep

gof_rows <- list(); spectra <- list(); gk <- 0L
t_start <- proc.time()[["elapsed"]]
for (ci in seq_len(nrow(grid))) {
  cell <- grid[ci, ]
  sampler <- get_sampler(cell$dist, cell$n)
  n_ok <- 0L
  for (rep in seq_len(cfg$reps)) {
    res <- run_one_rep(spec, sampler, rep, cell$mech, cell$rate,
                       mask_seed_for(ci, rep))
    if (is.null(res)) next
    n_ok <- n_ok + 1L
    tag <- data.frame(dist = cell$dist, n = cell$n, mech = cell$mech,
                      rate = cell$rate, rep = rep, stringsAsFactors = FALSE)
    gk <- gk + 1L; gof_rows[[gk]] <- cbind(tag, res$rows)
    key <- paste(cell$dist, cell$n, cell$mech, cell$rate, rep, sep = "|")
    spectra[[key]] <- res$spectra
  }
  cat(sprintf("  %-4s N=%-4d %-8s rate=%.2f  %d/%d ok\n",
              cell$dist, cell$n, cell$mech, cell$rate, n_ok, cfg$reps))
}
gof_df <- do.call(rbind, gof_rows)

# ---- aggregate: rejection + the consistency (noncentrality) diagnostic -------

aggregate_rejection <- function(d) {
  if (is.null(d) || !nrow(d)) return(data.frame())
  grp <- split(d, list(d$estimator, d$method, d$dist, d$n, d$mech, d$rate),
               drop = TRUE)
  do.call(rbind, lapply(grp, function(g) data.frame(
    estimator = g$estimator[1L], method = g$method[1L], dist = g$dist[1L],
    n = g$n[1L], mech = g$mech[1L], rate = g$rate[1L],
    n_rep = nrow(g), reject = rejection_rate(g$p_value, alpha),
    row.names = NULL, stringsAsFactors = FALSE)))
}
summary_df <- aggregate_rejection(gof_df)

# Under a consistent estimator E[base_stat] = trace(UGamma); the excess
# ncp_hat = mean(base) - mean(trace) is the gap to the reference-law mean. It is
# ~0 only under NORMAL data; under non-normality it is already positive at
# complete data (a finite-sample sandwich-trace effect) and, under MAR + non-
# normality, both estimators inherit the Gaussian-imputation point bias, which
# inflates it further. The base statistic and trace are shared across methods, so
# restrict to one method (std) per (estimator, rep) to avoid recounting.
noncentrality_df <- local({
  d <- gof_df[gof_df$method == "std", ]
  if (!nrow(d)) return(data.frame())
  grp <- split(d, list(d$estimator, d$dist, d$n, d$mech, d$rate), drop = TRUE)
  do.call(rbind, lapply(grp, function(g) data.frame(
    estimator = g$estimator[1L], dist = g$dist[1L], n = g$n[1L],
    mech = g$mech[1L], rate = g$rate[1L], df = g$df[1L], n_rep = nrow(g),
    mean_base = mean(g$base_stat), mean_trace = mean(g$trace),
    ncp_hat = mean(g$base_stat) - mean(g$trace),
    realized_rate = mean(g$realized_rate),
    row.names = NULL, stringsAsFactors = FALSE)))
})

# ---- lavaan parity oracle (rep 1, h0 cells) ---------------------------------

parity_rows <- list(); pk <- 0L
if (have_lav) {
  cat("lavaan parity: FIML MLR/Yuan-Bentler + two.stage (h0 cells, rep 1)\n")
  par_grid <- unique(grid[, c("dist", "n", "mech", "rate")])
  for (pidx in seq_len(nrow(par_grid))) {
    pc <- par_grid[pidx, ]
    ci <- which(grid$dist == pc$dist & grid$n == pc$n &
                grid$mech == pc$mech & grid$rate == pc$rate)[1L]
    sampler <- get_sampler(pc$dist, pc$n)
    mm <- apply_missingness(sampler$draw(1L), pop$ov, pc$mech, pc$rate,
                            seed = mask_seed_for(ci, 1L))
    f_fiml <- fit_fiml_model(spec, mm$df)
    f_ml2s <- fit_ml2s_model(spec, mm$df)
    if (is.null(f_fiml) || is.null(f_ml2s)) next
    b_fiml <- fmg_battery(f_fiml); b_ml2s <- fmg_battery(f_ml2s)
    if (is.null(b_fiml) || is.null(b_ml2s)) next
    mag <- list(
      fiml_base = b_fiml$rows$base_stat[b_fiml$rows$method == "std"][1L],
      fiml_sb_p = b_fiml$rows$p_value[b_fiml$rows$method == "sb"][1L],
      ml2s_base = b_ml2s$rows$base_stat[b_ml2s$rows$method == "std"][1L],
      ml2s_sb_p = b_ml2s$rows$p_value[b_ml2s$rows$method == "sb"][1L])
    for (r in parity_rows_for_cell(model_syntax(), mm$df, pc$dist, pc$mech,
                                   pc$rate, mag)) {
      pk <- pk + 1L; parity_rows[[pk]] <- r
    }
  }
}
parity_df <- if (pk) do.call(rbind, parity_rows) else data.frame()
elapsed <- proc.time()[["elapsed"]] - t_start

# ---- write outputs ----------------------------------------------------------

raw_dir <- file.path(results_dir(create = TRUE), "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(grid, file.path(results_dir(), "cells.csv"))
write_csv(gof_df, file.path(raw_dir, "gof_fits.csv"))
saveRDS(spectra, file.path(raw_dir, "spectra.rds"))
write_csv(summary_df, file.path(results_dir(), "summary_rejection.csv"))
if (nrow(noncentrality_df)) {
  write_csv(noncentrality_df, file.path(results_dir(), "noncentrality.csv"))
}
if (nrow(parity_df)) write_csv(parity_df, file.path(results_dir(), "parity.csv"))
write_metadata(
  file.path(results_dir(), "metadata.csv"),
  values = list(reps = cfg$reps, ns = paste(cfg$ns, collapse = ","),
                seed_base = cfg$seed_base, lavaan_parity = have_lav,
                cells_filter = cfg$cells_filter %||% "",
                dists = paste(cfg$dists, collapse = ","),
                mechs = paste(cfg$mechs, collapse = ","),
                rates = paste(cfg$rates, collapse = ","),
                missingness = "Savalei-Bentler 2005 MCAR/MAR (experiments/_support)",
                estimators = "FIML, ML2S (two-stage ML)",
                statistics = paste(fmg_battery_tests(), collapse = ","),
                model = "single-group 1-factor 6-indicator CFA (correct), df=9",
                alpha = alpha, elapsed_seconds = round(elapsed, 1)),
  packages = c("magmaan", if (have_lav) "lavaan"))

cat(sprintf("done in %.1fs; wrote results to %s\n", elapsed, results_dir()))
