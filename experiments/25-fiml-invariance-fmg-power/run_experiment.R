#!/usr/bin/env Rscript
# Measurement-invariance difference tests under non-normal incomplete data:
# FMG/pEBA vs the MLR/SB default, under FIML and two-stage ML (ML2S).
#
# See report.qmd for the full design. STATUS: scaffold. The runnable path is the
# one-factor six-indicator smoke model on NORMAL data, which exercises the whole
# pipeline -- both estimators, the full ladder (configural/metric/scalar/strict),
# every eigenvalue p-value, and the Satorra-2000 difference test -- and writes
# the long-format output schema. Two pieces are deliberately gated and marked
# TODO until the design is fixed with Njal:
#   (1) the Brace & Savalei two-factor p in {8,16,30} populations (population.R), and
#   (2) the non-normal families vm/ig/pl (generators.R).
# `cells.csv` still enumerates the FULL intended design for discussion.
#
# Usage:
#   Rscript run_experiment.R --help
#   Rscript run_experiment.R --smoke              # 1 cell, normal data, pipeline check
#   Rscript run_experiment.R --reps 5000          # executable grid (normal, p=6) at scale

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
  out <- list(reps = 300L, seed_base = 20260616L,
              mechs = c("MCAR", "MAR"), rates = c(0.15, 0.30),
              smoke = FALSE)
  i <- 1L; take <- function(i) args[[i + 1L]]
  while (i <= length(args)) {
    a <- args[[i]]
    if (a %in% c("-h", "--help")) {
      cat("Usage: Rscript run_experiment.R [--reps N] [--seed-base S]\n",
          "       [--mechs MCAR,MAR] [--rates 0.15,0.30] [--smoke]\n\n",
          "  Two-group invariance ladder; FIML + ML2S; MLR/SB vs the FMG\n",
          "  eigenvalue battery; difference tests via Satorra-2000. Scaffold:\n",
          "  runs the p=6 NORMAL smoke model (see report.qmd). cells.csv lists\n",
          "  the full intended design (p in {8,16,30}, vm/ig/pl, gradient).\n", sep = "")
      quit(save = "no", status = 0L)
    } else if (a == "--reps") { i <- i + 1L; out$reps <- as.integer(take(i - 1L))
    } else if (startsWith(a, "--reps=")) { out$reps <- as.integer(sub("^--reps=", "", a))
    } else if (a == "--seed-base") { i <- i + 1L; out$seed_base <- as.integer(take(i - 1L))
    } else if (a == "--mechs") { i <- i + 1L; out$mechs <- parse_csv_arg(take(i - 1L))
    } else if (a == "--rates") { i <- i + 1L; out$rates <- parse_csv_numeric(take(i - 1L))
    } else if (a == "--smoke") { out$smoke <- TRUE
    } else stop("unknown argument: ", a, call. = FALSE)
    i <- i + 1L
  }
  out
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))

# ---- design grid (full intended factorial, for discussion / cells.csv) -------
# This is what we MEAN to run once the populations and families are built out.
# Executable subset right now: dist == "norm" and p == 6 (mapped from the design
# by the smoke model). Everything here is enumerated so Njal can see the shape.

design_grid <- function() {
  expand.grid(
    p          = c(8L, 16L, 30L),
    n_total    = c(100L, 200L, 500L, 1000L),
    ratio      = c("1:1", "1:3"),
    dist       = c("norm", "vm2", "ig2", "pl2"),
    mech       = c("complete", "MCAR", "MAR"),
    rate       = c(0.15, 0.30),
    cond       = c("null", "weak", "strong", "strict"),
    delta      = c(0.20),            # placeholder; calibrate to Delta-RMSEA (TODO)
    k          = c(1L, 2L),          # # noninvariant indicators (partial invariance)
    stringsAsFactors = FALSE
  )
}

# ---- executable cells (what the scaffold can actually run today) -------------

executable_cells <- function(cfg) {
  miss <- rbind(
    data.frame(mech = "complete", rate = 0.0, stringsAsFactors = FALSE),
    expand.grid(mech = cfg$mechs, rate = cfg$rates, stringsAsFactors = FALSE))
  conds <- c("null", "weak", "strong", "strict")
  cells <- do.call(rbind, lapply(conds, function(cd) {
    data.frame(p = 6L, n_total = 400L, ratio = "1:1", dist = "norm",
               miss, cond = cd, delta = 0.30, k = 1L,
               stringsAsFactors = FALSE)
  }))
  cells
}

n_per_group <- function(n_total, ratio) {
  if (ratio == "1:1") rep(n_total %/% 2L, 2L)
  else c(n_total %/% 4L, n_total - n_total %/% 4L)   # 1:3
}

violate_of <- function(cond, delta, k) {
  if (cond == "null") NULL else list(rung = cond, delta = delta, k = as.integer(k))
}

truth_of <- function(cond, rung) {
  if (cond == "null") "h0" else if (cond == rung) "power" else "contaminated"
}

# ---- run one executable cell -------------------------------------------------

run_cell <- function(cell, reps, seed_base) {
  pop <- build_population(cell$p)
  ns  <- n_per_group(cell$n_total, cell$ratio)
  vio <- violate_of(cell$cond, cell$delta, cell$k)
  sampler <- make_sampler(pop, ns, dist = cell$dist, violate = vio,
                          seed_base = seed_base)
  acc <- list(); ok <- 0L
  for (r in seq_len(reps)) {
    rows <- run_one_rep(pop, sampler, r, cell$mech, cell$rate,
                        mask_seed = seed_base + 100000L + r)
    if (is.null(rows)) next
    ok <- ok + 1L
    rows$cond <- cell$cond; rows$mech <- cell$mech; rows$rate <- cell$rate
    rows$p <- cell$p; rows$n_total <- cell$n_total; rows$ratio <- cell$ratio
    rows$dist <- cell$dist; rows$delta <- cell$delta; rows$k <- cell$k
    rows$truth <- vapply(rows$rung, truth_of, character(1), cond = cell$cond)
    acc[[length(acc) + 1L]] <- rows
  }
  list(rows = if (length(acc)) do.call(rbind, c(acc, list(make.row.names = FALSE)))
              else NULL,
       converged = ok)
}

# ---- main --------------------------------------------------------------------

suppressMessages(library(magmaan))
if (exists("set_single_threaded_math")) set_single_threaded_math()

dir.create("results", showWarnings = FALSE)
write.csv(design_grid(), file.path("results", "cells.csv"), row.names = FALSE)

cells <- executable_cells(cfg)
if (cfg$smoke) {
  cells <- cells[cells$cond == "null" & cells$mech == "MCAR" & cells$rate == 0.30, ]
  cfg$reps <- max(cfg$reps %/% 20L, 10L)
  message("SMOKE: 1 cell (p=6 norm, null, MCAR .30), ", cfg$reps, " reps, ",
          "both estimators, full ladder.")
}

t0 <- Sys.time()
res <- lapply(seq_len(nrow(cells)), function(i) {
  cl <- cells[i, ]
  message(sprintf("[%d/%d] cond=%s mech=%s rate=%.2f", i, nrow(cells),
                  cl$cond, cl$mech, cl$rate))
  run_cell(cl, cfg$reps, cfg$seed_base + i * 1000L)
})

rows_list <- Filter(Negate(is.null), lapply(res, `[[`, "rows"))
if (!length(rows_list)) stop("no replicates converged", call. = FALSE)
fits <- do.call(rbind, c(rows_list, list(make.row.names = FALSE)))
write.csv(fits, file.path("results", "fits.csv"), row.names = FALSE)

# Rejection-rate summary (the headline table feedstock).
summ <- aggregate(p_value ~ cond + mech + rate + dist + estimator + rung +
                    outcome + method + h1_information + truth,
                  data = fits, FUN = function(p) mean(p < 0.05, na.rm = TRUE),
                  na.action = na.pass)
names(summ)[names(summ) == "p_value"] <- "reject"
write.csv(summ, file.path("results", "summary_rejection.csv"), row.names = FALSE)

meta <- data.frame(
  key = c("reps", "seed_base", "estimators", "model", "dist", "note", "elapsed_s"),
  value = c(cfg$reps, cfg$seed_base, "FIML,ML2S", "p6-1factor-smoke", "norm",
            "scaffold; B&S p8/16/30 + vm/ig/pl are the build-out",
            round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)),
  stringsAsFactors = FALSE)
write.csv(meta, file.path("results", "metadata.csv"), row.names = FALSE)

message("done. rows=", nrow(fits), " -> results/{fits,summary_rejection,cells,metadata}.csv")
