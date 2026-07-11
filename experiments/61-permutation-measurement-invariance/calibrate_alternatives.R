#!/usr/bin/env Rscript
# Calibrate reference-study departures to a target studentized-permutation power
# in the normal, balanced, p=8, N=440 cell. Common random seeds are reused over
# candidates within each step.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[[1L]]),
                                                 mustWork = TRUE) else
    normalizePath("calibrate_alternatives.R", mustWork = FALSE)
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
suppressWarnings(suppressMessages(library(magmaan)))
set_single_threaded_math()
source(experiment_path("R", "main_study.R"))
source(experiment_path("R", "simulation.R"))
source(experiment_path("R", "inference.R"))

cfg <- list(reps = 100L, permutations = 99L, iterations = 6L, target = .50,
            cores = max(1L, parallel::detectCores() - 1L), seed_base = 20260712,
            output = experiment_path("results", "alternative-calibration.csv"))
args <- commandArgs(trailingOnly = TRUE)
i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--smoke") {
    cfg$reps <- 5L; cfg$permutations <- 19L; cfg$iterations <- 2L
    cfg$output <- experiment_path("results", "alternative-calibration-smoke.csv")
  } else if (a == "--reps") cfg$reps <- as.integer(take())
  else if (a == "--permutations") cfg$permutations <- as.integer(take())
  else if (a == "--iterations") cfg$iterations <- as.integer(take())
  else if (a == "--target") cfg$target <- as.numeric(take())
  else if (a == "--cores") cfg$cores <- as.integer(take())
  else if (a == "--seed-base") cfg$seed_base <- as.numeric(take())
  else if (a == "--output") cfg$output <- take()
  else if (a %in% c("-h", "--help")) {
    cat("Usage: Rscript calibrate_alternatives.R [--smoke] [--reps N] ",
        "[--permutations B] [--iterations N] [--target P] [--cores N] ",
        "[--output FILE]\n")
    quit(save = "no", status = 0L)
  } else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (cfg$reps < 1L || cfg$permutations < 19L || cfg$iterations < 1L ||
    cfg$cores < 1L || !is.finite(cfg$target) || cfg$target <= 0 || cfg$target >= 1)
  stop("calibration needs positive reps/iterations/cores, at least 19 permutations, ",
       "and a target in (0,1)", call. = FALSE)
dir.create(dirname(cfg$output), recursive = TRUE, showWarnings = FALSE)

reference_cell <- function(step, delta) {
  cells <- reference_cells("full")
  cell <- cells[cells$p == 8L & cells$n_total == 440L & cells$ratio == "1:1" &
                  cells$generator == "normal" & cells$step == step &
                  cells$truth == "alternative", , drop = FALSE][1L, ]
  cell$delta <- delta
  cell
}

candidate_power <- function(step, delta) {
  cell <- reference_cell(step, delta)
  pop <- population_for_cell(cell)
  sampler <- calibrate_cell_sampler(pop, cell)
  simulations <- lapply(seq_len(cfg$reps), function(rep_id) {
    seed <- cfg$seed_base + match(step, c("metric", "scalar", "strict")) * 100000003 +
      rep_id * 1009
    list(X = draw_cell_replication(sampler, c(cell$n1, cell$n2), seed), seed = seed)
  })
  worker <- function(rep_id) {
    sim <- simulations[[rep_id]]
    observed <- tryCatch(wald_statistics(sim$X[[1L]], sim$X[[2L]], pop, step),
                         error = function(e) NULL)
    if (is.null(observed)) return(NA_real_)
    p <- permutation_pvalues(sim$X[[1L]], sim$X[[2L]], pop, step, observed,
                             cfg$permutations, sim$seed + 700000001,
                             include_lrt = FALSE)$p_studentized
    p
  }
  p <- if (cfg$cores > 1L) {
    unlist(parallel::mclapply(seq_len(cfg$reps), worker,
                             mc.cores = min(cfg$cores, cfg$reps),
                             mc.preschedule = TRUE, mc.set.seed = FALSE))
  } else vapply(seq_len(cfg$reps), worker, numeric(1))
  usable <- p[is.finite(p)]
  list(power = if (length(usable)) mean(usable <= .05) else NA_real_,
       usable = length(usable))
}

bounds <- list(metric = c(0, .60), scalar = c(0, 1.00), strict = c(0, 1.20))
history <- list()
final <- list()
for (step in names(bounds)) {
  interval <- bounds[[step]]
  for (iteration in seq_len(cfg$iterations)) {
    delta <- mean(interval)
    started <- proc.time()[["elapsed"]]
    result <- candidate_power(step, delta)
    row <- data.frame(step = step, iteration = iteration, delta = delta,
                      estimated_power = result$power, usable = result$usable,
                      reps = cfg$reps, permutations = cfg$permutations,
                      elapsed_seconds = proc.time()[["elapsed"]] - started)
    history[[length(history) + 1L]] <- row
    write_csv(do.call(rbind, history), cfg$output)
    message(sprintf("%s iteration %d: delta %.4f -> power %.3f (%d/%d usable)",
                    step, iteration, delta, result$power, result$usable, cfg$reps))
    if (!is.finite(result$power)) stop("no usable calibration replications", call. = FALSE)
    if (result$power < cfg$target) interval[[1L]] <- delta else interval[[2L]] <- delta
  }
  candidates <- do.call(rbind, history)
  candidates <- candidates[candidates$step == step, ]
  chosen <- candidates[which.min(abs(candidates$estimated_power - cfg$target)), ]
  final[[step]] <- data.frame(step = step, delta = chosen$delta,
                             estimated_power = chosen$estimated_power,
                             reps = cfg$reps, permutations = cfg$permutations)
}
final_path <- sub("\\.csv$", "-deltas.csv", cfg$output)
write_csv(do.call(rbind, final), final_path)
message("Wrote calibration history: ", cfg$output,
        "\nWrote runner deltas: ", final_path)
