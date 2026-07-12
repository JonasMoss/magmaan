#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "expansion_design.R"))   # draw_block, severe consts
source(file.path(script_dir, "R", "dimension_design.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_dimension.R [--smoke|--probe|--big|--sandwich] [options]\n\n",
  "Does the score-base FMG advantage persist as ambient dimension grows?\n",
  "Holds tested df = 8 fixed; grows p in {10,20,30}; severe copula (vm/pl);\n",
  "two sample-adequacy regimes (per-group n/q ~ 2.5 tight, ~6 adequate).\n\n",
  "Options:\n",
  "  --smoke          One cell, 4 reps (default).\n",
  "  --probe          12-cell grid (p 10/20/30, homogeneous).\n",
  "  --big            32-cell grid (p 10/20/30/40, both geometries).\n",
  "  --sandwich       Big grid, 200 reps and one flip, direct-score audit.\n",
  "  --reps N         Replications per cell (probe 200, big 1000).\n",
  "  --flips N        Random flips per replication (default: 499).\n",
  "  --cores N        Parallel cell workers.\n",
  "  --seed-base N    Deterministic seed base.\n",
  "  --results-dir P  Output directory.\n",
  "  --help           Show this help.\n", sep = "")

opts <- list(mode = "smoke", reps = NULL, flips = NULL,
             cores = max(1L, parallel::detectCores() - 2L),
             seed_base = 20260714L, results_dir = NULL)
args <- commandArgs(TRUE); i <- 1L
take <- function() { i <<- i + 1L; args[[i]] }
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") { usage(); quit(status = 0L) }
  else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--probe") opts$mode <- "probe"
  else if (a == "--big") opts$mode <- "big"
  else if (a == "--sandwich") opts$mode <- "sandwich"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--flips") opts$flips <- as.integer(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") opts$seed_base <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (is.null(opts$reps)) opts$reps <- switch(opts$mode,
  smoke = 4L, big = 1000L, sandwich = 200L, 200L)
if (is.null(opts$flips)) opts$flips <- if (opts$mode == "smoke") 39L else
  if (opts$mode == "sandwich") 1L else 499L
results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

adequacy_ratio <- c(tight = 2.5, adequate = 6.0)
large_grid <- opts$mode %in% c("big", "sandwich")
factors <- if (large_grid) c(2L, 4L, 6L, 8L) else c(2L, 4L, 6L)
hets <- if (large_grid) c("homogeneous", "geometry") else "homogeneous"
grid <- expand.grid(n_factors = factors, adequacy = names(adequacy_ratio),
                    distribution = c("vm", "pl"), heterogeneity = hets,
                    stringsAsFactors = FALSE)
grid$p <- 5L * grid$n_factors
grid$ratio <- adequacy_ratio[grid$adequacy]
grid$n_total <- mapply(dim_total_n, grid$n_factors, grid$ratio)
if (opts$mode == "smoke") grid <- grid[grid$n_factors == 2L &
  grid$adequacy == "adequate" & grid$distribution == "vm", , drop = FALSE]
grid$cell_id <- seq_len(nrow(grid))

fmg_transforms <- c("SB", "SS", "MV", "SF", "pEBA4", "ALL")
score_methods <- c(SB = "sb", SS = "ss", MV = "mv", SF = "scaled_f",
                   pEBA4 = "peba", ALL = "all")
score_param <- c(SB = 4, SS = 4, MV = 4, SF = 4, pEBA4 = 4, ALL = 4)
p_columns <- c("p_flip_basic", "p_flip_effective", "p_flip_standardized",
               "p_score_sandwich", paste0("p_score_", tolower(fmg_transforms)),
               paste0("p_lr_", tolower(fmg_transforms)))

empty_rep <- function(rep_id, error) {
  out <- data.frame(rep = rep_id, ok = FALSE, error = error)
  for (nm in p_columns) out[[nm]] <- NA_real_
  out$score_eigen_mean <- NA_real_
  out$score_eigen_cv <- NA_real_
  out$score_eigen_ratio <- NA_real_
  out$sandwich_available <- NA
  out$sandwich_condition <- NA_real_
  out
}

one_rep <- function(cell, rep_id) {
  F <- cell$n_factors
  ov <- dim_ov(F); pop <- dim_population(F, cell$heterogeneity); pair <- dim_specs(F)
  sizes <- c(cell$n_total %/% 2L, cell$n_total - cell$n_total %/% 2L)
  set.seed(opts$seed_base + cell$cell_id * 100000L + rep_id)
  blocks <- lapply(1:2, function(g) {
    m <- flip_expansion_draw_block(sizes[g], pop$mu[[g]], pop$Sigma[[g]], cell$distribution)
    colnames(m) <- ov; m })
  dat <- data.frame(rbind(blocks[[1]], blocks[[2]]),
                    school = rep(c("A", "B"), sizes), check.names = FALSE)
  fit <- tryCatch(lapply(pair, magmaan, data = dat, estimator = "ML",
                         optimizer = "nlopt-lbfgs-slsqp-fallback"), error = function(e) e)
  if (inherits(fit, "error")) return(empty_rep(rep_id, conditionMessage(fit)))
  flip <- tryCatch(score_flip_test(fit$configural, fit$restricted, dat,
                    n_flips = opts$flips,
                    seed = opts$seed_base + cell$cell_id * 1000000L + rep_id),
                   error = function(e) e)
  if (inherits(flip, "error")) return(empty_rep(rep_id, conditionMessage(flip)))

  score_fmg <- function(k) tryCatch(magmaan:::infer_fmg_test(
    flip$statistic_effective, flip$df, flip$eigenvalues,
    method = score_methods[[k]], param = score_param[[k]])$p_value,
    error = function(e) NA_real_)
  p_score <- setNames(vapply(fmg_transforms, score_fmg, numeric(1L)), fmg_transforms)

  rb <- lapply(c("A", "B"), function(g) as.matrix(dat[dat$school == g, ov, drop = FALSE]))
  fmg <- if (opts$mode == "sandwich") NULL else tryCatch(fmg_nested(
    fit$configural, fit$restricted, data = rb, tests = fmg_transforms,
    A.method = "exact"), error = function(e) NULL)
  p_lr <- setNames(rep(NA_real_, length(fmg_transforms)), fmg_transforms)
  if (!is.null(fmg)) {
    key <- sub("_ml$", "", fmg$label)
    want <- c(SB = "sb", SS = "ss", MV = "mv", SF = "sf", pEBA4 = "peba4", ALL = "all")
    hit <- match(want, key)
    p_lr[!is.na(hit)] <- fmg$p_value[hit[!is.na(hit)]]
  }

  out <- data.frame(rep = rep_id, ok = TRUE, error = "",
    p_flip_basic = flip$p_basic, p_flip_effective = flip$p_effective,
    p_flip_standardized = flip$p_standardized,
    p_score_sandwich = flip$p_sandwich,
    score_eigen_mean = mean(flip$eigenvalues),
    score_eigen_cv = sd(flip$eigenvalues) / mean(flip$eigenvalues),
    score_eigen_ratio = max(flip$eigenvalues) / min(flip$eigenvalues),
    sandwich_available = flip$sandwich_available,
    sandwich_condition = flip$sandwich_condition)
  for (t in fmg_transforms) out[[paste0("p_score_", tolower(t))]] <- p_score[[t]]
  for (t in fmg_transforms) out[[paste0("p_lr_", tolower(t))]] <- p_lr[[t]]
  out
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  rows <- lapply(seq_len(opts$reps), function(r) tryCatch(
    one_rep(cell, r), error = function(e) empty_rep(r, conditionMessage(e))))
  out <- do.call(rbind, rows)
  for (nm in names(cell)) out[[nm]] <- cell[[nm]]
  out
}

cat(sprintf("mode=%s cells=%d reps=%d flips=%d cores=%d\n",
            opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores))
t0 <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L && nrow(grid) > 1L) {
  pieces <- parallel::mclapply(seq_len(nrow(grid)), run_cell,
                               mc.cores = min(opts$cores, nrow(grid)),
                               mc.preschedule = TRUE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), function(k) {
    cat(sprintf("  cell %d/%d\n", k, nrow(grid))); flush.console(); run_cell(k)
  })
}
raw <- do.call(rbind, pieces)
output_stem <- if (opts$mode == "sandwich") "dimension_sandwich" else "dimension"
write.csv(raw, file.path(results_dir, paste0(output_stem, "_replications.csv")),
          row.names = FALSE)

summarize_cell <- function(x) {
  ok <- x[x$ok, , drop = FALSE]
  out <- data.frame(
    reps = nrow(x), reps_ok = nrow(ok), failure_rate = mean(!x$ok),
    sandwich_available_rate = mean(ok$sandwich_available),
    median_sandwich_condition = median(ok$sandwich_condition, na.rm = TRUE),
    median_score_eigen_mean = median(ok$score_eigen_mean, na.rm = TRUE),
    median_score_eigen_cv = median(ok$score_eigen_cv, na.rm = TRUE),
    median_score_eigen_ratio = median(ok$score_eigen_ratio, na.rm = TRUE))
  for (nm in p_columns)
    out[[paste0("rejection_", sub("^p_", "", nm))]] <-
      if (nrow(ok)) mean(ok[[nm]] < .05, na.rm = TRUE) else NA_real_
  out
}
split_raw <- split(raw, raw$cell_id)
summary <- do.call(rbind, lapply(split_raw, summarize_cell))
summary$cell_id <- as.integer(names(split_raw))
summary <- merge(grid, summary, by = "cell_id", sort = TRUE)
write.csv(summary, file.path(results_dir, paste0(output_stem, "_summary.csv")),
          row.names = FALSE)

metadata <- data.frame(
  key = c("mode", "cells", "reps", "flips", "cores", "seed_base",
          "elapsed_seconds", "magmaan_version", "R_version"),
  value = c(opts$mode, nrow(grid), opts$reps, opts$flips, opts$cores,
            opts$seed_base, proc.time()[["elapsed"]] - t0,
            as.character(packageVersion("magmaan")), R.version.string))
write.csv(metadata, file.path(results_dir, paste0(output_stem, "_metadata.csv")),
          row.names = FALSE)
cat(sprintf("wrote dimension results to %s (%.1fs)\n", results_dir,
            proc.time()[["elapsed"]] - t0))
