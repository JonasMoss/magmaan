#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "expansion_design.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_multiplier_probe.R [profile] [options]\n\n",
  "Focused multiplier-score replay of the difficult q=8 severe-copula cells.\n\n",
  "Profiles:\n",
  "  --smoke          One cell, 4 replications, 39 multipliers (default).\n",
  "  --probe          Eight cells, 200 replications, 499 multipliers.\n\n",
  "  --refine         Two-point moment curve plus centered/studentized Mammen.\n\n",
  "  --confirm        Fresh normal/t5/skew confirmation, 500 replications.\n\n",
  "  --confirm-power  Matched 12-cell sparse-effect gate, 300 replications.\n\n",
  "Options:\n",
  "  --reps N         Override replications per cell.\n",
  "  --multipliers N  Override multiplier draws per method and replication.\n",
  "  --weights L      Comma-separated rademacher,mammen,gaussian,\n",
  "                   centered-exponential.\n",
  "  --gammas L       Two-point third moments for --refine. Default:\n",
  "                   0.5,0.75,0.9,1.\n",
  "  --n-total L      Comma-separated total N values. Default: 60,200.\n",
  "  --alternative A  sparse or dense for --confirm-power. Default: sparse.\n",
  "  --effect X       Loading-effect norm for --confirm-power. Default: 0.5.\n",
  "  --cores N        Parallel cell workers.\n",
  "  --seed-base N    Deterministic seed base. Default: 20260713, or\n",
  "                   20260720/20260721 for confirmation null/power.\n",
  "  --max-cells N    Retain only the first N selected cells.\n",
  "  --results-dir P  Output directory. Default: experiment results/.\n",
  "  --help           Show this help.\n", sep = "")

parse_csv <- function(x) strsplit(x, ",", fixed = TRUE)[[1L]]
opts <- list(
  mode = "smoke", reps = NULL, multipliers = NULL,
  weights = c("rademacher", "mammen", "gaussian",
              "centered-exponential"),
  gammas = c(0.5, 0.75, 0.9, 1),
  n_total = c(60L, 200L),
  alternative = NULL, effect = NULL,
  cores = max(1L, parallel::detectCores() - 2L),
  seed_base = 20260713L, max_cells = NULL, results_dir = NULL)
args <- commandArgs(TRUE)
seed_base_explicit <- FALSE
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) {
    stop("missing value after ", args[[i - 1L]], call. = FALSE)
  }
  args[[i]]
}
while (i <= length(args)) {
  a <- args[[i]]
  if (a == "--help") {
    usage()
    quit(status = 0L)
  } else if (a == "--smoke") opts$mode <- "smoke"
  else if (a == "--probe") opts$mode <- "probe"
  else if (a == "--refine") opts$mode <- "refine"
  else if (a == "--confirm") opts$mode <- "confirm"
  else if (a == "--confirm-power") opts$mode <- "confirm-power"
  else if (a == "--reps") opts$reps <- as.integer(take())
  else if (a == "--multipliers") opts$multipliers <- as.integer(take())
  else if (a == "--weights") opts$weights <- parse_csv(take())
  else if (a == "--gammas") opts$gammas <- as.numeric(parse_csv(take()))
  else if (a == "--n-total") opts$n_total <- as.integer(parse_csv(take()))
  else if (a == "--alternative") opts$alternative <- take()
  else if (a == "--effect") opts$effect <- as.numeric(take())
  else if (a == "--cores") opts$cores <- as.integer(take())
  else if (a == "--seed-base") {
    opts$seed_base <- as.integer(take())
    seed_base_explicit <- TRUE
  }
  else if (a == "--max-cells") opts$max_cells <- as.integer(take())
  else if (a == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", a, call. = FALSE)
  i <- i + 1L
}
if (!seed_base_explicit && opts$mode %in% c("confirm", "confirm-power")) {
  opts$seed_base <- if (opts$mode == "confirm") 20260720L else 20260721L
}
if (is.null(opts$reps)) {
  opts$reps <- switch(
    opts$mode, smoke = 4L, confirm = 500L, `confirm-power` = 300L, 200L)
}
if (is.null(opts$multipliers)) {
  opts$multipliers <- if (opts$mode == "smoke") 39L else 499L
}
allowed_weights <- c(
  "rademacher", "mammen", "gaussian", "centered-exponential")
if (!length(opts$weights) || any(!opts$weights %in% allowed_weights)) {
  stop("--weights contains an unknown multiplier law", call. = FALSE)
}
if (!length(opts$gammas) || any(!is.finite(opts$gammas)) ||
    any(opts$gammas < 0) || any(opts$gammas > 1e6)) {
  stop("--gammas must be finite and non-negative", call. = FALSE)
}
if (anyNA(c(opts$reps, opts$multipliers, opts$cores)) ||
    any(c(opts$reps, opts$multipliers, opts$cores) < 1L)) {
  stop("replications, multipliers, and cores must be positive", call. = FALSE)
}
if (!length(opts$n_total) || anyNA(opts$n_total) ||
    any(!opts$n_total %in% c(60L, 100L, 200L))) {
  stop("--n-total must be drawn from 60,100,200", call. = FALSE)
}
if (opts$mode == "confirm-power") {
  opts$alternative <- opts$alternative %||% "sparse"
  opts$effect <- opts$effect %||% 0.5
  if (!opts$alternative %in% c("sparse", "dense")) {
    stop("--alternative must be sparse or dense", call. = FALSE)
  }
  if (length(opts$effect) != 1L || !is.finite(opts$effect) ||
      opts$effect <= 0) {
    stop("--effect must be one finite positive number", call. = FALSE)
  }
} else if (!is.null(opts$alternative) || !is.null(opts$effect)) {
  stop("--alternative and --effect require --confirm-power", call. = FALSE)
}

results_dir <- opts$results_dir %||% file.path(script_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

if (opts$mode %in% c("confirm", "confirm-power")) {
  # A fresh seed base makes this independent of the discovery stress block.
  # IDs retain the full regular-grid positions for deterministic extension.
  regular_grid <- expand.grid(
    df = c(1L, 4L, 8L), n_total = c(60L, 100L, 200L),
    balance = c("1:1", "1:3"),
    heterogeneity = c("homogeneous", "scale", "geometry"),
    distribution = c("normal", "t5", "skew"),
    stringsAsFactors = FALSE)
  regular_grid$source_cell_id <- seq_len(nrow(regular_grid))
  grid <- subset(
    regular_grid,
    df == 8L & n_total %in% opts$n_total & balance == "1:1" &
      heterogeneity %in% c("homogeneous", "geometry"))
} else {
  # Preserve experiment 62's original stress-cell IDs so the data and
  # Rademacher streams replay the existing stress run exactly.
  stress_grid <- expand.grid(
    df = c(1L, 8L), n_total = c(60L, 100L, 200L),
    balance = "1:1", heterogeneity = c("homogeneous", "geometry"),
    distribution = c("vm", "pl"), stringsAsFactors = FALSE)
  stress_grid$stress_cell_id <- seq_len(nrow(stress_grid))
  stress_grid$source_cell_id <- stress_grid$stress_cell_id
  grid <- subset(stress_grid, df == 8L & n_total %in% opts$n_total)
}
grid$alternative <- if (opts$mode == "confirm-power") {
  opts$alternative
} else "null"
grid$effect <- if (opts$mode == "confirm-power") opts$effect else 0
if (opts$mode == "smoke") grid <- head(grid, 1L)
if (!is.null(opts$max_cells)) grid <- head(grid, opts$max_cells)
grid$cell_id <- seq_len(nrow(grid))
if (!nrow(grid)) stop("no multiplier-probe cells selected", call. = FALSE)

specs <- flip_expansion_specs(8L)
safe_name <- function(x) {
  gsub(".", "_", gsub("-", "_", x, fixed = TRUE), fixed = TRUE)
}
column_for <- function(arm) paste0("p_", safe_name(arm))
time_column_for <- function(arm) paste0("seconds_", safe_name(arm))
if (opts$mode == "refine") {
  gamma_labels <- format(opts$gammas, trim = TRUE, scientific = FALSE)
  arm_specs <- data.frame(
    arm = paste0("two_point_", gamma_labels),
    multiplier = "two-point",
    two_point_skewness = opts$gammas,
    center = FALSE,
    studentization = "none",
    third_moment = opts$gammas,
    fourth_moment = 1 + opts$gammas^2,
    stringsAsFactors = FALSE)
  arm_specs <- rbind(
    arm_specs,
    data.frame(
      arm = c("mammen_centered", "mammen_weighted_meat",
              "mammen_centered_weighted_meat"),
      multiplier = "mammen", two_point_skewness = 1,
      center = c(TRUE, FALSE, TRUE),
      studentization = c("none", "weighted-meat", "weighted-meat"),
      third_moment = 1, fourth_moment = 2,
      stringsAsFactors = FALSE))
} else if (opts$mode %in% c("confirm", "confirm-power")) {
  arm_specs <- data.frame(
    arm = c("mammen", "mammen_centered"),
    multiplier = "mammen", two_point_skewness = 1,
    center = c(FALSE, TRUE), studentization = "none",
    third_moment = 1, fourth_moment = 2,
    stringsAsFactors = FALSE)
} else {
  moment_lookup <- data.frame(
    multiplier = c(
      "rademacher", "mammen", "gaussian", "centered-exponential"),
    third_moment = c(0, 1, 0, 2),
    fourth_moment = c(1, 2, 3, 9),
    stringsAsFactors = FALSE)
  arm_specs <- data.frame(
    arm = opts$weights, multiplier = opts$weights,
    two_point_skewness = 1, center = FALSE,
    studentization = "none", stringsAsFactors = FALSE)
  arm_specs <- merge(
    arm_specs, moment_lookup, by = "multiplier", sort = FALSE)
  arm_specs <- arm_specs[match(opts$weights, arm_specs$arm), ]
}
p_columns <- vapply(arm_specs$arm, column_for, character(1))
time_columns <- vapply(arm_specs$arm, time_column_for, character(1))

empty_rep <- function(rep_id, error) {
  out <- data.frame(
    rep = rep_id, ok = FALSE, error = error,
    p_sandwich = NA_real_, p_peba4 = NA_real_, p_all = NA_real_,
    statistic_effective = NA_real_, statistic_sandwich = NA_real_,
    sandwich_condition = NA_real_, gaussian_all_abs_gap = NA_real_,
    fit_seconds = NA_real_, total_seconds = NA_real_,
    stringsAsFactors = FALSE)
  for (name in p_columns) out[[name]] <- NA_real_
  for (name in time_columns) out[[name]] <- NA_real_
  out
}

one_rep <- function(cell, rep_id) {
  started <- proc.time()[["elapsed"]]
  data_seed <- opts$seed_base + cell$source_cell_id * 100000L + rep_id
  dat <- flip_expansion_draw_data(
    cell$n_total, cell$balance, cell$heterogeneity, cell$distribution,
    data_seed, df = 8L,
    alternative = cell$alternative, effect = cell$effect)
  fit_started <- proc.time()[["elapsed"]]
  fit0 <- tryCatch(
    magmaan(
      specs$restricted, data = dat, estimator = "ML",
      optimizer = "nlopt-lbfgs-slsqp-fallback"),
    error = function(e) e)
  fit_seconds <- proc.time()[["elapsed"]] - fit_started
  if (inherits(fit0, "error")) {
    return(empty_rep(rep_id, conditionMessage(fit0)))
  }

  multiplier_seed <-
    opts$seed_base + cell$source_cell_id * 1000000L + rep_id
  tests <- vector("list", nrow(arm_specs))
  names(tests) <- arm_specs$arm
  elapsed <- setNames(rep(NA_real_, nrow(arm_specs)), arm_specs$arm)
  for (j in seq_len(nrow(arm_specs))) {
    arm <- arm_specs[j, ]
    multiplier_started <- proc.time()[["elapsed"]]
    tests[[arm$arm]] <- tryCatch(
      score_flip_test(
        specs$configural, fit0, data = dat, n_flips = opts$multipliers,
        seed = multiplier_seed, calibration = "effective",
        multiplier = arm$multiplier,
        two_point_skewness = arm$two_point_skewness,
        center_multiplier_scores = arm$center,
        multiplier_studentization = arm$studentization),
      error = function(e) e)
    elapsed[[arm$arm]] <- proc.time()[["elapsed"]] - multiplier_started
  }
  errors <- vapply(
    tests, function(x) {
      if (inherits(x, "error")) conditionMessage(x) else ""
    }, character(1))
  if (any(nzchar(errors))) {
    return(empty_rep(
      rep_id,
      paste(names(errors)[nzchar(errors)], errors[nzchar(errors)],
            sep = ": ", collapse = " | ")))
  }

  reference <- tests[[1L]]
  p_peba4 <- tryCatch(
    magmaan:::infer_fmg_test(
      reference$statistic_effective, reference$df,
      reference$eigenvalues, method = "peba", param = 4)$p_value,
    error = function(e) NA_real_)
  out <- data.frame(
    rep = rep_id, ok = TRUE, error = "",
    p_sandwich = reference$p_sandwich,
    p_peba4 = p_peba4,
    p_all = reference$p_mixture,
    statistic_effective = reference$statistic_effective,
    statistic_sandwich = reference$statistic_sandwich,
    sandwich_condition = reference$sandwich_condition,
    gaussian_all_abs_gap = if (any(arm_specs$multiplier == "gaussian")) {
      gaussian_arm <- arm_specs$arm[arm_specs$multiplier == "gaussian"][[1L]]
      abs(tests[[gaussian_arm]]$p_effective - reference$p_mixture)
    } else NA_real_,
    fit_seconds = fit_seconds,
    total_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE)
  for (j in seq_len(nrow(arm_specs))) {
    arm <- arm_specs[j, ]
    test <- tests[[arm$arm]]
    out[[column_for(arm$arm)]] <-
      if (arm$studentization == "weighted-meat") {
        test$p_multiplier_studentized
      } else test$p_effective
    out[[time_column_for(arm$arm)]] <- elapsed[[arm$arm]]
  }
  out
}

run_cell <- function(k) {
  cell <- as.list(grid[k, , drop = FALSE])
  rows <- lapply(seq_len(opts$reps), function(rep_id) {
    tryCatch(
      one_rep(cell, rep_id),
      error = function(e) empty_rep(rep_id, conditionMessage(e)))
  })
  out <- do.call(rbind, rows)
  for (name in names(cell)) out[[name]] <- cell[[name]]
  out
}

cat(sprintf(
  "multiplier probe: %d cells x %d reps x %d draws x %d laws; %d cores\n",
  nrow(grid), opts$reps, opts$multipliers, nrow(arm_specs), opts$cores))
started <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L && nrow(grid) > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(grid)), run_cell,
    mc.cores = min(opts$cores, nrow(grid)),
    mc.preschedule = TRUE, mc.set.seed = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), function(k) {
    cat(sprintf("  cell %d/%d\n", k, nrow(grid)))
    flush.console()
    run_cell(k)
  })
}
raw <- do.call(rbind, pieces)
prefix <- switch(
  opts$mode, smoke = "multiplier_smoke", probe = "multiplier_probe",
  refine = "multiplier_refine", confirm = "multiplier_confirm",
  `confirm-power` = "multiplier_power")
write.csv(
  raw, file.path(results_dir, paste0(prefix, "_replications.csv")),
  row.names = FALSE)

safe_mean <- function(x) {
  if (any(is.finite(x))) mean(x, na.rm = TRUE) else NA_real_
}
summarize_cell <- function(x) {
  ok <- x[x$ok, , drop = FALSE]
  out <- data.frame(
    reps = nrow(x), reps_ok = nrow(ok), failure_rate = mean(!x$ok),
    rejection_sandwich = safe_mean(ok$p_sandwich < .05),
    rejection_peba4 = safe_mean(ok$p_peba4 < .05),
    rejection_all = safe_mean(ok$p_all < .05),
    mean_gaussian_all_abs_gap = safe_mean(ok$gaussian_all_abs_gap),
    median_sandwich_condition = stats::median(
      ok$sandwich_condition, na.rm = TRUE),
    median_fit_ms = 1000 * stats::median(ok$fit_seconds, na.rm = TRUE),
    median_total_ms = 1000 * stats::median(ok$total_seconds, na.rm = TRUE))
  for (arm in arm_specs$arm) {
    out[[paste0("rejection_", safe_name(arm))]] <-
      safe_mean(ok[[column_for(arm)]] < .05)
    out[[paste0("median_", safe_name(arm), "_ms")]] <-
      1000 * stats::median(ok[[time_column_for(arm)]], na.rm = TRUE)
  }
  out
}
split_raw <- split(raw, raw$cell_id)
summary <- do.call(rbind, lapply(split_raw, summarize_cell))
summary$cell_id <- as.integer(names(split_raw))
summary <- merge(grid, summary, by = "cell_id", sort = TRUE)
write.csv(
  summary, file.path(results_dir, paste0(prefix, "_summary.csv")),
  row.names = FALSE)

moments <- if (opts$mode %in% c("refine", "confirm", "confirm-power")) {
  transform(
    arm_specs, mean = 0, variance = 1,
    weighted_meat = studentization == "weighted-meat")
} else {
  data.frame(
    multiplier = arm_specs$multiplier,
    mean = 0, variance = 1,
    third_moment = arm_specs$third_moment,
    fourth_moment = arm_specs$fourth_moment)
}
write.csv(
  moments,
  file.path(results_dir, paste0(prefix, "_moments.csv")),
  row.names = FALSE)
metadata <- data.frame(
  key = c(
    "mode", "cells", "reps", "multipliers", "arms", "cores",
    "seed_base", "alternative", "effect", "elapsed_seconds",
    "magmaan_version", "R_version"),
  value = c(
    opts$mode, nrow(grid), opts$reps, opts$multipliers,
    paste(arm_specs$arm, collapse = ","), opts$cores, opts$seed_base,
    unique(grid$alternative), unique(grid$effect),
    proc.time()[["elapsed"]] - started,
    as.character(packageVersion("magmaan")), R.version.string))
write.csv(
  metadata, file.path(results_dir, paste0(prefix, "_metadata.csv")),
  row.names = FALSE)
cat(sprintf(
  "wrote %s results to %s (%.1fs)\n",
  prefix, results_dir, proc.time()[["elapsed"]] - started))
