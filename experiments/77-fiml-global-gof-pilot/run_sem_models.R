#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "R", "sem_models.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_sem_models.R [options]\n\n",
  "Small-rep native VM/IG FIML timing pilot for four compact SEMs.\n\n",
  "  --reps N             Replications per cell (default 10).\n",
  "  --n N                Sample size (default 120).\n",
  "  --cores N            Parallel cell workers (default up to 4).\n",
  "  --models CSV         Model ids (default all four).\n",
  "  --distributions CSV  Default normal,vm1,ig1,vm2,ig2.\n",
  "  --missingness CSV    Default complete,mcar_30.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  reps = 10L, n = 120L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  models = NULL,
  distributions = c("normal", "vm1", "ig1", "vm2", "ig2"),
  missingness = c("complete", "mcar_30"),
  seed_base = 20260820L, results_dir = NULL)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
  else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--models") opts$models <- parse_csv_arg(take())
  else if (arg == "--distributions") {
    opts$distributions <- parse_csv_arg(take())
  } else if (arg == "--missingness") {
    opts$missingness <- parse_csv_arg(take())
  } else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(opts$reps > 0L, opts$n >= 80L, opts$cores > 0L,
          opts$seed_base >= 0L)

models <- sem_model_catalog()
if (is.null(opts$models)) opts$models <- names(models)
unknown_models <- setdiff(opts$models, names(models))
if (length(unknown_models)) {
  stop("unknown models: ", paste(unknown_models, collapse = ", "),
       call. = FALSE)
}
models <- models[opts$models]
allowed_distributions <- c("normal", "vm1", "ig1", "vm2", "ig2")
allowed_missingness <- c("complete", "mcar_30")
stopifnot(all(opts$distributions %in% allowed_distributions),
          all(opts$missingness %in% allowed_missingness))

results <- opts$results_dir %||%
  file.path(script_dir, "results", "sem-model-smoke")
dir.create(results, recursive = TRUE, showWarnings = FALSE)

calibration_rows <- list()
samplers <- list()
for (model in models) {
  for (distribution in opts$distributions) {
    key <- paste(model$model_id, distribution, sep = "::")
    begin <- proc.time()[["elapsed"]]
    sampler <- tryCatch(
      sem_calibrate_sampler(model, distribution), error = function(e) e)
    seconds <- proc.time()[["elapsed"]] - begin
    ok <- !inherits(sampler, "error")
    calibration_rows[[key]] <- data.frame(
      model_id = model$model_id, model_label = model$model_label,
      distribution = distribution, calibration_ok = ok,
      calibration_seconds = seconds,
      calibration_error = if (ok) "" else conditionMessage(sampler),
      stringsAsFactors = FALSE)
    samplers[[key]] <- sampler
  }
}
calibration <- do.call(rbind, calibration_rows)
row.names(calibration) <- NULL
write_csv(calibration, file.path(results, "generator_calibration.csv"))

grid <- expand.grid(
  model_id = names(models), distribution = opts$distributions,
  missingness = opts$missingness,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid$model_label <- vapply(grid$model_id, function(id) models[[id]]$model_label,
                           character(1L))
grid$p <- vapply(grid$model_id, function(id) models[[id]]$p, integer(1L))
grid$expected_df <- vapply(
  grid$model_id, function(id) models[[id]]$expected_df, integer(1L))
grid$n <- opts$n
write_csv(grid, file.path(results, "design.csv"))

empty_rep <- function(cell, rep_id, seed) {
  data.frame(
    cell_id = cell$cell_id, model_id = cell$model_id,
    model_label = cell$model_label, distribution = cell$distribution,
    missingness = cell$missingness, p = cell$p,
    expected_df = cell$expected_df, n = cell$n, rep = rep_id, seed = seed,
    fit_ok = FALSE, fmg_ok = FALSE, mlr_ok = FALSE,
    fit_error = "", fmg_error = "", mlr_error = "",
    realized_missing_eligible = NA_real_, fitted_df = NA_integer_,
    npar = NA_integer_, fit_seconds = NA_real_, fmg_seconds = NA_real_,
    mlr_seconds = NA_real_, total_seconds = NA_real_,
    p_lrt_naive = NA_real_, p_lrt_mlr = NA_real_, p_lrt_sb = NA_real_,
    p_lrt_ss = NA_real_, p_lrt_peba4 = NA_real_, p_lrt_all = NA_real_,
    flip_status = "blocked: general FIML saturated-moment projection not shipped",
    stringsAsFactors = FALSE)
}

one_rep <- function(cell, rep_id) {
  begin <- proc.time()[["elapsed"]]
  seed <- sem_seed(opts$seed_base + cell$cell_id * 100003L + rep_id)
  out <- empty_rep(cell, rep_id, seed)
  model <- models[[cell$model_id]]
  sampler <- samplers[[paste(cell$model_id, cell$distribution, sep = "::")]]
  if (inherits(sampler, "error")) {
    out$fit_error <- paste0("generator calibration: ", conditionMessage(sampler))
    out$total_seconds <- proc.time()[["elapsed"]] - begin
    return(out)
  }
  X <- tryCatch({
    draw <- sem_draw(model, sampler, cell$n, seed)
    set.seed(seed + 700001L)
    sem_apply_missingness(draw, cell$missingness)
  }, error = function(e) e)
  if (inherits(X, "error")) {
    out$fit_error <- paste0("draw: ", conditionMessage(X))
    out$total_seconds <- proc.time()[["elapsed"]] - begin
    return(out)
  }
  out$realized_missing_eligible <-
    mean(is.na(X[, -1L, drop = FALSE]))

  fit_begin <- proc.time()[["elapsed"]]
  fit <- tryCatch(magmaan::magmaan(
    model$spec, as.data.frame(X), estimator = "FIML",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8),
    se = "none", test = "none"), error = function(e) e)
  out$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fit, "error") || !isTRUE(fit$converged)) {
    out$fit_error <- if (inherits(fit, "error")) conditionMessage(fit)
      else "FIML fit did not converge"
    out$total_seconds <- proc.time()[["elapsed"]] - begin
    return(out)
  }
  out$fit_ok <- TRUE
  if (length(fit$npar) == 1L) out$npar <- as.integer(fit$npar)

  fmg_begin <- proc.time()[["elapsed"]]
  fmg <- tryCatch(
    magmaan::fmg_tests(fit, tests = c("SB", "SS", "pEBA4", "all")),
    error = function(e) e)
  out$fmg_seconds <- proc.time()[["elapsed"]] - fmg_begin
  if (inherits(fmg, "error")) {
    out$fmg_error <- conditionMessage(fmg)
  } else {
    key <- sub("_ml$", "", fmg$label)
    lookup <- stats::setNames(fmg$p_value, key)
    out$fitted_df <- as.integer(fmg$df[[1L]])
    out$p_lrt_naive <- stats::pchisq(
      fmg$base_statistic[[1L]], fmg$df[[1L]], lower.tail = FALSE)
    value_or_na <- function(name) {
      if (name %in% names(lookup)) unname(lookup[[name]]) else NA_real_
    }
    out$p_lrt_sb <- value_or_na("sb")
    out$p_lrt_ss <- value_or_na("ss")
    out$p_lrt_peba4 <- value_or_na("peba4")
    out$p_lrt_all <- value_or_na("all")
    out$fmg_ok <- TRUE
  }

  mlr_begin <- proc.time()[["elapsed"]]
  mlr <- tryCatch(
    magmaan::magmaan_core$estimate_fiml_robust_mlr(fit),
    error = function(e) e)
  out$mlr_seconds <- proc.time()[["elapsed"]] - mlr_begin
  if (inherits(mlr, "error")) {
    out$mlr_error <- conditionMessage(mlr)
  } else {
    out$p_lrt_mlr <- stats::pchisq(
      mlr$chisq_scaled, mlr$df, lower.tail = FALSE)
    out$mlr_ok <- is.finite(out$p_lrt_mlr)
    if (!out$mlr_ok) out$mlr_error <- "MLR p-value is non-finite"
  }
  out$total_seconds <- proc.time()[["elapsed"]] - begin
  out
}

run_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf("  cell %d: %s, %s, %s", cell$cell_id, cell$model_id,
                  cell$distribution, cell$missingness))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

cat(sprintf("cells=%d reps=%d n=%d cores=%d\n", nrow(grid), opts$reps,
            opts$n, opts$cores))
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(grid)), run_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(grid)), run_cell)
}
raw <- do.call(rbind, pieces)
row.names(raw) <- NULL
wall_seconds <- proc.time()[["elapsed"]] - wall_begin
write_csv(raw, file.path(results, "replications.csv"))

group_vars <- c("model_id", "model_label", "distribution", "missingness",
                "p", "expected_df")
group_id <- interaction(raw[group_vars], drop = TRUE, lex.order = TRUE)
mean_or_na <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(as.numeric(x), na.rm = TRUE)
}
timing <- do.call(rbind, lapply(split(seq_len(nrow(raw)), group_id), function(ii) {
  z <- raw[ii, , drop = FALSE]
  data.frame(
    z[1L, group_vars, drop = FALSE],
    fit_seconds = mean_or_na(z$fit_seconds),
    fmg_seconds = mean_or_na(z$fmg_seconds),
    mlr_seconds = mean_or_na(z$mlr_seconds),
    total_seconds = mean_or_na(z$total_seconds),
    fit_success_rate = mean(z$fit_ok),
    fmg_success_rate = mean(z$fmg_ok),
    mlr_finite_rate = mean(z$mlr_ok),
    stringsAsFactors = FALSE)
}))
row.names(timing) <- NULL
write_csv(timing, file.path(results, "timing_summary.csv"))

runtime_cpu_seconds <- sum(raw$total_seconds, na.rm = TRUE)
observed_speedup <- runtime_cpu_seconds / wall_seconds
setup_seconds <- sum(calibration$calibration_seconds)
seconds_per_full_sweep <- sum(timing$total_seconds)
projection <- do.call(rbind, lapply(c(100L, 500L, 2000L), function(reps) {
  panels <- c(null_only = 1L, null_plus_power = 2L)
  do.call(rbind, lapply(names(panels), function(panel) {
    scenario_multiplier <- panels[[panel]]
    projected_runtime <- seconds_per_full_sweep * reps * scenario_multiplier /
      max(1, observed_speedup)
    data.frame(
      panel = panel, reps_per_cell = reps,
      cells = nrow(grid) * scenario_multiplier,
      total_replications = reps * nrow(grid) * scenario_multiplier,
      projected_setup_seconds = setup_seconds,
      projected_runtime_seconds = projected_runtime,
      projected_wall_seconds = setup_seconds + projected_runtime,
      projected_wall_hours = (setup_seconds + projected_runtime) / 3600,
      observed_parallel_speedup = observed_speedup,
      stringsAsFactors = FALSE)
  }))
}))
write_csv(projection, file.path(results, "timing_projection.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  reps = opts$reps, n = opts$n, cores = opts$cores, cells = nrow(grid),
  distributions = opts$distributions, missingness = opts$missingness,
  models = names(models), seed_base = opts$seed_base,
  setup_seconds = setup_seconds, runtime_wall_seconds = wall_seconds,
  runtime_cpu_seconds = runtime_cpu_seconds,
  observed_parallel_speedup = observed_speedup,
  fit_failures = sum(!raw$fit_ok), fmg_failures = sum(!raw$fmg_ok),
  mlr_nonfinite_or_failures = sum(!raw$mlr_ok),
  flip_status = unique(raw$flip_status)), packages = "magmaan")

cat(sprintf(
  "setup=%.1fs runtime_wall=%.1fs observed_speedup=%.2fx failures=%d/%d\n",
  setup_seconds, wall_seconds, observed_speedup, sum(!raw$fit_ok), nrow(raw)))
print(timing[, c("model_id", "distribution", "missingness", "total_seconds",
                 "fit_success_rate", "fmg_success_rate", "mlr_finite_rate")],
      row.names = FALSE, digits = 3)
cat("\nProjected wall time for null-only and doubled null-plus-power panels:\n")
print(projection[, c("panel", "reps_per_cell", "total_replications",
                     "projected_wall_seconds", "projected_wall_hours")],
      row.names = FALSE, digits = 3)
