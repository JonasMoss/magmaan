#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "..", "_support", "R", "missingness.R"))
source(file.path(script_dir, "R", "sem_models.R"))
source(file.path(script_dir, "R", "sem_power.R"))
source(file.path(script_dir, "R", "sem_summaries.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript run_sem_models.R [options]\n\n",
  "Paired native VM/IG FIML/ML2S pilot for five compact SEMs.\n\n",
  "  --reps N             Replications per cell (default 10).\n",
  "  --n N                Sample size (default 120).\n",
  "  --flips N            Global-score multiplier draws (default 199).\n",
  "  --cores N            Parallel cell workers (default up to 4).\n",
  "  --chunk-size N       Replications per resumable chunk (default 25).\n",
  "  --models CSV         Model ids (default all five).\n",
  "  --estimators CSV     Default FIML,ML2S.\n",
  "  --distributions CSV  Default normal,vm1,ig1,vm2,ig2.\n",
  "  --missingness CSV    Default complete,mcar_30,mar_30.\n",
  "  --truth CSV          Default null; choices null,sparse,diffuse.\n",
  "  --regions CSV        Default identified_null,nonnormal_mar_stress.\n",
  "  --power-calibration-file P  Required when truth includes power.\n",
  "  --shard-index J --shard-count K  Deterministic cell sharding.\n",
  "  --max-cells N        Optional post-filter cell cap.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  reps = 10L, n = 120L, flips = 199L, chunk_size = 25L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  models = NULL, estimators = c("FIML", "ML2S"),
  distributions = c("normal", "vm1", "ig1", "vm2", "ig2"),
  missingness = c("complete", "mcar_30", "mar_30"),
  truth = "null",
  regions = c("identified_null", "nonnormal_mar_stress"),
  power_calibration_file = NULL,
  shard_index = 1L, shard_count = 1L, max_cells = NULL,
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
  else if (arg == "--flips") opts$flips <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--chunk-size") opts$chunk_size <- as.integer(take())
  else if (arg == "--models") opts$models <- parse_csv_arg(take())
  else if (arg == "--estimators") opts$estimators <- toupper(parse_csv_arg(take()))
  else if (arg == "--distributions") {
    opts$distributions <- parse_csv_arg(take())
  } else if (arg == "--missingness") {
    opts$missingness <- parse_csv_arg(take())
  } else if (arg == "--truth") opts$truth <- parse_csv_arg(take())
  else if (arg == "--regions") opts$regions <- parse_csv_arg(take())
  else if (arg == "--power-calibration-file") {
    opts$power_calibration_file <- take()
  } else if (arg == "--shard-index") opts$shard_index <- as.integer(take())
  else if (arg == "--shard-count") opts$shard_count <- as.integer(take())
  else if (arg == "--max-cells") opts$max_cells <- as.integer(take())
  else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(opts$reps > 0L, opts$n >= 80L, opts$flips > 0L, opts$cores > 0L,
          opts$chunk_size > 0L,
          opts$shard_index > 0L, opts$shard_count > 0L,
          opts$shard_index <= opts$shard_count,
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
allowed_missingness <- c("complete", "mcar_30", "mar_30")
allowed_truth <- c("null", sem_power_alternatives)
allowed_regions <- c("identified_null", "nonnormal_mar_stress")
stopifnot(all(opts$distributions %in% allowed_distributions),
          all(opts$missingness %in% allowed_missingness),
          all(opts$truth %in% allowed_truth),
          all(opts$regions %in% allowed_regions),
          length(opts$estimators) > 0L,
          all(opts$estimators %in% c("FIML", "ML2S")))
if (!is.null(opts$max_cells)) stopifnot(opts$max_cells > 0L)

power_calibration <- if (any(opts$truth != "null")) {
  sem_read_power_calibration(opts$power_calibration_file, models, opts$n)
} else data.frame()

population_for <- function(model, truth) {
  if (truth == "null") return(sem_power_population(model, truth))
  row <- power_calibration[
    power_calibration$model_id == model$model_id &
      power_calibration$alternative == truth, , drop = FALSE]
  sem_power_population(model, truth, row$effect[[1L]])
}

results <- opts$results_dir %||%
  file.path(script_dir, "results", "sem-model-smoke")
dir.create(results, recursive = TRUE, showWarnings = FALSE)

grid <- expand.grid(
  model_id = names(models), distribution = opts$distributions,
  missingness = opts$missingness, truth = opts$truth,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
grid$analysis_region <- ifelse(
  grid$missingness == "mar_30" & grid$distribution != "normal",
  "nonnormal_mar_stress", "identified_null")
grid <- grid[grid$analysis_region %in% opts$regions, , drop = FALSE]
grid$cell_id <- seq_len(nrow(grid))
grid$pair_id <- as.integer(interaction(
  grid[c("model_id", "distribution", "missingness")],
  drop = TRUE, lex.order = TRUE))
if (opts$shard_count > 1L) {
  take_shard <- (seq_len(nrow(grid)) - 1L) %% opts$shard_count + 1L ==
    opts$shard_index
  grid <- grid[take_shard, , drop = FALSE]
}
if (!is.null(opts$max_cells)) grid <- head(grid, opts$max_cells)
if (!nrow(grid)) stop("cell filters selected no rows", call. = FALSE)
grid$model_label <- vapply(grid$model_id, function(id) models[[id]]$model_label,
                           character(1L))
grid$p <- vapply(grid$model_id, function(id) models[[id]]$p, integer(1L))
grid$expected_df <- vapply(
  grid$model_id, function(id) models[[id]]$expected_df, integer(1L))
grid$n <- opts$n
grid$alternative_effect <- 0
grid$population_fml <- 0
for (index in which(grid$truth != "null")) {
  row <- power_calibration[
    power_calibration$model_id == grid$model_id[[index]] &
      power_calibration$alternative == grid$truth[[index]], , drop = FALSE]
  grid$alternative_effect[[index]] <- row$effect[[1L]]
  grid$population_fml[[index]] <- row$achieved_fml[[1L]]
}

calibration_rows <- list()
samplers <- list()
population_keys <- unique(grid[c("model_id", "truth", "distribution")])
for (index in seq_len(nrow(population_keys))) {
    model <- models[[population_keys$model_id[[index]]]]
    truth <- population_keys$truth[[index]]
    distribution <- population_keys$distribution[[index]]
    population <- population_for(model, truth)
    key <- paste(model$model_id, truth, distribution, sep = "::")
    begin <- proc.time()[["elapsed"]]
    sampler <- tryCatch(
      sem_calibrate_sampler(
        model, distribution, Sigma = population$Sigma, mu = population$mu),
      error = function(e) e)
    seconds <- proc.time()[["elapsed"]] - begin
    ok <- !inherits(sampler, "error")
    calibration_rows[[key]] <- data.frame(
      model_id = model$model_id, model_label = model$model_label,
      truth = truth, distribution = distribution,
      alternative_effect = population$effect, calibration_ok = ok,
      calibration_seconds = seconds,
      calibration_error = if (ok) "" else conditionMessage(sampler),
      stringsAsFactors = FALSE)
    samplers[[key]] <- sampler
}
calibration <- do.call(rbind, calibration_rows)
row.names(calibration) <- NULL
write_csv(calibration, file.path(results, "generator_calibration.csv"))

design <- grid[rep(seq_len(nrow(grid)), each = length(opts$estimators)), , drop = FALSE]
design$estimator <- rep(opts$estimators, times = nrow(grid))
row.names(design) <- NULL
write_csv(design, file.path(results, "design.csv"))

config_path <- file.path(results, "run_config.csv")
code_hash <- paste(unname(tools::md5sum(c(
  file.path(script_dir, "run_sem_models.R"),
  file.path(script_dir, "R", "sem_models.R"),
  file.path(script_dir, "R", "sem_power.R"),
  file.path(script_dir, "R", "sem_summaries.R"),
  file.path(script_dir, "..", "_support", "R", "missingness.R")))),
  collapse = ":")
run_config <- data.frame(
  reps = opts$reps, n = opts$n, flips = opts$flips,
  chunk_size = opts$chunk_size, estimators = paste(opts$estimators, collapse = ","),
  models = paste(names(models), collapse = ","),
  distributions = paste(opts$distributions, collapse = ","),
  missingness = paste(opts$missingness, collapse = ","),
  truth = paste(opts$truth, collapse = ","),
  regions = paste(opts$regions, collapse = ","),
  power_calibration_file = opts$power_calibration_file %||% "",
  power_calibration_hash = if (is.null(opts$power_calibration_file)) "" else
    unname(tools::md5sum(opts$power_calibration_file)),
  shard_index = opts$shard_index, shard_count = opts$shard_count,
  max_cells = opts$max_cells %||% "",
  seed_base = opts$seed_base, code_hash = code_hash,
  stringsAsFactors = FALSE)
if (file.exists(config_path)) {
  old_config <- read.csv(config_path, stringsAsFactors = FALSE,
                         check.names = FALSE)
  if (!identical(as.character(old_config[1L, names(run_config)]),
                 as.character(run_config[1L, ]))) {
    stop("existing result configuration differs; choose a new --results-dir",
         call. = FALSE)
  }
} else {
  write_csv(run_config, config_path)
}
raw_dir <- file.path(results, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

empty_rep <- function(cell, rep_id, seed, estimator) {
  data.frame(
    cell_id = cell$cell_id, model_id = cell$model_id,
    model_label = cell$model_label, distribution = cell$distribution,
    missingness = cell$missingness, truth = cell$truth,
    analysis_region = cell$analysis_region, pair_id = cell$pair_id,
    alternative_effect = cell$alternative_effect,
    population_fml = cell$population_fml,
    estimator = estimator, p = cell$p,
    expected_df = cell$expected_df, n = cell$n, rep = rep_id, seed = seed,
    fit_ok = FALSE, fmg_ok = FALSE, mlr_ok = FALSE, flip_ok = FALSE,
    flip_expected_mammen_ok = FALSE,
    flip_corrected_ok = FALSE, flip_nominal_geometry = FALSE,
    flip_corrected_nominal_geometry = FALSE,
    fit_error = "", fmg_error = "", mlr_error = "", flip_error = "",
    flip_expected_mammen_error = "",
    flip_corrected_error = "",
    realized_missing_eligible = NA_real_, fitted_df = NA_integer_,
    npar = NA_integer_, fit_seconds = NA_real_, fmg_seconds = NA_real_,
    mlr_seconds = NA_real_, flip_seconds = NA_real_,
    flip_expected_mammen_seconds = NA_real_,
    flip_corrected_seconds = NA_real_, total_seconds = NA_real_,
    p_lrt_naive = NA_real_, p_lrt_mlr = NA_real_, p_lrt_sb = NA_real_,
    p_lrt_ss = NA_real_, p_lrt_peba4 = NA_real_, p_lrt_all = NA_real_,
    p_flip_effective = NA_real_, p_flip_expected_mammen = NA_real_,
    p_flip_corrected = NA_real_,
    p_score_sb = NA_real_,
    p_score_ss = NA_real_, p_score_peba4 = NA_real_,
    p_score_all = NA_real_,
    p_score_corrected_sb = NA_real_, p_score_corrected_ss = NA_real_,
    p_score_corrected_peba4 = NA_real_, p_score_corrected_all = NA_real_,
    flip_statistic = NA_real_,
    flip_df = NA_integer_, flip_tangent_rank = NA_integer_,
    flip_corrected_statistic = NA_real_, flip_corrected_df = NA_integer_,
    flip_corrected_tangent_rank = NA_integer_,
    stringsAsFactors = FALSE)
}

one_rep <- function(cell, rep_id) {
  begin <- proc.time()[["elapsed"]]
  seed <- sem_seed(opts$seed_base + cell$pair_id * 100003L + rep_id)
  model <- models[[cell$model_id]]
  sampler <- samplers[[paste(
    cell$model_id, cell$truth, cell$distribution, sep = "::")]]
  if (inherits(sampler, "error")) {
    return(do.call(rbind, lapply(opts$estimators, function(estimator) {
      out <- empty_rep(cell, rep_id, seed, estimator)
      out$fit_error <- paste0("generator calibration: ", conditionMessage(sampler))
      out$total_seconds <- proc.time()[["elapsed"]] - begin
      out
    })))
  }
  X <- tryCatch({
    draw <- sem_draw(model, sampler, cell$n, seed)
    set.seed(seed + 700001L)
    sem_apply_missingness(draw, cell$missingness)
  }, error = function(e) e)
  if (inherits(X, "error")) {
    return(do.call(rbind, lapply(opts$estimators, function(estimator) {
      out <- empty_rep(cell, rep_id, seed, estimator)
      out$fit_error <- paste0("draw: ", conditionMessage(X))
      out$total_seconds <- proc.time()[["elapsed"]] - begin
      out
    })))
  }
  realized_missing <- mean(is.na(X[, -1L, drop = FALSE]))
  control <- list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)
  fd <- tryCatch(magmaan::df_to_fiml_data(as.data.frame(X), model$spec),
                 error = function(e) e)
  em_begin <- proc.time()[["elapsed"]]
  em <- if (inherits(fd, "error")) fd else tryCatch(
    magmaan::magmaan_core$estimate_saturated_em_moments(fd, control = control),
    error = function(e) e)
  em_seconds <- proc.time()[["elapsed"]] - em_begin

  one_estimator <- function(estimator) {
    estimator_begin <- proc.time()[["elapsed"]]
    out <- empty_rep(cell, rep_id, seed, estimator)
    out$realized_missing_eligible <- realized_missing
    if (inherits(em, "error")) {
      out$fit_error <- paste0("Stage-1 EM: ", conditionMessage(em))
      out$total_seconds <- proc.time()[["elapsed"]] - estimator_begin +
        em_seconds / length(opts$estimators)
      return(out)
    }

    fit_begin <- proc.time()[["elapsed"]]
    fit <- tryCatch({
      if (estimator == "FIML") {
        value <- magmaan::magmaan_core$fit_fiml(
          model$spec, fd, optimizer = "nlopt-lbfgs-slsqp-fallback",
          control = control)
        value$stage1 <- em
        value
      } else {
        magmaan::magmaan_core$fit_ml2s(
          model$spec, fd, optimizer = "nlopt-lbfgs-slsqp-fallback",
          control = control, stage1 = em)
      }
    }, error = function(e) e)
    out$fit_seconds <- proc.time()[["elapsed"]] - fit_begin
    if (inherits(fit, "error") || !isTRUE(fit$converged)) {
      out$fit_error <- if (inherits(fit, "error")) conditionMessage(fit)
        else paste(estimator, "fit did not converge")
      out$total_seconds <- proc.time()[["elapsed"]] - estimator_begin +
        em_seconds / length(opts$estimators)
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
    scalar <- if (estimator == "FIML") tryCatch(
      magmaan::magmaan_core$estimate_fiml_robust_mlr(fit),
      error = function(e) e) else fit$ml2s
    out$mlr_seconds <- proc.time()[["elapsed"]] - mlr_begin
    if (inherits(scalar, "error") || is.null(scalar)) {
      out$mlr_error <- if (inherits(scalar, "error")) conditionMessage(scalar)
        else "scaled two-stage statistic is unavailable"
    } else {
      out$p_lrt_mlr <- stats::pchisq(
        scalar$chisq_scaled, scalar$df, lower.tail = FALSE)
      out$mlr_ok <- is.finite(out$p_lrt_mlr)
      if (!out$mlr_ok) out$mlr_error <- "scaled p-value is non-finite"
    }

    flip_begin <- proc.time()[["elapsed"]]
    flip <- tryCatch(
      magmaan::global_score_flip_test(
        fit, n_flips = opts$flips, seed = seed + 900001L,
        multiplier = "rademacher"),
      error = function(e) e)
    out$flip_seconds <- proc.time()[["elapsed"]] - flip_begin
    if (inherits(flip, "error")) {
      out$flip_error <- conditionMessage(flip)
    } else {
      out$p_flip_effective <- flip$p_effective
      score_fmg <- function(method, param = 4) tryCatch(
        magmaan:::infer_fmg_test(
          flip$statistic_effective, flip$df, flip$eigenvalues,
          method = method, param = param)$p_value,
        error = function(e) NA_real_)
      out$p_score_sb <- flip$p_mean_scaled
      out$p_score_ss <- score_fmg("ss")
      out$p_score_peba4 <- score_fmg("peba", 4)
      out$p_score_all <- flip$p_mixture
      out$flip_statistic <- flip$statistic_effective
      out$flip_df <- as.integer(flip$df)
      out$flip_tangent_rank <- as.integer(flip$tangent_rank)
      out$flip_ok <- is.finite(out$p_flip_effective)
      out$flip_nominal_geometry <- isTRUE(out$flip_ok) &&
        identical(out$flip_df, as.integer(cell$expected_df))
      if (!out$flip_ok) out$flip_error <- "global flip p-value is non-finite"
    }

    if (estimator == "FIML") {
      expected_mammen_begin <- proc.time()[["elapsed"]]
      expected_mammen <- tryCatch(
        magmaan::global_score_flip_test(
          fit, n_flips = opts$flips, seed = seed + 900001L,
          multiplier = "mammen", sensitivity = "expected"),
        error = function(e) e)
      out$flip_expected_mammen_seconds <-
        proc.time()[["elapsed"]] - expected_mammen_begin
      if (inherits(expected_mammen, "error")) {
        out$flip_expected_mammen_error <- conditionMessage(expected_mammen)
      } else {
        out$p_flip_expected_mammen <- expected_mammen$p_effective
        out$flip_expected_mammen_ok <-
          is.finite(out$p_flip_expected_mammen)
        if (!out$flip_expected_mammen_ok) {
          out$flip_expected_mammen_error <-
            "expected-sensitivity Mammen p-value is non-finite"
        }
      }

      corrected_begin <- proc.time()[["elapsed"]]
      corrected <- tryCatch(
        magmaan::global_score_flip_test(
          fit, n_flips = opts$flips, seed = seed + 900001L,
          multiplier = "mammen", sensitivity = "observed"),
        error = function(e) e)
      out$flip_corrected_seconds <-
        proc.time()[["elapsed"]] - corrected_begin
      if (inherits(corrected, "error")) {
        out$flip_corrected_error <- conditionMessage(corrected)
      } else {
        out$p_flip_corrected <- corrected$p_effective
        corrected_score_fmg <- function(method, param = 4) tryCatch(
          magmaan:::infer_fmg_test(
            corrected$statistic_effective, corrected$df,
            corrected$eigenvalues, method = method, param = param)$p_value,
          error = function(e) NA_real_)
        out$p_score_corrected_sb <- corrected$p_mean_scaled
        out$p_score_corrected_ss <- corrected_score_fmg("ss")
        out$p_score_corrected_peba4 <- corrected_score_fmg("peba", 4)
        out$p_score_corrected_all <- corrected$p_mixture
        out$flip_corrected_statistic <- corrected$statistic_effective
        out$flip_corrected_df <- as.integer(corrected$df)
        out$flip_corrected_tangent_rank <-
          as.integer(corrected$tangent_rank)
        out$flip_corrected_ok <- is.finite(out$p_flip_corrected)
        out$flip_corrected_nominal_geometry <-
          isTRUE(out$flip_corrected_ok) &&
          identical(out$flip_corrected_df, as.integer(cell$expected_df))
        if (!out$flip_corrected_ok) {
          out$flip_corrected_error <-
            "corrected global multiplier p-value is non-finite"
        }
      }
    }
    out$total_seconds <- proc.time()[["elapsed"]] - estimator_begin +
      em_seconds / length(opts$estimators)
    out
  }
  do.call(rbind, lapply(opts$estimators, one_estimator))
}

chunk_rows <- do.call(rbind, lapply(seq_len(nrow(grid)), function(index) {
  starts <- seq.int(1L, opts$reps, by = opts$chunk_size)
  data.frame(
    grid_index = index, first_rep = starts,
    last_rep = pmin(opts$reps, starts + opts$chunk_size - 1L))
}))

run_chunk <- function(task_index) {
  task <- chunk_rows[task_index, ]
  cell <- as.list(grid[task$grid_index, , drop = FALSE])
  chunk_path <- file.path(raw_dir, sprintf(
    "cell_%03d_reps_%05d_%05d.csv", cell$cell_id,
    task$first_rep, task$last_rep))
  expected_reps <- seq.int(task$first_rep, task$last_rep)
  expected_rows <- length(expected_reps) * length(opts$estimators)
  if (file.exists(chunk_path)) {
    old <- tryCatch(read.csv(chunk_path, stringsAsFactors = FALSE,
                             check.names = FALSE), error = function(e) NULL)
    valid <- !is.null(old) && nrow(old) == expected_rows &&
      identical(sort(unique(old$rep)), expected_reps) &&
      identical(sort(unique(old$estimator)), sort(opts$estimators)) &&
      all(old$cell_id == cell$cell_id)
    if (valid) return(old)
    stop("invalid existing result chunk: ", chunk_path, call. = FALSE)
  }
  if (task_index == 1L || task_index %% 25L == 0L) {
    message(sprintf("  chunk %d/%d: cell %d, reps %d-%d", task_index,
                    nrow(chunk_rows), cell$cell_id, task$first_rep,
                    task$last_rep))
  }
  out <- do.call(rbind, lapply(expected_reps, function(rep_id) {
    one_rep(cell, rep_id)
  }))
  write_csv(out, chunk_path)
  out
}

cat(sprintf("cells=%d chunks=%d reps=%d n=%d flips=%d cores=%d\n", nrow(grid),
            nrow(chunk_rows), opts$reps, opts$n, opts$flips, opts$cores))
wall_begin <- proc.time()[["elapsed"]]
if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  pieces <- parallel::mclapply(
    seq_len(nrow(chunk_rows)), run_chunk,
    mc.cores = min(opts$cores, nrow(chunk_rows)), mc.preschedule = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(chunk_rows)), run_chunk)
}
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep,
                 match(raw$estimator, opts$estimators)), ]
row.names(raw) <- NULL
wall_seconds <- proc.time()[["elapsed"]] - wall_begin
write_csv(raw, file.path(results, "replications.csv"))
sem_write_method_summaries(raw, results)

group_vars <- c("model_id", "model_label", "estimator", "distribution",
                "missingness", "analysis_region", "truth",
                "alternative_effect", "population_fml", "p", "expected_df")
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
    flip_seconds = mean_or_na(z$flip_seconds),
    flip_expected_mammen_seconds = mean_or_na(
      z$flip_expected_mammen_seconds),
    flip_corrected_seconds = mean_or_na(z$flip_corrected_seconds),
    total_seconds = mean_or_na(z$total_seconds),
    fit_success_rate = mean(z$fit_ok),
    fmg_success_rate = mean(z$fmg_ok),
    mlr_finite_rate = mean(z$mlr_ok),
    flip_success_rate = mean(z$flip_ok),
    flip_nominal_geometry_rate = mean(z$flip_nominal_geometry),
    flip_corrected_success_rate = if (z$estimator[[1L]] == "FIML") {
      mean(z$flip_corrected_ok)
    } else NA_real_,
    flip_corrected_nominal_geometry_rate =
      if (z$estimator[[1L]] == "FIML") {
        mean(z$flip_corrected_nominal_geometry)
      } else NA_real_,
    stringsAsFactors = FALSE)
}))
row.names(timing) <- NULL
write_csv(timing, file.path(results, "timing_summary.csv"))

runtime_cpu_seconds <- sum(raw$total_seconds, na.rm = TRUE)
observed_speedup <- runtime_cpu_seconds / wall_seconds
setup_seconds <- sum(calibration$calibration_seconds)
seconds_per_full_sweep <- sum(timing$total_seconds)
projection <- do.call(rbind, lapply(c(100L, 500L, 2000L), function(reps) {
  projected_runtime <- seconds_per_full_sweep * reps /
    max(1, observed_speedup)
  data.frame(
    panel = "selected_design", reps_per_cell = reps,
    cells = nrow(grid) * length(opts$estimators),
    total_replications = reps * nrow(grid) * length(opts$estimators),
    projected_setup_seconds = setup_seconds,
    projected_runtime_seconds = projected_runtime,
    projected_wall_seconds = setup_seconds + projected_runtime,
    projected_wall_hours = (setup_seconds + projected_runtime) / 3600,
    observed_parallel_speedup = observed_speedup,
    stringsAsFactors = FALSE)
}))
write_csv(projection, file.path(results, "timing_projection.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  reps = opts$reps, n = opts$n, flips = opts$flips,
  chunk_size = opts$chunk_size,
  cores = opts$cores, cells = nrow(grid) * length(opts$estimators),
  distributions = opts$distributions, missingness = opts$missingness,
  truth = opts$truth, regions = opts$regions,
  models = names(models), estimators = opts$estimators,
  power_calibration_file = opts$power_calibration_file %||% "",
  alpha_convention = "p <= 0.05; p < 0.05 retained as sensitivity",
  seed_base = opts$seed_base,
  setup_seconds = setup_seconds, runtime_wall_seconds = wall_seconds,
  runtime_cpu_seconds = runtime_cpu_seconds,
  observed_parallel_speedup = observed_speedup,
  fit_failures = sum(!raw$fit_ok), fmg_failures = sum(!raw$fmg_ok),
  mlr_nonfinite_or_failures = sum(!raw$mlr_ok),
  flip_failures = sum(!raw$flip_ok),
  flip_non_nominal_geometry = sum(raw$flip_ok & !raw$flip_nominal_geometry),
  flip_expected_mammen_failures = sum(
    raw$estimator == "FIML" & !raw$flip_expected_mammen_ok),
  flip_corrected_failures = sum(
    raw$estimator == "FIML" & !raw$flip_corrected_ok),
  flip_corrected_non_nominal_geometry = sum(
    raw$estimator == "FIML" & raw$flip_corrected_ok &
      !raw$flip_corrected_nominal_geometry)),
  packages = "magmaan")

cat(sprintf(
  "setup=%.1fs runtime_wall=%.1fs observed_speedup=%.2fx failures=%d/%d\n",
  setup_seconds, wall_seconds, observed_speedup, sum(!raw$fit_ok), nrow(raw)))
print(timing[, c("model_id", "distribution", "missingness", "truth",
                 "total_seconds",
                 "estimator",
                 "fit_success_rate", "fmg_success_rate", "mlr_finite_rate",
                 "flip_success_rate", "flip_nominal_geometry_rate",
                 "flip_corrected_success_rate",
                 "flip_corrected_nominal_geometry_rate")],
      row.names = FALSE, digits = 3)
cat("\nProjected wall time for the selected design:\n")
print(projection[, c("panel", "reps_per_cell", "total_replications",
                     "projected_wall_seconds", "projected_wall_hours")],
      row.names = FALSE, digits = 3)
