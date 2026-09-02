#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "..", "_support", "R", "missingness.R"))
source(file.path(script_dir, "R", "sem_models.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript check_lrt_score_crossing.R [options]\n\n",
  "Crosses the FIML LRT and projected-score statistics with each other's\n",
  "estimated FMG spectra. The four pEBA(4) cells isolate the statistic from\n",
  "the spectrum while holding the fitted model and data fixed.\n\n",
  "  --reps N             Replications per cell (default 100).\n",
  "  --n N                Sample size (default 120).\n",
  "  --cores N            Parallel cell workers (default up to 4).\n",
  "  --models CSV         Model ids (default one_factor_6).\n",
  "  --distributions CSV  Default normal,vm2,ig2.\n",
  "  --missingness CSV    Default complete,mar_30.\n",
  "  --sensitivity NAME   Score sensitivity: expected or observed.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  reps = 100L,
  n = 120L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  models = "one_factor_6",
  distributions = c("normal", "vm2", "ig2"),
  missingness = c("complete", "mar_30"),
  sensitivity = "expected",
  seed_base = 20260902L,
  results_dir = NULL)
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
  } else if (arg == "--sensitivity") opts$sensitivity <- take()
  else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}

stopifnot(
  opts$reps > 0L,
  opts$n >= 80L,
  opts$cores > 0L,
  opts$sensitivity %in% c("expected", "observed"),
  all(opts$distributions %in% c("normal", "vm1", "ig1", "vm2", "ig2")),
  all(opts$missingness %in% c("complete", "mcar_30", "mar_30")))

models <- sem_model_catalog()
unknown_models <- setdiff(opts$models, names(models))
if (length(unknown_models)) {
  stop("unknown models: ", paste(unknown_models, collapse = ", "),
       call. = FALSE)
}
models <- models[opts$models]
results <- opts$results_dir %||% file.path(
  script_dir, "results", paste0("lrt-score-crossing-", opts$sensitivity))
dir.create(results, recursive = TRUE, showWarnings = FALSE)

grid <- expand.grid(
  model_id = names(models),
  distribution = opts$distributions,
  missingness = opts$missingness,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid$pair_id <- as.integer(interaction(
  grid[c("model_id", "distribution", "missingness")],
  drop = TRUE, lex.order = TRUE))

samplers <- list()
for (model_id in names(models)) {
  for (distribution in opts$distributions) {
    key <- paste(model_id, distribution, sep = "::")
    samplers[[key]] <- sem_calibrate_sampler(
      models[[model_id]], distribution)
  }
}

fmg_p <- function(statistic, df, eigenvalues) {
  magmaan:::infer_fmg_test(
    statistic, df, eigenvalues,
    method = "peba", param = 4,
    truncate_negative = TRUE)$p_value
}

one_rep <- function(cell, rep_id) {
  model <- models[[cell$model_id]]
  sampler <- samplers[[paste(cell$model_id, cell$distribution, sep = "::")]]
  seed <- sem_seed(opts$seed_base + cell$pair_id * 100003L + rep_id)
  begin <- proc.time()[["elapsed"]]
  out <- data.frame(
    cell_id = cell$cell_id,
    model_id = cell$model_id,
    distribution = cell$distribution,
    missingness = cell$missingness,
    n = opts$n,
    rep = rep_id,
    seed = seed,
    ok = FALSE,
    error = "",
    df = NA_integer_,
    statistic_lrt = NA_real_,
    statistic_score = NA_real_,
    eigen_mean_lrt = NA_real_,
    eigen_cv_lrt = NA_real_,
    eigen_mean_score = NA_real_,
    eigen_cv_score = NA_real_,
    p_lrt_lrt = NA_real_,
    p_lrt_score = NA_real_,
    p_score_lrt = NA_real_,
    p_score_score = NA_real_,
    seconds = NA_real_,
    stringsAsFactors = FALSE)

  ans <- tryCatch({
    X <- sem_draw(model, sampler, opts$n, seed)
    set.seed(seed + 700001L)
    X <- sem_apply_missingness(X, cell$missingness)
    fd <- magmaan::df_to_fiml_data(as.data.frame(X), model$spec)
    control <- list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)
    em <- magmaan::magmaan_core$estimate_saturated_em_moments(
      fd, control = control)
    fit <- magmaan::magmaan_core$fit_fiml(
      model$spec, fd,
      optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control)
    fit$stage1 <- em
    if (!isTRUE(fit$converged)) stop("FIML fit did not converge")

    lrt <- magmaan::fmg_tests(fit, tests = "all")
    score <- magmaan::global_score_flip_test(
      fit,
      n_flips = 1L,
      seed = seed + 900001L,
      multiplier = "rademacher",
      sensitivity = opts$sensitivity)
    df <- as.integer(lrt$df[[1L]])
    if (df != as.integer(score$df)) stop("LRT and score df differ")
    statistic_lrt <- lrt$base_statistic[[1L]]
    statistic_score <- score$statistic_effective
    eigen_lrt <- lrt$eigenvalues[[1L]]
    eigen_score <- score$eigenvalues
    list(
      df = df,
      statistic_lrt = statistic_lrt,
      statistic_score = statistic_score,
      eigen_mean_lrt = mean(eigen_lrt),
      eigen_cv_lrt = stats::sd(eigen_lrt) / mean(eigen_lrt),
      eigen_mean_score = mean(eigen_score),
      eigen_cv_score = stats::sd(eigen_score) / mean(eigen_score),
      p_lrt_lrt = fmg_p(statistic_lrt, df, eigen_lrt),
      p_lrt_score = fmg_p(statistic_lrt, df, eigen_score),
      p_score_lrt = fmg_p(statistic_score, df, eigen_lrt),
      p_score_score = fmg_p(statistic_score, df, eigen_score))
  }, error = function(e) e)

  if (inherits(ans, "error")) {
    out$error <- conditionMessage(ans)
  } else {
    for (name in names(ans)) out[[name]] <- ans[[name]]
    out$ok <- all(is.finite(unlist(ans)))
    if (!out$ok) out$error <- "non-finite statistic, spectrum, or p-value"
  }
  out$seconds <- proc.time()[["elapsed"]] - begin
  out
}

one_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf(
    "cell %d/%d: %s / %s / %s",
    index, nrow(grid), cell$model_id, cell$distribution, cell$missingness))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

cat(sprintf(
  "cells=%d reps=%d n=%d sensitivity=%s cores=%d\n",
  nrow(grid), opts$reps, opts$n, opts$sensitivity, opts$cores))
begin <- proc.time()[["elapsed"]]
pieces <- if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  parallel::mclapply(
    seq_len(nrow(grid)), one_cell,
    mc.cores = min(opts$cores, nrow(grid)), mc.preschedule = FALSE)
} else lapply(seq_len(nrow(grid)), one_cell)
raw <- do.call(rbind, pieces)
raw <- raw[order(raw$cell_id, raw$rep), ]
row.names(raw) <- NULL
write_csv(raw, file.path(results, "replications.csv"))

summarize_cell <- function(z) {
  attempted <- nrow(z)
  z <- z[z$ok, , drop = FALSE]
  if (!nrow(z)) return(data.frame())
  data.frame(
    z[1L, c("model_id", "distribution", "missingness", "n"), drop = FALSE],
    attempted = attempted,
    usable = nrow(z),
    rejection_lrt_lrt = mean(z$p_lrt_lrt <= .05),
    rejection_lrt_score = mean(z$p_lrt_score <= .05),
    rejection_score_lrt = mean(z$p_score_lrt <= .05),
    rejection_score_score = mean(z$p_score_score <= .05),
    mean_abs_p_spectrum_effect_lrt = mean(abs(z$p_lrt_score - z$p_lrt_lrt)),
    mean_abs_p_spectrum_effect_score = mean(abs(z$p_score_score - z$p_score_lrt)),
    mean_abs_p_statistic_effect_lrt_spectrum =
      mean(abs(z$p_score_lrt - z$p_lrt_lrt)),
    mean_abs_p_statistic_effect_score_spectrum =
      mean(abs(z$p_score_score - z$p_lrt_score)),
    mean_eigen_lrt = mean(z$eigen_mean_lrt),
    mean_eigen_score = mean(z$eigen_mean_score),
    mean_statistic_lrt = mean(z$statistic_lrt),
    mean_statistic_score = mean(z$statistic_score),
    stringsAsFactors = FALSE)
}
groups <- split(seq_len(nrow(raw)), interaction(
  raw[c("model_id", "distribution", "missingness")],
  drop = TRUE, lex.order = TRUE))
summary <- do.call(rbind, lapply(groups, function(ii) summarize_cell(raw[ii, ])))
row.names(summary) <- NULL
write_csv(summary, file.path(results, "summary.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  reps = opts$reps,
  n = opts$n,
  models = names(models),
  distributions = opts$distributions,
  missingness = opts$missingness,
  score_sensitivity = opts$sensitivity,
  seed_base = opts$seed_base,
  failures = sum(!raw$ok),
  runtime_wall_seconds = proc.time()[["elapsed"]] - begin),
  packages = "magmaan")

cat(sprintf(
  "runtime_wall=%.1fs failures=%d/%d\n\n",
  proc.time()[["elapsed"]] - begin, sum(!raw$ok), nrow(raw)))
print(summary, row.names = FALSE, digits = 3)
