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
  "Usage: Rscript smoke_ml2s_rls.R [options]\n\n",
  "Crosses the ML2S Stage-2 ML and fitted-model RLS base statistics with the\n",
  "same two-stage UGamma spectrum. Both statistics use the same saturated-EM\n",
  "moments and the same fitted SEM; only the discrepancy value changes.\n\n",
  "  --reps N             Replications per cell (default 200).\n",
  "  --n CSV              Sample sizes (default 120,500).\n",
  "  --cores N            Parallel cell workers (default up to 4).\n",
  "  --model ID           SEM model id (default one_factor_6).\n",
  "  --distributions CSV  Default normal,vm2,ig2.\n",
  "  --missingness CSV    Default complete,mcar_30,mar_30.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  reps = 200L,
  n = c(120L, 500L),
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  model = "one_factor_6",
  distributions = c("normal", "vm2", "ig2"),
  missingness = c("complete", "mcar_30", "mar_30"),
  seed_base = 20260905L,
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
  if (arg %in% c("-h", "--help")) {
    usage()
    quit(save = "no", status = 0L)
  } else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(parse_csv_arg(take()))
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--model") opts$model <- take()
  else if (arg == "--distributions") {
    opts$distributions <- parse_csv_arg(take())
  } else if (arg == "--missingness") {
    opts$missingness <- parse_csv_arg(take())
  } else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}

stopifnot(
  opts$reps > 0L,
  all(opts$n >= 80L),
  opts$cores > 0L,
  all(opts$distributions %in% c("normal", "vm1", "ig1", "vm2", "ig2")),
  all(opts$missingness %in% c("complete", "mcar_30", "mar_30")))

models <- sem_model_catalog()
if (!opts$model %in% names(models)) {
  stop("unknown model: ", opts$model, call. = FALSE)
}
model <- models[[opts$model]]
results <- opts$results_dir %||% file.path(
  script_dir, "results", "ml2s-rls-smoke")
dir.create(results, recursive = TRUE, showWarnings = FALSE)

grid <- expand.grid(
  n = opts$n,
  distribution = opts$distributions,
  missingness = opts$missingness,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE)
grid$cell_id <- seq_len(nrow(grid))
grid$null_contract <- ifelse(
  grid$missingness != "mar_30" | grid$distribution == "normal",
  "pseudo-null", "nonnormal-MAR estimand stress")

samplers <- setNames(lapply(opts$distributions, function(distribution) {
  sem_calibrate_sampler(model, distribution)
}), opts$distributions)

fmg_p <- function(statistic, df, eigenvalues, method, param = 4) {
  magmaan:::infer_fmg_test(
    statistic, df, eigenvalues,
    method = method, param = param,
    truncate_negative = TRUE)$p_value
}

one_rep <- function(cell, rep_id) {
  begin <- proc.time()[["elapsed"]]
  seed <- sem_seed(opts$seed_base + cell$cell_id * 100003L + rep_id)
  out <- data.frame(
    cell_id = cell$cell_id,
    model_id = model$model_id,
    distribution = cell$distribution,
    missingness = cell$missingness,
    null_contract = cell$null_contract,
    n = cell$n,
    rep = rep_id,
    seed = seed,
    ok = FALSE,
    error = "",
    df = NA_integer_,
    statistic_ml = NA_real_,
    statistic_rls = NA_real_,
    statistic_rls_mean = NA_real_,
    statistic_rls_covariance = NA_real_,
    difference_rls_minus_ml = NA_real_,
    ratio_rls_to_ml = NA_real_,
    p_standard_ml = NA_real_,
    p_standard_rls = NA_real_,
    p_sb_ml = NA_real_,
    p_sb_rls = NA_real_,
    p_ss_ml = NA_real_,
    p_ss_rls = NA_real_,
    p_peba4_ml = NA_real_,
    p_peba4_rls = NA_real_,
    p_all_ml = NA_real_,
    p_all_rls = NA_real_,
    eigen_mean = NA_real_,
    eigen_cv = NA_real_,
    realized_missing = NA_real_,
    seconds = NA_real_,
    stringsAsFactors = FALSE)

  ans <- tryCatch({
    X <- sem_draw(model, samplers[[cell$distribution]], cell$n, seed)
    set.seed(seed + 700001L)
    X <- sem_apply_missingness(X, cell$missingness)
    realized_missing <- mean(is.na(X))
    fd <- magmaan::df_to_fiml_data(as.data.frame(X), model$spec)
    control <- list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)
    stage1 <- magmaan::magmaan_core$estimate_saturated_em_moments(
      fd, control = control)
    fit <- magmaan::magmaan_core$fit_ml2s(
      model$spec, fd,
      optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control,
      stage1 = stage1)
    if (!isTRUE(fit$converged)) stop("ML2S fit did not converge")
    if (is.null(fit$ml2s$eigvals)) stop("ML2S spectrum is unavailable")

    statistic_ml <- as.numeric(fit$ml2s$chisq)
    rls <- magmaan:::infer_nt_moment_quadratic_fit(
      fit, magmaan:::model_implied(fit))
    statistic_rls <- rls$statistic
    df <- as.integer(fit$ml2s$df)
    eigenvalues <- as.numeric(fit$ml2s$eigvals)
    if (length(eigenvalues) != df) stop("ML2S spectrum and df disagree")

    list(
      df = df,
      statistic_ml = statistic_ml,
      statistic_rls = statistic_rls,
      statistic_rls_mean = rls$mean,
      statistic_rls_covariance = rls$covariance,
      difference_rls_minus_ml = statistic_rls - statistic_ml,
      ratio_rls_to_ml = statistic_rls / statistic_ml,
      p_standard_ml = fmg_p(statistic_ml, df, eigenvalues, "standard"),
      p_standard_rls = fmg_p(statistic_rls, df, eigenvalues, "standard"),
      p_sb_ml = fmg_p(statistic_ml, df, eigenvalues, "sb"),
      p_sb_rls = fmg_p(statistic_rls, df, eigenvalues, "sb"),
      p_ss_ml = fmg_p(statistic_ml, df, eigenvalues, "ss"),
      p_ss_rls = fmg_p(statistic_rls, df, eigenvalues, "ss"),
      p_peba4_ml = fmg_p(statistic_ml, df, eigenvalues, "peba"),
      p_peba4_rls = fmg_p(statistic_rls, df, eigenvalues, "peba"),
      p_all_ml = fmg_p(statistic_ml, df, eigenvalues, "all"),
      p_all_rls = fmg_p(statistic_rls, df, eigenvalues, "all"),
      eigen_mean = mean(eigenvalues),
      eigen_cv = stats::sd(eigenvalues) / mean(eigenvalues),
      realized_missing = realized_missing)
  }, error = function(e) e)

  if (inherits(ans, "error")) {
    out$error <- conditionMessage(ans)
  } else {
    for (name in names(ans)) out[[name]] <- ans[[name]]
    out$ok <- all(is.finite(unlist(ans)))
    if (!out$ok) out$error <- "non-finite statistic or diagnostic"
  }
  out$seconds <- proc.time()[["elapsed"]] - begin
  out
}

one_cell <- function(index) {
  cell <- as.list(grid[index, , drop = FALSE])
  message(sprintf(
    "cell %d/%d: n=%d / %s / %s",
    index, nrow(grid), cell$n, cell$distribution, cell$missingness))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

cat(sprintf(
  "cells=%d reps=%d model=%s cores=%d\n",
  nrow(grid), opts$reps, model$model_id, opts$cores))
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
    z[1L, c("model_id", "distribution", "missingness", "null_contract", "n"),
      drop = FALSE],
    attempted = attempted,
    usable = nrow(z),
    mean_ml = mean(z$statistic_ml),
    mean_rls = mean(z$statistic_rls),
    mean_rls_mean_component = mean(z$statistic_rls_mean),
    mean_rls_covariance_component = mean(z$statistic_rls_covariance),
    mean_difference = mean(z$difference_rls_minus_ml),
    mean_abs_difference = mean(abs(z$difference_rls_minus_ml)),
    median_ratio = stats::median(z$ratio_rls_to_ml),
    ratio_q10 = unname(stats::quantile(z$ratio_rls_to_ml, 0.10)),
    ratio_q90 = unname(stats::quantile(z$ratio_rls_to_ml, 0.90)),
    correlation = if (nrow(z) > 1L) {
      stats::cor(z$statistic_ml, z$statistic_rls)
    } else NA_real_,
    rejection_standard_ml = mean(z$p_standard_ml <= 0.05),
    rejection_standard_rls = mean(z$p_standard_rls <= 0.05),
    rejection_sb_ml = mean(z$p_sb_ml <= 0.05),
    rejection_sb_rls = mean(z$p_sb_rls <= 0.05),
    rejection_ss_ml = mean(z$p_ss_ml <= 0.05),
    rejection_ss_rls = mean(z$p_ss_rls <= 0.05),
    rejection_peba4_ml = mean(z$p_peba4_ml <= 0.05),
    rejection_peba4_rls = mean(z$p_peba4_rls <= 0.05),
    rejection_all_ml = mean(z$p_all_ml <= 0.05),
    rejection_all_rls = mean(z$p_all_rls <= 0.05),
    mean_eigen = mean(z$eigen_mean),
    mean_eigen_cv = mean(z$eigen_cv),
    mean_realized_missing = mean(z$realized_missing),
    stringsAsFactors = FALSE)
}

groups <- split(seq_len(nrow(raw)), interaction(
  raw[c("n", "distribution", "missingness")],
  drop = TRUE, lex.order = TRUE))
summary <- do.call(rbind, lapply(groups, function(ii) summarize_cell(raw[ii, ])))
row.names(summary) <- NULL
summary <- summary[order(summary$n, summary$distribution, summary$missingness), ]
write_csv(summary, file.path(results, "summary.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  reps = opts$reps,
  sample_sizes = paste(opts$n, collapse = ","),
  model = model$model_id,
  distributions = paste(opts$distributions, collapse = ","),
  missingness = paste(opts$missingness, collapse = ","),
  seed_base = opts$seed_base,
  failures = sum(!raw$ok),
  runtime_wall_seconds = proc.time()[["elapsed"]] - begin,
  base_statistics = "ML2S Stage-2 ML discrepancy; fitted-model mean+covariance RLS",
  reference_spectrum = "common ML2S UGamma spectrum"),
  packages = "magmaan")

cat(sprintf(
  "runtime_wall=%.1fs failures=%d/%d\n\n",
  proc.time()[["elapsed"]] - begin, sum(!raw$ok), nrow(raw)))
print(summary, row.names = FALSE, digits = 3)
