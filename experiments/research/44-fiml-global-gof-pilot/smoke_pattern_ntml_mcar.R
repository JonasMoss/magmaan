#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaan)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "R", "sem_models.R"))

reps <- as.integer(Sys.getenv("PNTML_REPS", "20"))
n <- as.integer(Sys.getenv("PNTML_N", "250"))
seed_base <- as.integer(Sys.getenv("PNTML_SEED", "20260905"))
results_dir <- Sys.getenv(
  "PNTML_RESULTS_DIR",
  file.path(script_dir, "results", "pattern-ntml-mcar-smoke"))
stopifnot(reps > 0L, n >= 80L, seed_base >= 0L)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

model <- sem_model_one_factor()
sampler <- sem_calibrate_sampler(model, "normal")
control <- list(max_iter = 1500L, ftol = 1e-10, gtol = 1e-7)

one_rep <- function(mechanism, rep) {
  seed <- seed_base + rep + if (mechanism == "mcar_30") 100000L else 0L
  x <- sem_draw(model, sampler, n, seed)
  x <- sem_apply_missingness(x, mechanism, rate = 0.30)
  realized_missing <- mean(is.na(x[, -1L, drop = FALSE]))
  raw <- df_to_fiml_data(as.data.frame(x), model$spec)

  begin <- proc.time()[["elapsed"]]
  result <- tryCatch({
    stage1 <- magmaan_core$estimate_saturated_em_moments(raw)
    fiml <- magmaan_core$fit_fiml(
      model$spec, raw, optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control)
    ml2s <- magmaan_core$fit_ml2s(
      model$spec, raw, optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control, stage1 = stage1)
    pntml <- frontier_fit_pattern_ntml(
      model$spec, raw, optimizer = "nlopt-lbfgs-slsqp-fallback",
      control = control, stage1 = stage1)
    if (!all(vapply(list(fiml, ml2s, pntml), function(z) isTRUE(z$converged),
                    logical(1L)))) {
      stop("at least one estimator did not converge")
    }
    fiml_info <- magmaan_core$inference_fiml_information_vcov(fiml)$expected
    fiml_lrt <- magmaan_core$estimate_fiml_robust_mlr(fiml)
    if (!isTRUE(fiml_info$ok)) stop(fiml_info$error)

    data.frame(
      mechanism = mechanism, rep = rep, seed = seed, n = n,
      realized_missing = realized_missing, ok = TRUE, error = "",
      theta_diff_pntml_fiml = max(abs(pntml$theta - fiml$theta)),
      theta_diff_ml2s_fiml = max(abs(ml2s$theta - fiml$theta)),
      theta_diff_pntml_ml2s = max(abs(pntml$theta - ml2s$theta)),
      median_se_ratio_pntml_fiml =
        median(pntml$se / fiml_info$se_model),
      chisq_pntml = pntml$chisq,
      chisq_fiml = fiml_lrt$chisq,
      chisq_diff_pntml_fiml = abs(pntml$chisq - fiml_lrt$chisq),
      reject_pntml_05 = stats::pchisq(
        pntml$chisq, pntml$df, lower.tail = FALSE) < 0.05,
      reject_fiml_05 = stats::pchisq(
        fiml_lrt$chisq, fiml_lrt$df, lower.tail = FALSE) < 0.05,
      pntml_scale = pntml$scaling_factor,
      max_pntml_eigen_deviation = max(abs(pntml$pntml$eigvals - 1)),
      seconds = proc.time()[["elapsed"]] - begin,
      stringsAsFactors = FALSE)
  }, error = function(e) {
    data.frame(
      mechanism = mechanism, rep = rep, seed = seed, n = n,
      realized_missing = realized_missing, ok = FALSE,
      error = conditionMessage(e),
      theta_diff_pntml_fiml = NA_real_, theta_diff_ml2s_fiml = NA_real_,
      theta_diff_pntml_ml2s = NA_real_,
      median_se_ratio_pntml_fiml = NA_real_,
      chisq_pntml = NA_real_, chisq_fiml = NA_real_,
      chisq_diff_pntml_fiml = NA_real_, reject_pntml_05 = NA,
      reject_fiml_05 = NA, pntml_scale = NA_real_,
      max_pntml_eigen_deviation = NA_real_,
      seconds = proc.time()[["elapsed"]] - begin,
      stringsAsFactors = FALSE)
  })
  result
}

rows <- lapply(c("complete", "mcar_30"), function(mechanism) {
  do.call(rbind, lapply(seq_len(reps), function(rep) one_rep(mechanism, rep)))
})
replications <- do.call(rbind, rows)
row.names(replications) <- NULL

summaries <- lapply(split(replications, replications$mechanism), function(z) {
  usable <- z[z$ok, , drop = FALSE]
  if (!nrow(usable)) {
    return(data.frame(
      mechanism = z$mechanism[[1L]], reps = nrow(z), usable = 0L,
      realized_missing = NA_real_, median_theta_diff_pntml_fiml = NA_real_,
      max_theta_diff_pntml_fiml = NA_real_,
      median_theta_diff_ml2s_fiml = NA_real_,
      median_se_ratio_pntml_fiml = NA_real_,
      mean_abs_chisq_diff_pntml_fiml = NA_real_,
      reject_pntml_05 = NA_real_, reject_fiml_05 = NA_real_,
      max_pntml_eigen_deviation = NA_real_, mean_seconds = NA_real_,
      stringsAsFactors = FALSE))
  }
  data.frame(
    mechanism = z$mechanism[[1L]], reps = nrow(z), usable = nrow(usable),
    realized_missing = mean(usable$realized_missing),
    median_theta_diff_pntml_fiml =
      median(usable$theta_diff_pntml_fiml),
    max_theta_diff_pntml_fiml = max(usable$theta_diff_pntml_fiml),
    median_theta_diff_ml2s_fiml = median(usable$theta_diff_ml2s_fiml),
    median_se_ratio_pntml_fiml =
      median(usable$median_se_ratio_pntml_fiml),
    mean_abs_chisq_diff_pntml_fiml =
      mean(usable$chisq_diff_pntml_fiml),
    reject_pntml_05 = mean(usable$reject_pntml_05),
    reject_fiml_05 = mean(usable$reject_fiml_05),
    max_pntml_eigen_deviation =
      max(usable$max_pntml_eigen_deviation),
    mean_seconds = mean(usable$seconds), stringsAsFactors = FALSE)
})
summary <- do.call(rbind, summaries)
row.names(summary) <- NULL

utils::write.csv(replications, file.path(results_dir, "replications.csv"),
                 row.names = FALSE)
utils::write.csv(summary, file.path(results_dir, "summary.csv"),
                 row.names = FALSE)
utils::write.csv(data.frame(
  model = model$model_id, distribution = "normal",
  mechanisms = "complete,mcar_30", reps = reps, n = n,
  seed_base = seed_base, generated_at = format(Sys.time(), tz = "UTC"),
  stringsAsFactors = FALSE), file.path(results_dir, "metadata.csv"),
  row.names = FALSE)

print(summary, row.names = FALSE)
if (!all(replications$ok)) quit(save = "no", status = 1L)
