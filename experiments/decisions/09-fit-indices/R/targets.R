independence_spec <- function(population, means = population$record$meanstructure) {
  variables <- colnames(population$stats$S[[1]])
  syntax <- paste(paste(variables, "~~", variables), collapse = "\n")
  population_spec(population$record, syntax, means)
}
index_targets <- function(f0, f0_b, df, df_b, groups, srmr) {
  c(F0 = max(f0, 0), F0_b = max(f0_b, 0), df = df, df_b = df_b,
    rmsea = sqrt(groups*max(f0,0)/df),
    cfi = if (f0_b > 0) max(0, min(1, 1-max(f0,0)/f0_b)) else NA_real_,
    tli = if (f0_b > 0) 1-(max(f0,0)/df)/(f0_b/df_b) else NA_real_, srmr = srmr)
}

target_control <- function() list(nlopt = list(ftol_rel = 1e-14, xtol_rel = 1e-12, max_eval = 10000L))

fit_target <- function(model, data, estimator = "ML", kind) {
  initial <- magmaanlab::fit_model(model, data, estimator = estimator)
  control <- target_control()
  control$start <- initial$partable
  fit <- magmaanlab::fit_model(model, data, estimator = estimator, control = control)
  diagnostic <- function(x, stage) data.frame(kind = kind, stage = stage,
    converged = isTRUE(x$converged), optimizer_status = x$optimizer_status,
    stationarity = x$diagnostics$verdict$stationarity,
    newton_distance = x$diagnostics$newton_accuracy$distance,
    budget = x$diagnostics$newton_accuracy$budget, iterations = x$iterations)
  list(fit = fit, diagnostics = rbind(diagnostic(initial, "default"), diagnostic(fit, "refined")))
}

compute_targets <- function(population, calibration, cell, seed, target_n = 200000L) {
  ss <- population$stats
  ss$S <- lapply(calibration$groups, `[[`, "sigma")
  ss$mean <- lapply(calibration$groups, `[[`, "mu")
  started <- proc.time()[[3]]
  h1 <- fit_target(population$model, ss, kind = "complete_h1")
  fit <- h1$fit
  baseline_model <- independence_spec(population)
  b <- fit_target(baseline_model, ss, kind = "complete_baseline")
  baseline <- b$fit
  diagnostics <- rbind(h1$diagnostics, b$diagnostics)
  complete_ok <- isTRUE(fit$converged) && isTRUE(baseline$converged)
  df <- with(magmaanlab::structural_identification(population$model), n_moments-n_parameters)
  df_b <- with(magmaanlab::structural_identification(baseline_model), n_moments-n_parameters)
  fm <- magmaanlab::fit_measures(fit)
  complete <- index_targets(fit$fmin, baseline$fmin, df, df_b, length(ss$S), fm$srmr)
  if (!complete_ok) complete[] <- NA_real_
  own <- complete; method <- "exact_population_moments"; n <- 0L
  if (cell$estimator == "FIML" && cell$missing_rate > 0) {
    data <- apply_mar(draw_generator(calibration, target_n, seed), cell, seed+1000L)
    # Ordinary fits always include means. The summary target also has unrestricted
    # means for covariance-only basis models; structured means remain in h1 syntax.
    model <- population_spec(population$record, means = TRUE)
    baseline_model <- independence_spec(population, TRUE)
    h1 <- fit_target(model, data, "FIML", "mar_h1")
    b <- fit_target(baseline_model, data, "FIML", "mar_baseline")
    fit <- h1$fit; baseline <- b$fit
    diagnostics <- rbind(diagnostics, h1$diagnostics, b$diagnostics)
    fm <- magmaanlab::fit_measures(fit, robust = FALSE)
    df <- fm$df; df_b <- fm$baseline.df
    # FIML fmin is an objective, not the saturated LR discrepancy. Use the
    # library LR contrasts from user and independence fits, divided by N.
    baseline_fm <- magmaanlab::fit_measures(baseline, robust = FALSE)
    own <- index_targets(fm$chisq/target_n, baseline_fm$chisq/target_n,
                         df, df_b, length(ss$S), fm$srmr)
    method <- "fixed_large_draw_FIML_LR_against_saturated_H1"; n <- target_n
  }
  ok <- all(diagnostics$converged[diagnostics$stage == "refined"])
  if (!ok) own[] <- NA_real_
  list(status = if (ok) "available" else "failed",
       reason = if (ok) NA_character_ else "Refined target failed library convergence verdict",
       diagnostics = diagnostics, controls = target_control(),
       target_seconds = proc.time()[[3]]-started, own = own, complete = complete, method = method, n = n, seed = seed,
       mar = "logistic; standardized cause; slope=1; draw-calibrated intercept",
       cause = if (nzchar(cell$cause)) cell$cause else colnames(ss$S[[1]])[1],
       missing_targets = if (nzchar(cell$missing_targets)) cell$missing_targets else
         paste(tail(colnames(ss$S[[1]])[-1], floor(ncol(ss$S[[1]])/2)), collapse = ";"))
}
