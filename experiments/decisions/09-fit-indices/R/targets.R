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

compute_targets <- function(population, calibration, cell, seed, target_n = 200000L) {
  ss <- population$stats
  ss$S <- lapply(calibration$groups, `[[`, "sigma")
  ss$mean <- lapply(calibration$groups, `[[`, "mu")
  fit <- magmaanlab::fit_model(population$model, ss)
  baseline_model <- independence_spec(population)
  baseline <- magmaanlab::fit_model(baseline_model, ss)
  if (!isTRUE(fit$converged) || !isTRUE(baseline$converged)) stop("Complete population target fit failed")
  df <- with(magmaanlab::structural_identification(population$model), n_moments-n_parameters)
  df_b <- with(magmaanlab::structural_identification(baseline_model), n_moments-n_parameters)
  fm <- magmaanlab::fit_measures(fit)
  complete <- index_targets(fit$fmin, baseline$fmin, df, df_b, length(ss$S), fm$srmr)
  own <- complete; method <- "exact_population_moments"; n <- 0L
  if (cell$estimator == "FIML" && cell$missing_rate > 0) {
    data <- apply_mar(draw_generator(calibration, target_n, seed), cell, seed+1000L)
    # Ordinary fits always include means. The summary target also has unrestricted
    # means for covariance-only basis models; structured means remain in h1 syntax.
    model <- population_spec(population$record, means = TRUE)
    baseline_model <- independence_spec(population, TRUE)
    fit <- magmaanlab::fit_model(model, data, estimator = "FIML")
    baseline <- magmaanlab::fit_model(baseline_model, data, estimator = "FIML")
    if (!isTRUE(fit$converged) || !isTRUE(baseline$converged)) stop("MAR population target fit failed")
    fm <- magmaanlab::fit_measures(fit, robust = FALSE)
    df <- fm$df; df_b <- fm$baseline.df
    # FIML fmin is an objective, not the saturated LR discrepancy. Use the
    # library LR contrasts from user and independence fits, divided by N.
    baseline_fm <- magmaanlab::fit_measures(baseline, robust = FALSE)
    own <- index_targets(fm$chisq/target_n, baseline_fm$chisq/target_n,
                         df, df_b, length(ss$S), fm$srmr)
    method <- "fixed_large_draw_FIML_LR_against_saturated_H1"; n <- target_n
  }
  list(own = own, complete = complete, method = method, n = n, seed = seed,
       mar = "logistic; standardized cause; slope=1; draw-calibrated intercept",
       cause = if (nzchar(cell$cause)) cell$cause else colnames(ss$S[[1]])[1],
       missing_targets = if (nzchar(cell$missing_targets)) cell$missing_targets else
         paste(tail(colnames(ss$S[[1]])[-1], floor(ncol(ss$S[[1]])/2)), collapse = ";"))
}
