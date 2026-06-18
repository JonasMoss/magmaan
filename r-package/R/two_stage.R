# Two-stage missing-data SEM estimator (research comparator).
#
# Stage 1: saturated EM (μ̂_em, Σ̂_em) under MAR + normality, via
#          `magmaan_core$estimate_saturated_em_moments(raw_data)`.
# Stage 2: complete-data ML or GLS on the EM moments — `estimate_ml` /
#          `estimate_gls` with `sample_stats = list(S = cov, mean = mean,
#          nobs = n_obs)`.
#
# For `kind = "ml"` this is the packaged ML2S surface: Stage-1 ACOV
# ingredients feed the Savalei-Bentler corrected SEs and scaled chi-square.
# `kind = "gls"` remains a point-estimate research comparator.
#
# Exposed via `magmaan_core$estimate_two_stage_em` in `zzz_core.R`.
# Not a stable default estimator — currently used as a research
# comparator in experiments / the pairwise-robust-sem paper.

estimate_two_stage_em_impl <- function(partable, raw_data,
                                       kind = c("ml", "gls"),
                                       h_step = 1e-4,
                                       optimizer = NULL,
                                       control = NULL,
                                       bounds = NULL,
                                       stage1 = NULL,
                                       stage2_weight = "nt",
                                       dls_a = 0.5) {
  kind <- match.arg(kind)
  stage2_weight <- match.arg(stage2_weight,
                             c("nt", "dwls", "adf", "dls", "wls"))

  # Stage 1 (saturated EM moments) depends only on the data, so a caller fitting
  # several models to one dataset can build it once and pass it in via `stage1`
  # (the object returned by `estimate_saturated_em_moments()` / a prior fit's
  # `$stage1`), skipping the redundant rebuild.
  em <- if (is.null(stage1)) saturated_em_moments_impl(raw_data, h_step = h_step)
        else stage1

  n_blocks <- length(em$cov)
  if (n_blocks == 0L)
    stop("estimate_two_stage_em: saturated_em_moments returned no blocks",
         call. = FALSE)

  cov_list  <- em$cov
  mean_list <- em$mean
  nobs <- as.integer(em$n_obs)

  sample_stats <- list(S = cov_list, mean = mean_list, nobs = nobs)
  b <- bounds_arg(bounds, partable, sample_stats, "estimate_two_stage_em")

  # A non-NT Stage-2 weight defines a *weighted* Stage-2 estimator (DWLS / ADF /
  # DLS), not ML. Fit it by the matching moment quadratic; the robust correction
  # below uses the same weight. `kind = "ml", stage2_weight = "nt"` is the
  # lavaan robust.two.stage path.
  weighted_stage2 <- identical(kind, "ml") && !identical(stage2_weight, "nt")

  fit <- if (weighted_stage2) {
    W <- two_stage_stage2_weight_blocks_impl(em, stage2_weight = stage2_weight,
                                             dls_a = dls_a)
    fit_wls_impl(partable, sample_stats, W = W, optimizer = optimizer,
                 control = control, bounds = b)
  } else {
    switch(kind,
      ml  = fit_ml_impl(partable,  sample_stats, optimizer = optimizer,
                        control = control, bounds = b),
      gls = fit_gls_impl(partable, sample_stats, optimizer = optimizer,
                         control = control, bounds = b)
    )
  }

  fit$estimator <- if (identical(kind, "gls")) "two_stage_gls"
                   else if (identical(stage2_weight, "nt")) "ML2S"
                   else paste0("ML2S_", toupper(stage2_weight))
  fit$stage1 <- list(
    mean = em$mean, cov = em$cov, n_obs = em$n_obs,
    H = em$H, J = em$J, acov = em$acov)
  fit$raw_data <- raw_data
  if (identical(kind, "ml")) {
    correction <- estimate_two_stage_em_ml_inference(
      fit, raw_data, h_step = h_step,
      stage2_weight = stage2_weight, dls_a = dls_a)
    fit$ml2s <- correction
    fit$vcov <- correction$vcov
    fit$se <- correction$se
    fit$chisq <- correction$chisq
    fit$df <- correction$df
    fit$chisq_scaled <- correction$chisq_scaled
    fit$scaling_factor <- correction$scaling_factor
  }
  fit
}
