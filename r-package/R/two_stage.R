# Two-stage missing-data SEM estimator (research comparator).
#
# Stage 1: saturated EM (μ̂_em, Σ̂_em) under MAR + normality, via
#          `magmaan_core$estimate_saturated_em_moments(raw_data)`.
# Stage 2: complete-data ML or GLS on the EM moments — `estimate_ml` /
#          `estimate_gls` with `sample_stats = list(S = cov, mean = mean,
#          nobs = n_obs)`.
#
# This is the point-estimate version of the Savalei-Bentler (2009)
# two-stage approach. The Stage-1 ACOV ingredients (H, J, ACOV) are
# returned alongside the fit so a future second-stage SE correction
# can consume them; this wrapper itself does not yet build corrected
# standard errors.
#
# Exposed via `magmaan_core$estimate_two_stage_em` in `zzz_core.R`.
# Not a stable default estimator — currently used as a research
# comparator in experiments / the pairwise-robust-sem paper.

estimate_two_stage_em_impl <- function(partable, raw_data,
                                       kind = c("ml", "gls"),
                                       h_step = 1e-4,
                                       optimizer = NULL,
                                       control = NULL,
                                       bounds = NULL) {
  kind <- match.arg(kind)

  em <- saturated_em_moments_impl(raw_data, h_step = h_step)

  n_blocks <- length(em$cov)
  if (n_blocks == 0L)
    stop("estimate_two_stage_em: saturated_em_moments returned no blocks",
         call. = FALSE)

  cov_list  <- em$cov
  mean_list <- em$mean
  nobs <- as.integer(em$n_obs)

  sample_stats <- list(S = cov_list, mean = mean_list, nobs = nobs)

  fit <- switch(kind,
    ml  = fit_ml_impl(partable,  sample_stats, optimizer = optimizer,
                      control = control, bounds = bounds),
    gls = fit_gls_impl(partable, sample_stats, optimizer = optimizer,
                       control = control, bounds = bounds)
  )

  fit$estimator <- paste0("two_stage_", kind)
  fit$stage1 <- list(
    mean = em$mean, cov = em$cov, n_obs = em$n_obs,
    H = em$H, J = em$J, acov = em$acov)
  fit
}
