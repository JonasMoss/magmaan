# Public simulation helpers.
#
# These are thin exported aliases over the magmaan::sim C++ surface. They keep
# the two-stage calibration/draw contract visible from ordinary package use
# without requiring callers to reach through magmaan_core.

#' Calibrate model-implied continuous data generation.
#'
#' @param fit_or_partable A magmaan fit object or partable.
#' @param theta Optional free-parameter vector when `fit_or_partable` is a
#'   partable.
#' @export
sim_model_calibrate <- sim_model_calibrate_impl

#' Draw model-implied continuous data from a calibration object.
#'
#' @inheritParams sim_model_calibrate
#' @param calibration A calibration from [sim_model_calibrate()].
#' @param n Number of observations per replication.
#' @param reps Number of replications.
#' @param seed_base Base RNG seed.
#' @param generator Continuous generator name; default `"normal"`.
#' @param df Degrees of freedom for Student t generation.
#' @param mixture_weights Mixture component probabilities.
#' @param mixture_scale_multipliers Positive component scale multipliers.
#' @param contamination_probability Probability of a contaminated observation.
#' @param contamination_scale_multiplier Scale multiplier for contamination.
#' @param slash_q Positive exponent for slash generation.
#' @param cholesky_jitter Non-negative diagonal jitter used for factorization.
#' @export
sim_model_draw <- sim_model_draw_impl

#' Calibrate and draw model-implied continuous data in one call.
#'
#' @inheritParams sim_model_draw
#' @inheritParams sim_model_calibrate
#' @export
sim_model_batch <- sim_model_batch_impl

#' Calibrate ordinal/continuous Pearson-code correlation generation.
#'
#' @param target_corr Target observed correlation matrix.
#' @param marginals List of marginal category probabilities, or `NULL` entries
#'   for continuous variables.
#' @param metric Correlation metric used for the target; default `"polychoric"`.
#' @param max_bisection_iter Maximum iterations for each correlation inversion.
#' @param calibration_tol Absolute tolerance for correlation calibration.
#' @param rho_bound Upper bound on the absolute latent correlation.
#' @param matrix_repair Correlation-matrix repair strategy.
#' @param matrix_repair_min_eigenvalue Minimum eigenvalue requested by repair.
#' @export
sim_ordcorr_calibrate <- sim_ordcorr_calibrate_impl

#' Rebuild an ordinal-correlation calibration from latent summary pieces.
#'
#' @inheritParams sim_ordcorr_calibrate
#' @param latent_corr Latent Gaussian correlation matrix.
#' @param kinds Integer variable-type codes from an existing calibration.
#' @param thresholds Threshold list from an existing calibration.
#' @export
sim_ordcorr_summary_calibrate <- sim_ordcorr_summary_calibrate_impl

#' Draw ordinal/continuous Pearson-code data from a calibration object.
#'
#' @param calibration A calibration from [sim_ordcorr_calibrate()].
#' @param cholesky_jitter Non-negative diagonal jitter used for factorization.
#' @param n Number of observations per replication.
#' @param reps Number of replications.
#' @param seed_base Base RNG seed.
#' @export
sim_ordcorr_draw <- sim_ordcorr_draw_impl

#' Calibrate and draw ordinal/continuous Pearson-code data in one call.
#'
#' @inheritParams sim_ordcorr_draw
#' @inheritParams sim_ordcorr_calibrate
#' @export
sim_ordcorr_batch <- sim_ordcorr_batch_impl

#' Calibrate grouped ordinal/continuous Pearson-code generation.
#'
#' @param target_corrs List of target correlation matrices, one per group.
#' @param marginals List of group marginal lists.
#' @param group_labels Optional group labels.
#' @inheritParams sim_ordcorr_calibrate
#' @export
sim_ordcorr_mg_calibrate <- sim_ordcorr_mg_calibrate_impl

#' Rebuild grouped ordinal-correlation calibrations from summary pieces.
#'
#' @param latent_corrs List of latent Gaussian correlation matrices.
#' @param kinds Integer variable-type codes from an existing calibration.
#' @param thresholds List of threshold lists, one per group.
#' @inheritParams sim_ordcorr_mg_calibrate
#' @export
sim_ordcorr_mg_summary_calibrate <- sim_ordcorr_mg_summary_calibrate_impl

#' Draw grouped ordinal/continuous data from a calibration object.
#'
#' @inheritParams sim_ordcorr_draw
#' @export
sim_ordcorr_mg_draw <- sim_ordcorr_mg_draw_impl

#' Calibrate and draw grouped ordinal/continuous data in one call.
#'
#' @inheritParams sim_ordcorr_mg_draw
#' @inheritParams sim_ordcorr_mg_calibrate
#' @export
sim_ordcorr_mg_batch <- sim_ordcorr_mg_batch_impl
