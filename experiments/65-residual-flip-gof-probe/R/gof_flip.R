# Experimental covariance-only GOF flip. The tested estimating functions are
# model-centred saturated covariance contributions projected through the
# expected-information residual basis B. Their observed quadratic is exactly
# magmaan's structured-weight RLS GOF statistic.

.flip_projected_rows <- function(projected, n_flips, seed) {
  stopifnot(is.matrix(projected), nrow(projected) > 1L, ncol(projected) > 0L,
            n_flips > 0L, is.finite(seed))
  total_begin <- proc.time()[["elapsed"]]
  n <- nrow(projected)
  df <- ncol(projected)
  u_observed <- colSums(projected)
  statistic_effective <- sum(u_observed^2) / n

  score_begin <- proc.time()[["elapsed"]]
  set.seed(as.integer(seed))
  signs <- matrix(sample(c(-1, 1), n_flips * n, replace = TRUE),
                  nrow = n_flips, ncol = n)
  flipped <- signs %*% projected
  statistic_flipped <- rowSums(flipped^2) / n
  p_effective <- (1 + sum(statistic_flipped >= statistic_effective)) /
    (n_flips + 1)
  score_seconds <- proc.time()[["elapsed"]] - score_begin

  standardization_begin <- proc.time()[["elapsed"]]
  empirical <- crossprod(projected)
  eig <- eigen(empirical, symmetric = TRUE)
  tol <- 1e-10 * max(1, eig$values[[1L]])
  standardization_available <- all(is.finite(eig$values)) &&
    min(eig$values) > tol
  condition <- if (min(eig$values) > 0) {
    max(eig$values) / min(eig$values)
  } else Inf
  p_standardized <- statistic_standardized <- NA_real_
  statistic_standardized_flipped <- rep(NA_real_, n_flips)
  transform <- NULL
  if (standardization_available) {
    transform <- sweep(eig$vectors, 2L, sqrt(eig$values), "/")
    statistic_standardized <- sum((u_observed %*% transform)^2)
  }
  standardization_setup_seconds <-
    proc.time()[["elapsed"]] - standardization_begin

  standardization_resample_begin <- proc.time()[["elapsed"]]
  if (standardization_available) {
    statistic_standardized_flipped <- rowSums((flipped %*% transform)^2)
    p_standardized <-
      (1 + sum(statistic_standardized_flipped >= statistic_standardized)) /
      (n_flips + 1)
  }
  standardization_resampling_seconds <-
    proc.time()[["elapsed"]] - standardization_resample_begin

  scaled_eigenvalues <- eig$values / n
  list(
    df = df, n = n, n_flips = n_flips,
    statistic_effective = statistic_effective,
    statistic_standardized = statistic_standardized,
    p_effective = p_effective,
    p_standardized = p_standardized,
    p_standardized_chisq = if (standardization_available)
      stats::pchisq(statistic_standardized, df, lower.tail = FALSE) else NA_real_,
    mc_se_effective = sqrt(p_effective * (1 - p_effective) / (n_flips + 1)),
    mc_se_standardized = if (standardization_available)
      sqrt(p_standardized * (1 - p_standardized) / (n_flips + 1)) else NA_real_,
    standardization_available = standardization_available,
    empirical_eigen_min = min(scaled_eigenvalues),
    empirical_eigen_mean = mean(scaled_eigenvalues),
    empirical_eigen_cv = if (df > 1L)
      stats::sd(scaled_eigenvalues) / mean(scaled_eigenvalues) else 0,
    empirical_eigen_ratio = condition,
    empirical_identity_distance = sqrt(mean((scaled_eigenvalues - 1)^2)),
    resampling_score_seconds = score_seconds,
    standardization_setup_seconds = standardization_setup_seconds,
    resampling_standardization_seconds = standardization_resampling_seconds,
    total_resampling_seconds = proc.time()[["elapsed"]] - total_begin,
    projected = projected,
    flipped_statistics_effective = statistic_flipped,
    flipped_statistics_standardized = statistic_standardized_flipped)
}

residual_flip_gof <- function(fit, X, n_flips = 199L, seed = 1L) {
  total_begin <- proc.time()[["elapsed"]]
  if (!is.matrix(X)) X <- as.matrix(X)
  if (anyNA(X) || any(!is.finite(X))) {
    stop("residual_flip_gof(): complete finite raw data are required.",
         call. = FALSE)
  }
  core <- magmaan::magmaan_core
  setup_begin <- proc.time()[["elapsed"]]
  uf <- core$infer_build_u_factor(fit, bread = "expected",
                                  moments = "structured")
  if (!identical(uf$kind, "ProjectionExpected") || length(uf$blocks) != 1L ||
      isTRUE(uf$has_means)) {
    stop("residual_flip_gof(): first probe supports one covariance-only ML block.",
         call. = FALSE)
  }
  centred <- core$infer_casewise_contributions(fit$partable, X)
  residual_matrix <- uf$blocks[[1L]]$S - uf$blocks[[1L]]$Sigma_hat
  residual <- residual_matrix[lower.tri(residual_matrix, diag = TRUE)]
  if (ncol(centred) != length(residual) || nrow(uf$B) != length(residual)) {
    stop("residual_flip_gof(): covariance contribution layout mismatch.",
         call. = FALSE)
  }
  model_centred <- sweep(centred, 2L, residual, "+")
  projected <- model_centred %*% uf$B
  setup_seconds <- proc.time()[["elapsed"]] - setup_begin

  out <- .flip_projected_rows(projected, as.integer(n_flips), seed)
  out$setup_seconds <- setup_seconds
  out$total_seconds <- proc.time()[["elapsed"]] - total_begin
  out$residual_norm <- sqrt(sum(residual^2))
  out$projected <- NULL
  out$flipped_statistics_effective <- NULL
  out$flipped_statistics_standardized <- NULL
  class(out) <- "magmaan_residual_flip_gof_probe"
  out
}
