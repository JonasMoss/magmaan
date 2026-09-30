# Estimators under comparison and per-fit records.

method_table <- function() {
  list(
    ml = list(
      label = "ML (unbounded)",
      fit = function(spec, dat) magmaan_core$fit_ml(spec, dat)),
    ml_bounded = list(
      label = "ML (variances >= 0)",
      fit = function(spec, dat) magmaan_core$fit_ml(spec, dat, bounds = "pos.var")),
    psd = list(
      label = "PSD-ML",
      fit = function(spec, dat) frontier_fit_ml_psd(spec, dat)),
    pen_l010 = list(
      label = "Penalized (lambda = 0.1)",
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.1)),
    pen_l025 = list(
      label = "Penalized (lambda = 0.25)",
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.25)),
    pen_l050 = list(
      label = "Penalized (lambda = 0.5)",
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 0.5)),
    pen_l100 = list(
      label = "Penalized (lambda = 1)",
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 1)),
    pen_l200 = list(
      label = "Penalized (lambda = 2)",
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 2)),
    det_l010 = det_method(0.1),
    det_l025 = det_method(0.25),
    det_l050 = det_method(0.5),
    det_l100 = det_method(1),
    det_l200 = det_method(2)
  )
}

# The latent-determinacy barrier: lambda log det Q, with Q the standardized
# covariance of the latents given the observed variables.
det_method <- function(weight) {
  force(weight)
  list(
    label = sprintf("Determinacy barrier (lambda = %g)", weight),
    fit = function(spec, dat) {
      frontier_fit_ml_multiinfo(spec, dat, weight = weight, target = "determinacy")
    })
}

param_label <- function(pt) paste(pt$lhs, pt$op, pt$rhs)

free_rows <- function(pt) {
  rows <- which(pt$free > 0L)
  rows[order(pt$free[rows])]
}

# Latent covariance and disturbance matrices rebuilt from the fitted partable.
latent_moments <- function(pt, lv) {
  m <- length(lv)
  B <- matrix(0, m, m, dimnames = list(lv, lv))
  Psi <- matrix(0, m, m, dimnames = list(lv, lv))
  for (i in seq_len(nrow(pt))) {
    lhs <- pt$lhs[[i]]
    rhs <- pt$rhs[[i]]
    if (!(lhs %in% lv && rhs %in% lv)) next
    if (pt$op[[i]] == "~") B[lhs, rhs] <- pt$est[[i]]
    if (pt$op[[i]] == "~~") Psi[lhs, rhs] <- Psi[rhs, lhs] <- pt$est[[i]]
  }
  A <- solve(diag(m) - B)
  list(Psi = Psi, Phi = A %*% Psi %*% t(A))
}

key_value <- function(design, pt) {
  lm <- latent_moments(pt, design$lv)
  switch(
    design$key_quantity,
    theta_x1 = pt$est[pt$op == "~~" & pt$lhs == "x1" & pt$rhs == "x1"],
    factor_correlation = lm$Phi[1, 2] / sqrt(lm$Phi[1, 1] * lm$Phi[2, 2]),
    r2_f3 = 1 - lm$Psi[3, 3] / lm$Phi[3, 3]
  )
}

# Population (pseudo-true) values in the model's own (marker) coordinates:
# the unconstrained ML fit to the population covariance, exact unless the
# design is misspecified. `se_unit` is the per-observation standard error
# (SE at N is se_unit / sqrt(N)) from the expected information there.
# `key_proper` is the key quantity of the PSD-constrained population fit.
population_truth <- function(design) {
  spec <- model_spec(design$syntax)
  big <- 1000000L
  pop <- list(S = list(design$Sigma), nobs = big)
  fit <- magmaan_core$fit_ml(spec, pop)
  exact <- !identical(design$kind, "misfit")
  if (!isTRUE(fit$converged) || (exact && fit$fmin > 1e-10)) {
    stop("population fit failed for design ", design$key, call. = FALSE)
  }
  rows <- free_rows(fit$partable)
  se <- tryCatch({
    info <- magmaan_core$inference_information_expected(fit)
    as.numeric(magmaan_core$infer_se(magmaan_core$inference_vcov(info, fit)))
  }, error = function(e) rep(NA_real_, length(rows)))
  psd <- frontier_fit_ml_psd(spec, pop)
  list(
    params = data.frame(
      param = param_label(fit$partable)[rows],
      truth = fit$partable$est[rows],
      se_unit = se * sqrt(big),
      stringsAsFactors = FALSE),
    key = key_value(design, fit$partable),
    key_proper = key_value(design, psd$partable),
    ncp_unit = 2 * fit$fmin)
}

vech_lower <- function(S) S[lower.tri(S, diag = TRUE)]

min_block_eigen <- function(blocks) {
  vals <- vapply(blocks, function(b) b$min_eigenvalue %||% NA_real_, numeric(1))
  if (!length(vals)) NA_real_ else min(vals)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

boundary_tol <- 1e-6

# One fit → one fit-level row plus parameter rows.
fit_record <- function(design, method, fit_fun, spec, dat, truth) {
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(suppressWarnings(fit_fun(spec, dat)), error = function(e) e)
  seconds <- proc.time()[["elapsed"]] - t0
  base <- data.frame(
    method = method, status = "ok", error = "", seconds = seconds,
    converged = NA, admissible = NA, min_eig_theta = NA_real_,
    min_eig_psi = NA_real_, classification = NA_character_,
    fmin = NA_real_, chi2 = NA_real_, df = NA_real_, pvalue = NA_real_,
    rls = NA_real_, browne = NA_real_,
    sigma_rmse = NA_real_, key_est = NA_real_, penalty = NA_real_,
    stringsAsFactors = FALSE)
  if (inherits(fit, "error")) {
    base$status <- "error"
    base$error <- conditionMessage(fit)
    return(list(fit = base, params = NULL))
  }
  adm <- fit$diagnostics$admissibility
  base$converged <- isTRUE(fit$converged)
  base$admissible <- isTRUE(adm$admissible)
  base$min_eig_theta <- min_block_eigen(adm$theta)
  base$min_eig_psi <- min_block_eigen(adm$psi)
  min_eig <- min(base$min_eig_theta, base$min_eig_psi, na.rm = TRUE)
  base$classification <- if (!base$admissible) {
    "improper"
  } else if (min_eig <= boundary_tol) {
    "boundary"
  } else {
    "interior"
  }
  base$fmin <- fit$fmin
  ss <- list(S = fit$S, nobs = fit$nobs, mean = fit$sample_mean)
  base$chi2 <- magmaan_core$infer_chi2_stat(ss, fit$fmin)
  base$df <- magmaan_core$infer_df_stat(fit$partable, ss)
  base$pvalue <- magmaan_core$infer_chi2_pvalue(base$chi2, base$df)
  # The unprojected RLS quadratic and Browne's projected residual statistic
  # (lavaan's browne.residual.nt.model), both at this method's estimate. They
  # coincide at the ML optimum. The projection removes the estimate's own
  # displacement, so `browne` tests the unconstrained model and chi2 - browne
  # is the admissibility part of the LR statistic.
  imp_all <- magmaan_core$model_implied(fit)
  base$rls <- tryCatch(
    magmaan_core$infer_nt_moment_quadratic_fit(fit, imp_all)$statistic,
    error = function(e) NA_real_)
  base$browne <- tryCatch({
    b <- magmaan_core$infer_rls_chi2_fit(fit, imp_all)
    if (is.list(b)) b$statistic else b
  }, error = function(e) NA_real_)
  implied <- imp_all$sigma[[1]]
  pop <- design$Sigma[fit$ov_names, fit$ov_names]
  base$sigma_rmse <- sqrt(mean((vech_lower(implied) - vech_lower(pop))^2))
  base$key_est <- tryCatch(key_value(design, fit$partable),
                           error = function(e) NA_real_)
  if (!is.null(fit$penalty)) base$penalty <- fit$penalty$value

  rows <- free_rows(fit$partable)
  se <- tryCatch({
    info <- magmaan_core$inference_information_expected(fit)
    as.numeric(magmaan_core$infer_se(magmaan_core$inference_vcov(info, fit)))
  }, error = function(e) rep(NA_real_, length(rows)))
  if (length(se) != length(rows)) se <- rep(NA_real_, length(rows))
  params <- data.frame(
    method = method,
    param = param_label(fit$partable)[rows],
    est = signif(fit$partable$est[rows], 7),
    se = signif(se, 7),
    stringsAsFactors = FALSE)
  params$truth <- signif(truth$truth[match(params$param, truth$param)], 7)
  list(fit = base, params = params)
}
