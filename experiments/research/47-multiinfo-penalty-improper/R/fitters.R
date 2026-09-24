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
      fit = function(spec, dat) frontier_fit_ml_multiinfo(spec, dat, weight = 2))
  )
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
    min_theta = {
      min(pt$est[pt$op == "~~" & pt$lhs == pt$rhs & !(pt$lhs %in% design$lv)])
    },
    factor_correlation = lm$Phi[1, 2] / sqrt(lm$Phi[1, 1] * lm$Phi[2, 2]),
    r2_f3 = 1 - lm$Psi[3, 3] / lm$Phi[3, 3]
  )
}

# Population parameter values in the model's own (marker) coordinates: the
# ordinary ML fit to the population covariance is exact.
population_truth <- function(design) {
  spec <- model_spec(design$syntax)
  fit <- magmaan_core$fit_ml(spec, list(S = list(design$Sigma), nobs = 1000000L))
  if (!isTRUE(fit$converged) || fit$fmin > 1e-10) {
    stop("population fit failed for design ", design$key, call. = FALSE)
  }
  rows <- free_rows(fit$partable)
  data.frame(
    param = param_label(fit$partable)[rows],
    truth = fit$partable$est[rows],
    stringsAsFactors = FALSE)
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
  implied <- magmaan_core$model_implied(fit)$sigma[[1]]
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
    est = fit$partable$est[rows],
    se = se,
    stringsAsFactors = FALSE)
  params$truth <- truth$truth[match(params$param, truth$param)]
  list(fit = base, params = params)
}
