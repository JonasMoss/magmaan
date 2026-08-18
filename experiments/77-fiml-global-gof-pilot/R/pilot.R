pilot_variables <- function(p = 5L) paste0("x", seq_len(p))

pilot_pairs <- function(p = 5L) {
  out <- utils::combn(seq_len(p), 2L)
  colnames(out) <- NULL
  out
}

pilot_covariance_labels <- function(p = 5L) {
  pairs <- pilot_pairs(p)
  apply(pairs, 2L, function(z) paste0("c", z[[1L]], z[[2L]]))
}

pilot_model_syntax <- function(p = 5L, restricted = FALSE) {
  ov <- pilot_variables(p)
  pairs <- pilot_pairs(p)
  variances <- paste0(ov, " ~~ v", seq_len(p), "*", ov)
  covariances <- apply(pairs, 2L, function(z) {
    paste0(ov[[z[[1L]]]], " ~~ c", z[[1L]], z[[2L]], "*",
           ov[[z[[2L]]]])
  })
  means <- paste0(ov, " ~ m", seq_len(p), "*1")
  syntax <- c(variances, covariances, means)
  if (isTRUE(restricted)) {
    constraints <- paste0(pilot_covariance_labels(p), " == 0")
    syntax <- c(syntax, constraints)
  }
  paste(syntax, collapse = "\n")
}

pilot_specs <- function(p = 5L) {
  list(
    H1 = magmaan::model_spec(
      pilot_model_syntax(p, restricted = FALSE), meanstructure = TRUE),
    H0 = magmaan::model_spec(
      pilot_model_syntax(p, restricted = TRUE), meanstructure = TRUE)
  )
}

pilot_design <- function(n = 120L, p = 5L) {
  out <- expand.grid(
    distribution = c("normal", "skewed"),
    missing_rate = c(0, 0.30),
    truth = c("null", "power"),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  out$n <- as.integer(n)
  out$p <- as.integer(p)
  out$df <- as.integer(p * (p - 1L) / 2L)
  out$cell_id <- seq_len(nrow(out))
  out
}

pilot_draw <- function(cell, seed) {
  set.seed(seed)
  p <- cell$p
  correlation <- diag(p)
  if (identical(cell$truth, "power")) {
    correlation[1L, 2L] <- correlation[2L, 1L] <- 0.35
  }
  Z <- matrix(stats::rnorm(cell$n * p), cell$n, p) %*% chol(correlation)
  X <- if (identical(cell$distribution, "normal")) {
    Z
  } else {
    # Standardized log-normal margins: pronounced skew/kurtosis while the null
    # remains exact because the Gaussian copula is independent in null cells.
    a <- 0.8
    Y <- exp(a * Z)
    centre <- exp(a^2 / 2)
    scale <- sqrt((exp(a^2) - 1) * exp(a^2))
    (Y - centre) / scale
  }
  colnames(X) <- pilot_variables(p)
  if (cell$missing_rate > 0) {
    mask <- matrix(
      stats::runif(cell$n * (p - 1L)) < cell$missing_rate,
      cell$n, p - 1L
    )
    X[, -1L][mask] <- NA_real_
  }
  as.data.frame(X)
}

pilot_restriction_matrix <- function(fit, p) {
  labels <- pilot_covariance_labels(p)
  index <- match(labels, as.character(fit$partable$label))
  if (anyNA(index)) stop("saturated fit is missing covariance labels",
                         call. = FALSE)
  free <- as.integer(fit$partable$free[index])
  if (any(free <= 0L) || anyDuplicated(free)) {
    stop("covariance labels do not identify distinct free parameters",
         call. = FALSE)
  }
  R <- matrix(0, nrow = length(free), ncol = fit$npar)
  R[cbind(seq_along(free), free)] <- 1
  R
}

pilot_p_columns <- c(
  "p_lrt_naive", "p_lrt_mlr", "p_lrt_sb", "p_lrt_ss",
  "p_lrt_peba4", "p_lrt_all",
  "p_score_chisq", "p_score_sb", "p_score_ss", "p_score_peba4",
  "p_score_all", "p_score_sandwich", "p_flip_effective",
  "p_wald_model", "p_wald_robust"
)

pilot_empty_p <- function() {
  stats::setNames(rep(NA_real_, length(pilot_p_columns)), pilot_p_columns)
}

pilot_one_rep <- function(cell, rep_id, specs, flips, seed_base) {
  total_begin <- proc.time()[["elapsed"]]
  p_values <- pilot_empty_p()
  seed <- seed_base + cell$cell_id * 100003L + rep_id
  data <- pilot_draw(cell, seed)
  realized_missing <- mean(is.na(as.matrix(data)))
  base <- data.frame(
    rep = rep_id, fit_h0_ok = FALSE, lrt_ok = FALSE, score_ok = FALSE,
    fit_h1_ok = FALSE, wald_ok = FALSE, error_h0 = "", error_lrt = "",
    error_score = "", error_h1 = "", error_wald = "",
    realized_missing = realized_missing, h0_seconds = NA_real_,
    lrt_seconds = NA_real_, score_seconds = NA_real_, h1_seconds = NA_real_,
    wald_seconds = NA_real_, total_seconds = NA_real_,
    score_df = NA_integer_, score_eigen_min = NA_real_,
    score_eigen_mean = NA_real_, score_eigen_cv = NA_real_,
    score_sandwich_condition = NA_real_, stringsAsFactors = FALSE
  )

  h0_begin <- proc.time()[["elapsed"]]
  fit_h0 <- tryCatch(magmaan::magmaan(
    specs$H0, data, estimator = "FIML", se = "none", test = "none",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)),
    error = function(e) e)
  base$h0_seconds <- proc.time()[["elapsed"]] - h0_begin
  if (inherits(fit_h0, "error") || !isTRUE(fit_h0$converged)) {
    base$error_h0 <- if (inherits(fit_h0, "error")) conditionMessage(fit_h0)
      else "restricted FIML fit did not converge"
    base$total_seconds <- proc.time()[["elapsed"]] - total_begin
    return(cbind(base, as.data.frame(as.list(p_values))))
  }
  base$fit_h0_ok <- TRUE

  lrt_begin <- proc.time()[["elapsed"]]
  lrt <- tryCatch({
    fmg <- magmaan::fmg_tests(
      fit_h0, tests = c("SB", "SS", "pEBA4", "all"))
    mlr <- magmaan::magmaan_core$estimate_fiml_robust_mlr(fit_h0)
    list(fmg = fmg, mlr = mlr)
  }, error = function(e) e)
  base$lrt_seconds <- proc.time()[["elapsed"]] - lrt_begin
  if (inherits(lrt, "error")) {
    base$error_lrt <- conditionMessage(lrt)
  } else {
    key <- sub("_ml$", "", lrt$fmg$label)
    lookup <- stats::setNames(lrt$fmg$p_value, key)
    p_values[c("p_lrt_naive", "p_lrt_mlr", "p_lrt_sb", "p_lrt_ss",
               "p_lrt_peba4", "p_lrt_all")] <- c(
      stats::pchisq(lrt$fmg$base_statistic[[1L]], lrt$fmg$df[[1L]],
                    lower.tail = FALSE),
      stats::pchisq(lrt$mlr$chisq_scaled, lrt$mlr$df, lower.tail = FALSE),
      lookup[["sb"]], lookup[["ss"]], lookup[["peba4"]], lookup[["all"]]
    )
    base$lrt_ok <- TRUE
  }

  score_begin <- proc.time()[["elapsed"]]
  score <- tryCatch(magmaan::score_flip_test(
    specs$H1, fit_h0, n_flips = flips, seed = seed + 70000001L,
    calibration = "effective"), error = function(e) e)
  base$score_seconds <- proc.time()[["elapsed"]] - score_begin
  if (inherits(score, "error")) {
    base$error_score <- conditionMessage(score)
  } else {
    score_ss <- magmaan::magmaan_core$robust_fmg_test(
      score$statistic_effective, score$df, score$eigenvalues,
      method = "ss", param = 4, truncate_negative = TRUE)
    score_peba4 <- magmaan::magmaan_core$robust_fmg_test(
      score$statistic_effective, score$df, score$eigenvalues,
      method = "peba", param = 4, truncate_negative = TRUE)
    p_values[c("p_score_chisq", "p_score_sb", "p_score_ss",
               "p_score_peba4", "p_score_all", "p_score_sandwich",
               "p_flip_effective")] <- c(
      score$p_chisq, score$p_mean_scaled, score_ss$p_value,
      score_peba4$p_value, score$p_mixture, score$p_sandwich,
      score$p_effective)
    base$score_ok <- TRUE
    base$score_df <- score$df
    base$score_eigen_min <- min(score$eigenvalues)
    base$score_eigen_mean <- mean(score$eigenvalues)
    base$score_eigen_cv <- stats::sd(score$eigenvalues) /
      mean(score$eigenvalues)
    base$score_sandwich_condition <- score$sandwich_condition
  }

  h1_begin <- proc.time()[["elapsed"]]
  fit_h1 <- tryCatch(magmaan::magmaan(
    specs$H1, data, estimator = "FIML", se = "none", test = "none",
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)),
    error = function(e) e)
  base$h1_seconds <- proc.time()[["elapsed"]] - h1_begin
  if (inherits(fit_h1, "error") || !isTRUE(fit_h1$converged)) {
    base$error_h1 <- if (inherits(fit_h1, "error")) conditionMessage(fit_h1)
      else "saturated FIML fit did not converge"
  } else {
    base$fit_h1_ok <- TRUE
    wald_begin <- proc.time()[["elapsed"]]
    wald <- tryCatch({
      R <- pilot_restriction_matrix(fit_h1, cell$p)
      model_vcov <- stats::vcov(fit_h1, regime = "model")
      robust_vcov <- stats::vcov(fit_h1, regime = "robust")
      list(
        model = magmaan::magmaan_core$inference_wald_test(
          fit_h1, R, model_vcov),
        robust = magmaan::magmaan_core$inference_wald_test(
          fit_h1, R, robust_vcov)
      )
    }, error = function(e) e)
    base$wald_seconds <- proc.time()[["elapsed"]] - wald_begin
    if (inherits(wald, "error")) {
      base$error_wald <- conditionMessage(wald)
    } else {
      p_values[c("p_wald_model", "p_wald_robust")] <-
        c(wald$model$pvalue, wald$robust$pvalue)
      base$wald_ok <- TRUE
    }
  }

  base$total_seconds <- proc.time()[["elapsed"]] - total_begin
  cbind(base, as.data.frame(as.list(p_values)))
}

pilot_wilson <- function(rejected, n, z = 1.95996398454005) {
  if (!n) return(c(lower = NA_real_, upper = NA_real_))
  phat <- rejected / n
  denominator <- 1 + z^2 / n
  centre <- (phat + z^2 / (2 * n)) / denominator
  half <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) /
    denominator
  c(lower = centre - half, upper = centre + half)
}

pilot_method_summary <- function(raw) {
  long <- do.call(rbind, lapply(pilot_p_columns, function(column) {
    data.frame(
      raw[c("cell_id", "distribution", "missing_rate", "truth", "n", "p",
            "df", "rep")],
      method = sub("^p_", "", column), p_value = raw[[column]],
      stringsAsFactors = FALSE
    )
  }))
  groups <- split(long, interaction(long$cell_id, long$method, drop = TRUE))
  summary <- do.call(rbind, lapply(groups, function(x) {
    valid <- is.finite(x$p_value)
    n <- sum(valid)
    rejected <- sum(x$p_value[valid] <= 0.05)
    ci <- pilot_wilson(rejected, n)
    data.frame(
      x[1L, c("cell_id", "distribution", "missing_rate", "truth", "n", "p",
               "df", "method")],
      valid = n, rejected = rejected,
      rejection_rate = if (n) rejected / n else NA_real_,
      ci_lower = ci[["lower"]], ci_upper = ci[["upper"]],
      stringsAsFactors = FALSE
    )
  }))
  summary <- summary[order(summary$cell_id, summary$method), ]
  row.names(summary) <- NULL
  list(long = long, summary = summary)
}
