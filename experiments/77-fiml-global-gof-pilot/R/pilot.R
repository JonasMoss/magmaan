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

pilot_distribution_catalog <- function() {
  data.frame(
    distribution = c(
      "normal", "t8", "t5", "lognormal_08", "lognormal_12",
      "contaminated_05x6", "t3"),
    distribution_label = c(
      "Gaussian", "t(8)", "t(5)", "log-normal, g=0.8",
      "log-normal, g=1.2", "5% row-scale contamination (x6)", "t(3)"),
    distribution_class = c(
      "Gaussian reference", "elliptical heavy tail", "elliptical heavy tail",
      "skewed finite moments", "skewed finite moments",
      "finite-moment contamination", "infinite fourth moment"),
    fourth_moment = c(
      "finite", "finite", "finite", "finite", "finite", "finite",
      "infinite"),
    stringsAsFactors = FALSE
  )
}

pilot_missingness_catalog <- function() {
  data.frame(
    missingness = c(
      "complete", "mcar_30", "mar_tail_30", "mar_monotone_30",
      "mnar_tail_30"),
    missingness_label = c(
      "complete", "30% MCAR", "30% MAR on observed x1",
      "30% monotone MAR dropout", "30% self-masked MNAR"),
    missingness_class = c(
      "complete", "MCAR", "MAR", "MAR monotone", "MNAR"),
    missing_rate = c(0, 0.30, 0.30, 0.30, 0.30),
    stringsAsFactors = FALSE
  )
}

pilot_analysis_region <- function(distribution, missingness, fourth_moment) {
  if (identical(missingness, "mnar_tail_30")) return("ignorability failure")
  if (identical(fourth_moment, "infinite")) return("moment boundary")
  if (grepl("^mar_", missingness) && !identical(distribution, "normal")) {
    return("MAR pseudo-ML stress")
  }
  if (!identical(distribution, "normal")) return("finite-moment robustness")
  "Gaussian reference"
}

pilot_design <- function(n = 120L, p = 5L) {
  distributions <- pilot_distribution_catalog()
  missingness <- pilot_missingness_catalog()
  index <- expand.grid(
    distribution_id = seq_len(nrow(distributions)),
    missingness_id = seq_len(nrow(missingness)),
    KEEP.OUT.ATTRS = FALSE
  )
  scenarios <- cbind(
    distributions[index$distribution_id, , drop = FALSE],
    missingness[index$missingness_id, , drop = FALSE])
  row.names(scenarios) <- NULL
  scenarios$scenario_id <- seq_len(nrow(scenarios))
  scenarios$analysis_region <- mapply(
    pilot_analysis_region, scenarios$distribution, scenarios$missingness,
    scenarios$fourth_moment, USE.NAMES = FALSE)
  scenarios$null_contract <- ifelse(
    scenarios$analysis_region %in%
      c("Gaussian reference", "finite-moment robustness"),
    "primary robustness target", ifelse(
      scenarios$analysis_region == "MAR pseudo-ML stress",
      "common-use stress", "deliberate contract violation"))

  out <- scenarios[rep(seq_len(nrow(scenarios)), each = 2L), , drop = FALSE]
  out$truth <- rep(c("null", "power"), times = nrow(scenarios))
  out$n <- as.integer(n)
  out$p <- as.integer(p)
  out$df <- as.integer(p * (p - 1L) / 2L)
  out$alternative_rho <- ifelse(out$truth == "power", 0.35, 0)
  out$cell_id <- seq_len(nrow(out))
  row.names(out) <- NULL
  out
}

pilot_standardized_lognormal <- function(z, g) {
  y <- exp(g * z)
  centre <- exp(g^2 / 2)
  scale <- sqrt((exp(g^2) - 1) * exp(g^2))
  (y - centre) / scale
}

pilot_draw_complete <- function(cell) {
  p <- as.integer(cell$p)
  correlation <- diag(p)
  if (identical(cell$truth, "power")) {
    correlation[1L, 2L] <- correlation[2L, 1L] <- cell$alternative_rho
  }
  Z <- matrix(stats::rnorm(cell$n * p), cell$n, p) %*% chol(correlation)
  if (identical(cell$distribution, "normal")) return(Z)
  if (identical(cell$distribution, "lognormal_08")) {
    return(pilot_standardized_lognormal(Z, 0.8))
  }
  if (identical(cell$distribution, "lognormal_12")) {
    return(pilot_standardized_lognormal(Z, 1.2))
  }
  if (cell$distribution %in% c("t8", "t5", "t3")) {
    degrees <- as.numeric(sub("^t", "", cell$distribution))
    row_scale <- sqrt(stats::rchisq(cell$n, degrees) / degrees)
    return(sweep(Z, 1L, row_scale, "/") * sqrt((degrees - 2) / degrees))
  }
  if (identical(cell$distribution, "contaminated_05x6")) {
    row_scale <- ifelse(stats::runif(cell$n) < 0.05, 6, 1)
    variance_scale <- sqrt(0.95 + 0.05 * 6^2)
    return(sweep(Z, 1L, row_scale / variance_scale, "*"))
  }
  stop("unknown pilot distribution: ", cell$distribution, call. = FALSE)
}

pilot_missingness_driver <- function(x) {
  scale <- stats::sd(x)
  if (!is.finite(scale) || scale <= sqrt(.Machine$double.eps)) {
    return(rep(0, length(x)))
  }
  pmax(-6, pmin(6, (x - mean(x)) / scale))
}

pilot_logit_intercept <- function(linear_predictor, target, monotone_p = NULL) {
  objective <- function(intercept) {
    probability <- stats::plogis(intercept + linear_predictor)
    expected <- if (is.null(monotone_p)) {
      mean(probability)
    } else {
      mean(vapply(seq_len(monotone_p - 1L), function(step) {
        mean(1 - (1 - probability)^step)
      }, numeric(1L)))
    }
    expected - target
  }
  stats::uniroot(objective, interval = c(-30, 30), tol = 1e-10)$root
}

pilot_apply_missingness <- function(X, cell) {
  n <- nrow(X)
  p <- ncol(X)
  mechanism <- cell$missingness
  if (identical(mechanism, "complete")) return(X)

  mask <- matrix(FALSE, n, p - 1L)
  if (identical(mechanism, "mcar_30")) {
    mask[] <- stats::runif(n * (p - 1L)) < cell$missing_rate
  } else if (identical(mechanism, "mar_tail_30")) {
    eta <- 1.25 * pilot_missingness_driver(X[, 1L])
    intercept <- pilot_logit_intercept(eta, cell$missing_rate)
    probability <- stats::plogis(intercept + eta)
    mask[] <- stats::runif(n * (p - 1L)) < probability
  } else if (identical(mechanism, "mar_monotone_30")) {
    eta <- 1.25 * pilot_missingness_driver(X[, 1L])
    intercept <- pilot_logit_intercept(
      eta, cell$missing_rate, monotone_p = p)
    probability <- stats::plogis(intercept + eta)
    dropped <- rep(FALSE, n)
    for (column in seq_len(p - 1L)) {
      dropped <- dropped | (stats::runif(n) < probability)
      mask[, column] <- dropped
    }
  } else if (identical(mechanism, "mnar_tail_30")) {
    for (column in seq_len(p - 1L)) {
      eta <- 1.25 * pilot_missingness_driver(X[, column + 1L])
      intercept <- pilot_logit_intercept(eta, cell$missing_rate)
      probability <- stats::plogis(intercept + eta)
      mask[, column] <- stats::runif(n) < probability
    }
  } else {
    stop("unknown pilot missingness mechanism: ", mechanism, call. = FALSE)
  }

  eligible <- X[, -1L, drop = FALSE]
  eligible[mask] <- NA_real_
  X[, -1L] <- eligible
  X
}

pilot_draw <- function(cell, seed) {
  set.seed(seed)
  X_complete <- pilot_draw_complete(cell)
  complete_r12 <- stats::cor(X_complete[, 1L], X_complete[, 2L])
  X <- pilot_apply_missingness(X_complete, cell)
  colnames(X) <- pilot_variables(cell$p)
  out <- as.data.frame(X)
  attr(out, "complete_r12") <- complete_r12
  out
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
  realized_missing_eligible <- mean(is.na(as.matrix(data[, -1L, drop = FALSE])))
  complete_r12 <- attr(data, "complete_r12")
  observed_r12 <- suppressWarnings(stats::cor(
    data[[1L]], data[[2L]], use = "complete.obs"))
  base <- data.frame(
    rep = rep_id, fit_h0_ok = FALSE, lrt_ok = FALSE, score_ok = FALSE,
    fit_h1_ok = FALSE, wald_ok = FALSE, error_h0 = "", error_lrt = "",
    error_score = "", error_h1 = "", error_wald = "",
    realized_missing = realized_missing,
    realized_missing_eligible = realized_missing_eligible,
    complete_r12 = complete_r12, observed_r12 = observed_r12,
    h0_seconds = NA_real_,
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
  design_columns <- intersect(c(
    "cell_id", "scenario_id", "distribution", "distribution_label",
    "distribution_class", "fourth_moment", "missingness",
    "missingness_label", "missingness_class", "missing_rate",
    "analysis_region", "null_contract", "truth", "alternative_rho",
    "n", "p", "df", "rep"), names(raw))
  long <- do.call(rbind, lapply(pilot_p_columns, function(column) {
    data.frame(
      raw[design_columns],
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
    summary_columns <- setdiff(design_columns, "rep")
    data.frame(
      x[1L, c(summary_columns, "method")],
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
