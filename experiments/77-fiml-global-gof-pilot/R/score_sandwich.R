score_sandwich_quadratic <- function(score, matrix, label) {
  matrix <- 0.5 * (matrix + t(matrix))
  factor <- tryCatch(chol(matrix), error = function(e) e)
  if (inherits(factor, "error")) {
    stop(label, " is not positive definite: ", conditionMessage(factor),
         call. = FALSE)
  }
  rhs <- forwardsolve(t(factor), score)
  solution <- backsolve(factor, rhs)
  as.numeric(crossprod(score, solution))
}

score_sandwich_condition <- function(matrix) {
  values <- eigen(
    0.5 * (matrix + t(matrix)), symmetric = TRUE,
    only.values = TRUE)$values
  c(
    min = min(values),
    condition = max(values) / min(values))
}

# Finite-sample diagnostics for the genuine sandwich-studentized projected
# score. The global score backend returns summed score, metric, and raw OPG
# meat in the same df-dimensional basis. Global centering estimates Var(psi);
# patternwise centering is deliberately avoided because pattern-conditional
# score means need not vanish under MAR.
#
# The two shrinkage weights are functions only of d/n and vanish for fixed d:
# no rejection-rate tuning is involved. Both shrink the centered meat toward
# its trace-matched expected-information metric.
score_sandwich_diagnostics <- function(score) {
  required <- c(
    "n_obs", "projected_score", "projected_metric", "projected_meat",
    "statistic_sandwich", "p_sandwich")
  missing <- setdiff(required, names(score))
  if (length(missing)) {
    stop("score result lacks sandwich geometry: ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
  u <- as.numeric(score$projected_score)
  metric <- as.matrix(score$projected_metric)
  meat_raw <- as.matrix(score$projected_meat)
  n <- as.integer(score$n_obs)
  d <- length(u)
  stopifnot(
    n > d, identical(dim(metric), c(d, d)),
    identical(dim(meat_raw), c(d, d)))

  meat_centered <- meat_raw - tcrossprod(u) / n
  centered_diagnostics <- score_sandwich_condition(meat_centered)
  if (!is.finite(centered_diagnostics[["min"]]) ||
      centered_diagnostics[["min"]] <= 0) {
    stop("centered projected meat is not positive definite", call. = FALSE)
  }
  statistic_centered <- score_sandwich_quadratic(
    u, meat_centered, "centered projected meat")
  statistic_hotelling <- (n - 1) / n * statistic_centered
  f_hotelling <- (n - d) / (d * (n - 1)) * statistic_hotelling

  metric_factor <- tryCatch(chol(0.5 * (metric + t(metric))),
                            error = function(e) e)
  if (inherits(metric_factor, "error")) {
    stop("projected metric is not positive definite: ",
         conditionMessage(metric_factor), call. = FALSE)
  }
  metric_inverse_meat <- backsolve(
    metric_factor,
    forwardsolve(t(metric_factor), meat_centered))
  target_scale <- sum(diag(metric_inverse_meat)) / d
  if (!is.finite(target_scale) || target_scale <= 0) {
    stop("trace-matched shrinkage target is not positive", call. = FALSE)
  }
  target <- target_scale * metric

  ratio <- d / n
  rho_light <- ratio / (1 + ratio)
  rho_sqrt <- sqrt(ratio) / (1 + sqrt(ratio))
  meat_light <- (1 - rho_light) * meat_centered + rho_light * target
  meat_sqrt <- (1 - rho_sqrt) * meat_centered + rho_sqrt * target
  statistic_light <- score_sandwich_quadratic(
    u, meat_light, "lightly regularized projected meat")
  statistic_sqrt <- score_sandwich_quadratic(
    u, meat_sqrt, "square-root regularized projected meat")

  raw_check <- score_sandwich_quadratic(
    u, meat_raw, "raw projected meat")
  tolerance <- 1e-7 * max(1, abs(score$statistic_sandwich))
  if (abs(raw_check - score$statistic_sandwich) > tolerance) {
    stop("returned projected meat does not reproduce sandwich statistic",
         call. = FALSE)
  }

  data.frame(
    sandwich_df = d,
    sandwich_n = n,
    sandwich_statistic_raw = raw_check,
    sandwich_p_raw = score$p_sandwich,
    sandwich_statistic_centered = statistic_centered,
    sandwich_p_centered_chisq = stats::pchisq(
      statistic_centered, d, lower.tail = FALSE),
    sandwich_statistic_hotelling = statistic_hotelling,
    sandwich_p_hotelling = stats::pf(
      f_hotelling, d, n - d, lower.tail = FALSE),
    sandwich_rho_light = rho_light,
    sandwich_statistic_shrink_light = statistic_light,
    sandwich_p_shrink_light = stats::pchisq(
      statistic_light, d, lower.tail = FALSE),
    sandwich_rho_sqrt = rho_sqrt,
    sandwich_statistic_shrink_sqrt = statistic_sqrt,
    sandwich_p_shrink_sqrt = stats::pchisq(
      statistic_sqrt, d, lower.tail = FALSE),
    sandwich_centered_min_eigenvalue =
      centered_diagnostics[["min"]],
    sandwich_centered_condition =
      centered_diagnostics[["condition"]],
    stringsAsFactors = FALSE)
}
