sem_power_alternatives <- c("sparse", "diffuse")

sem_power_pattern <- function(model, alternative) {
  if (!alternative %in% sem_power_alternatives) {
    stop("unknown power alternative: ", alternative, call. = FALSE)
  }
  sparse_pairs <- list(
    one_factor_6 = c(1L, 2L),
    two_factor_fmg_10 = c(1L, 6L),
    bifactor_12 = c(1L, 5L),
    three_factor_15 = c(1L, 6L),
    linear_growth_5 = c(2L, 3L))
  diffuse_items <- list(
    one_factor_6 = c(1L, 3L, 5L),
    two_factor_fmg_10 = c(1L, 3L, 6L, 8L),
    bifactor_12 = c(1L, 5L, 9L),
    three_factor_15 = c(1L, 6L, 11L),
    linear_growth_5 = c(1L, 3L, 5L))
  scale <- sqrt(diag(model$Sigma))
  direction <- matrix(0, model$p, model$p)
  if (alternative == "sparse") {
    pair <- sparse_pairs[[model$model_id]]
    direction[pair[[1L]], pair[[2L]]] <-
      scale[[pair[[1L]]]] * scale[[pair[[2L]]]]
    direction[pair[[2L]], pair[[1L]]] <-
      direction[pair[[1L]], pair[[2L]]]
    label <- paste0("single omitted residual covariance: ",
                    model$ov[[pair[[1L]]]], " ~~ ",
                    model$ov[[pair[[2L]]]])
    items <- paste(model$ov[pair], collapse = ",")
  } else {
    index <- diffuse_items[[model$model_id]]
    loading <- numeric(model$p)
    loading[index] <- scale[index]
    direction <- tcrossprod(loading)
    label <- paste0("omitted method factor on ",
                    paste(model$ov[index], collapse = ", "))
    items <- paste(model$ov[index], collapse = ",")
  }
  list(direction = direction, label = label, items = items)
}
sem_power_population <- function(model, truth = "null", effect = 0) {
  if (truth == "null") {
    return(list(Sigma = model$Sigma, mu = model$mu,
                alternative = "null", effect = 0))
  }
  pattern <- sem_power_pattern(model, truth)
  Sigma <- model$Sigma + effect * pattern$direction
  minimum <- min(eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values)
  if (!is.finite(minimum) || minimum <= 1e-8) {
    stop("power covariance is not positive definite for ", model$model_id,
         "/", truth, " at effect ", effect, call. = FALSE)
  }
  list(Sigma = Sigma, mu = model$mu, alternative = truth, effect = effect,
       label = pattern$label, items = pattern$items)
}

sem_power_target_ncp <- function(df, target_power = 0.50, alpha = 0.05) {
  critical <- stats::qchisq(1 - alpha, df)
  objective <- function(ncp) {
    stats::pchisq(critical, df, ncp = ncp, lower.tail = FALSE) - target_power
  }
  upper <- max(16, 2 * df)
  while (objective(upper) < 0) upper <- upper * 2
  stats::uniroot(objective, c(0, upper), tol = 1e-10)$root
}

sem_power_population_fml <- function(model, Sigma, mu = model$mu) {
  sample_stats <- list(
    S = list(Sigma), mean = list(mu), nobs = 1000000L)
  fit <- magmaan::magmaan(
    model$spec, sample_stats, estimator = "ML",
    optimizer = "nlopt-lbfgs-slsqp-fallback", se = "none", test = "none")
  if (!isTRUE(fit$converged) || !is.finite(fit$fmin)) {
    stop("population ML fit failed for ", model$model_id, call. = FALSE)
  }
  max(0, 2 * as.numeric(fit$fmin))
}

sem_calibrate_power_alternative <- function(model, alternative, n = 200L,
                                            target_power = 0.50,
                                            alpha = 0.05) {
  target_ncp <- sem_power_target_ncp(
    model$expected_df, target_power = target_power, alpha = alpha)
  target_fml <- target_ncp / (n - 1)
  discrepancy <- function(effect) {
    pop <- tryCatch(sem_power_population(model, alternative, effect),
                    error = function(e) e)
    if (inherits(pop, "error")) return(Inf)
    sem_power_population_fml(model, pop$Sigma, pop$mu)
  }
  lower <- 0
  upper <- 0.01
  upper_fml <- discrepancy(upper)
  while (is.finite(upper_fml) && upper_fml < target_fml && upper < 8) {
    upper <- upper * 2
    upper_fml <- discrepancy(upper)
  }
  if (!is.finite(upper_fml) || upper_fml < target_fml) {
    stop("could not bracket power calibration for ", model$model_id, "/",
         alternative, call. = FALSE)
  }
  effect <- stats::uniroot(
    function(value) discrepancy(value) - target_fml,
    c(lower, upper), tol = 1e-7)$root
  achieved_fml <- discrepancy(effect)
  critical <- stats::qchisq(1 - alpha, model$expected_df)
  implied_power <- stats::pchisq(
    critical, model$expected_df, ncp = (n - 1) * achieved_fml,
    lower.tail = FALSE)
  pattern <- sem_power_pattern(model, alternative)
  data.frame(
    model_id = model$model_id,
    model_label = model$model_label,
    alternative = alternative,
    alternative_label = pattern$label,
    affected_items = pattern$items,
    n = as.integer(n),
    alpha = alpha,
    target_power = target_power,
    target_ncp = target_ncp,
    target_fml = target_fml,
    effect = effect,
    achieved_fml = achieved_fml,
    implied_normal_lrt_power = implied_power,
    stringsAsFactors = FALSE)
}

sem_calibrate_power_design <- function(models, n = 200L,
                                       target_power = 0.50,
                                       alpha = 0.05) {
  rows <- lapply(models, function(model) {
    do.call(rbind, lapply(sem_power_alternatives, function(alternative) {
      sem_calibrate_power_alternative(
        model, alternative, n = n, target_power = target_power, alpha = alpha)
    }))
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

sem_read_power_calibration <- function(path, models, n) {
  if (is.null(path) || !file.exists(path)) {
    stop("power alternatives require --power-calibration-file", call. = FALSE)
  }
  out <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("model_id", "alternative", "n", "effect", "achieved_fml",
                "target_power", "implied_normal_lrt_power")
  if (!all(required %in% names(out))) {
    stop("malformed power calibration file: ", path, call. = FALSE)
  }
  wanted <- expand.grid(
    model_id = names(models), alternative = sem_power_alternatives,
    stringsAsFactors = FALSE)
  key <- paste(out$model_id, out$alternative, sep = "::")
  wanted_key <- paste(wanted$model_id, wanted$alternative, sep = "::")
  index <- match(wanted_key, key)
  if (anyNA(index) || any(out$n[index] != n) ||
      any(!is.finite(out$effect[index])) || any(out$effect[index] <= 0)) {
    stop("power calibration does not cover the selected models at n=", n,
         call. = FALSE)
  }
  out[index, , drop = FALSE]
}
