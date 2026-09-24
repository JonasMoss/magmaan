study2_seed <- function(x) {
  as.integer(x %% (.Machine$integer.max - 1))
}

study2_distribution_moments <- function(distribution, p) {
  if (identical(distribution, "normal")) {
    return(list(skew = rep(0, p), exkurt = rep(0, p)))
  }
  severity <- substring(distribution, nchar(distribution))
  if (identical(severity, "1")) {
    list(skew = rep(2, p), exkurt = rep(7, p))
  } else if (identical(severity, "2")) {
    list(skew = rep(3, p), exkurt = rep(21, p))
  } else {
    stop("unknown Study 2 distribution: ", distribution, call. = FALSE)
  }
}

study2_calibrate_sampler <- function(population, distribution) {
  begin <- proc.time()[["elapsed"]]
  moments <- study2_distribution_moments(distribution, population$p)
  Sigma <- population$Sigma
  sds <- sqrt(diag(Sigma))
  kind <- sub("[12]$", "", distribution)
  calibration <- switch(
    kind,
    normal = NULL,
    vm = magmaan:::sim_vm_calibrate_impl(
      stats::cov2cor(Sigma), moments$skew, moments$exkurt),
    ig = magmaan:::sim_ig_calibrate_impl(
      Sigma, moments$skew, moments$exkurt,
      root = "symmetric", generator_family = "pearson",
      quadrature_points = 81L),
    pl = magmaan:::sim_plsim_calibrate_impl(
      stats::cov2cor(Sigma), moments$skew, moments$exkurt,
      method = "hermite_then_rectangle", num_segments = 12L,
      quadrature_points = 31L, hermite_order = 24L),
    stop("unknown Study 2 distribution: ", distribution, call. = FALSE)
  )
  list(
    kind = kind,
    distribution = distribution,
    Sigma = Sigma,
    chol = if (kind == "normal") chol(Sigma) else NULL,
    scale = if (kind %in% c("vm", "pl")) sds else rep(1, population$p),
    calibration = calibration,
    setup_seconds = proc.time()[["elapsed"]] - begin
  )
}

study2_draw_group <- function(sampler, n, seed) {
  seed <- study2_seed(seed)
  p <- ncol(sampler$Sigma)
  if (sampler$kind == "normal") {
    set.seed(seed)
    return(matrix(stats::rnorm(n * p), nrow = n) %*% sampler$chol)
  }
  batch <- switch(
    sampler$kind,
    vm = magmaan:::sim_vm_draw_impl(
      sampler$calibration, n = n, reps = 1L, seed_base = seed),
    ig = magmaan:::sim_ig_draw_impl(
      sampler$calibration, n = n, reps = 1L, seed_base = seed,
      quadrature_points = 81L),
    pl = magmaan:::sim_plsim_draw_impl(
      sampler$calibration, n = n, reps = 1L, seed_base = seed),
    stop("unknown sampler kind: ", sampler$kind, call. = FALSE)
  )
  sweep(batch$draws[[1L]], 2L, sampler$scale, "*")
}

study2_draw_data <- function(cell, sampler, seed) {
  blocks <- lapply(seq_len(cell$groups), function(g) {
    X <- study2_draw_group(
      sampler, cell$n_group, seed + g * 100003L)
    colnames(X) <- paste0("x", seq_len(cell$p))
    storage.mode(X) <- "double"
    X
  })
  labels <- paste0("g", seq_len(cell$groups))
  data <- data.frame(
    do.call(rbind, blocks),
    group = factor(rep(labels, each = cell$n_group), levels = labels),
    check.names = FALSE
  )
  list(blocks = blocks, data = data)
}

study2_draw_diagnostics <- function(blocks, sampler) {
  X <- do.call(rbind, blocks)
  centre <- colMeans(X)
  Xc <- sweep(X, 2L, centre, "-")
  second <- colMeans(Xc^2)
  skew <- colMeans(Xc^3) / second^(3 / 2)
  exkurt <- colMeans(Xc^4) / second^2 - 3
  target_moments <- study2_distribution_moments(
    sampler$distribution, ncol(X))
  target_cov <- sampler$Sigma
  target_corr <- stats::cov2cor(target_cov)
  observed_corr <- stats::cor(X)
  off_diagonal <- row(target_corr) != col(target_corr)
  c(
    draw_mean_skewness = mean(skew),
    draw_mean_excess_kurtosis = mean(exkurt),
    draw_max_abs_skewness_error =
      max(abs(skew - target_moments$skew)),
    draw_max_abs_excess_kurtosis_error =
      max(abs(exkurt - target_moments$exkurt)),
    draw_max_abs_correlation_error =
      max(abs(observed_corr[off_diagonal] - target_corr[off_diagonal])),
    draw_max_relative_variance_error =
      max(abs(second - diag(target_cov)) / diag(target_cov))
  )
}

study2_sampler_diagnostics <- function(sampler) {
  p <- ncol(sampler$Sigma)
  moments <- study2_distribution_moments(sampler$distribution, p)
  intermediate_min_eigen <- NA_real_
  calibration_max_abs_correlation_error <- NA_real_
  if (sampler$kind == "normal") {
    intermediate_min_eigen <- min(eigen(
      stats::cov2cor(sampler$Sigma), symmetric = TRUE,
      only.values = TRUE)$values)
    calibration_max_abs_correlation_error <- 0
  } else if (sampler$kind == "vm") {
    calibration <- sampler$calibration
    coefficients <- calibration$coefficients
    rho <- calibration$intermediate_corr
    b <- coefficients[, 2L]
    c2 <- coefficients[, 3L]
    d <- coefficients[, 4L]
    achieved <- outer(b, b) * rho +
      3 * (outer(b, d) + outer(d, b)) * rho +
      9 * outer(d, d) * rho +
      2 * outer(c2, c2) * rho^2 +
      6 * outer(d, d) * rho^3
    diag(achieved) <- 1
    target <- calibration$target_corr
    intermediate_min_eigen <- min(eigen(
      rho, symmetric = TRUE, only.values = TRUE)$values)
    calibration_max_abs_correlation_error <- max(abs(achieved - target))
  } else if (sampler$kind == "pl") {
    calibration <- sampler$calibration
    intermediate_min_eigen <- min(eigen(
      calibration$intermediate_corr, symmetric = TRUE,
      only.values = TRUE)$values)
    calibration_max_abs_correlation_error <- max(abs(
      calibration$achieved_corr - stats::cov2cor(sampler$Sigma)))
  } else if (sampler$kind == "ig") {
    intermediate_min_eigen <- min(svd(
      sampler$calibration$root, nu = 0L, nv = 0L)$d)
  }
  data.frame(
    p = p,
    distribution = sampler$distribution,
    generator = sampler$kind,
    target_skewness = moments$skew[[1L]],
    target_excess_kurtosis = moments$exkurt[[1L]],
    setup_seconds = sampler$setup_seconds,
    intermediate_min_eigen = intermediate_min_eigen,
    calibration_max_abs_correlation_error =
      calibration_max_abs_correlation_error,
    calibration_ok = is.finite(intermediate_min_eigen) &&
      intermediate_min_eigen > 0 &&
      (!is.finite(calibration_max_abs_correlation_error) ||
         calibration_max_abs_correlation_error <= 1e-6),
    disk_cache_hit = isTRUE(sampler$disk_cache_hit),
    stringsAsFactors = FALSE
  )
}

study2_capture <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warnings))
}

study2_empty_replication <- function(cell, rep_id, plan, error = "") {
  p <- stats::setNames(
    as.list(rep(NA_real_, nrow(plan))),
    paste0("p_", plan$method_id)
  )
  base <- data.frame(
    rep = as.integer(rep_id),
    fit_ok = FALSE,
    nested_ok = FALSE,
    fit_error = error,
    nested_error = "",
    fit_warnings = "",
    nested_warnings = "",
    draw_seconds = NA_real_,
    fit_seconds = NA_real_,
    nested_seconds = NA_real_,
    replication_seconds = NA_real_,
    biased_eigen_mean = NA_real_,
    biased_eigen_cv = NA_real_,
    unbiased_eigen_mean = NA_real_,
    unbiased_eigen_cv = NA_real_,
    draw_mean_skewness = NA_real_,
    draw_mean_excess_kurtosis = NA_real_,
    draw_max_abs_skewness_error = NA_real_,
    draw_max_abs_excess_kurtosis_error = NA_real_,
    draw_max_abs_correlation_error = NA_real_,
    draw_max_relative_variance_error = NA_real_,
    stringsAsFactors = FALSE
  )
  for (name in names(cell)) base[[name]] <- cell[[name]]
  cbind(base, as.data.frame(p, check.names = FALSE))
}

study2_one_rep <- function(cell, rep_id, sampler, specs, seed,
                           return_details = FALSE) {
  plan <- study2_test_plan(cell$df)
  begin <- proc.time()[["elapsed"]]
  draw_begin <- proc.time()[["elapsed"]]
  sample <- study2_draw_data(cell, sampler, seed)
  draw_diagnostics <- study2_draw_diagnostics(sample$blocks, sampler)
  draw_seconds <- proc.time()[["elapsed"]] - draw_begin

  fit_begin <- proc.time()[["elapsed"]]
  fits <- tryCatch(study2_capture(lapply(specs, function(spec) {
    magmaan(
      spec, sample$data, estimator = "ML",
      optimizer = "nlopt-lbfgs-slsqp-fallback",
      se = "none", test = "none"
    )
  })), error = function(e) e)
  fit_seconds <- proc.time()[["elapsed"]] - fit_begin
  if (inherits(fits, "error")) {
    row <- study2_empty_replication(
      cell, rep_id, plan, paste0("fit: ", conditionMessage(fits)))
    row$draw_seconds <- draw_seconds
    for (name in names(draw_diagnostics)) {
      row[[name]] <- draw_diagnostics[[name]]
    }
    row$fit_seconds <- fit_seconds
    row$replication_seconds <- proc.time()[["elapsed"]] - begin
    return(if (return_details) list(row = row) else row)
  }
  names(fits$value) <- names(specs)
  if (!all(vapply(fits$value, function(x) isTRUE(x$converged), logical(1)))) {
    row <- study2_empty_replication(
      cell, rep_id, plan, "one or both ML fits did not converge")
    row$draw_seconds <- draw_seconds
    for (name in names(draw_diagnostics)) {
      row[[name]] <- draw_diagnostics[[name]]
    }
    row$fit_seconds <- fit_seconds
    row$fit_warnings <- paste(fits$warnings, collapse = " | ")
    row$replication_seconds <- proc.time()[["elapsed"]] - begin
    return(if (return_details) list(row = row) else row)
  }

  nested_begin <- proc.time()[["elapsed"]]
  nested <- tryCatch(study2_capture(fmg_nested(
    fits$value$H1, fits$value$H0,
    data = sample$blocks,
    tests = plan$input,
    A.method = "delta"
  )), error = function(e) e)
  nested_seconds <- proc.time()[["elapsed"]] - nested_begin
  if (inherits(nested, "error")) {
    row <- study2_empty_replication(cell, rep_id, plan)
    row$fit_ok <- TRUE
    row$nested_error <- paste0("nested: ", conditionMessage(nested))
    row$draw_seconds <- draw_seconds
    for (name in names(draw_diagnostics)) {
      row[[name]] <- draw_diagnostics[[name]]
    }
    row$fit_seconds <- fit_seconds
    row$nested_seconds <- nested_seconds
    row$fit_warnings <- paste(fits$warnings, collapse = " | ")
    row$replication_seconds <- proc.time()[["elapsed"]] - begin
    return(if (return_details) {
      list(row = row, sample = sample, fits = fits$value)
    } else row)
  }
  result <- nested$value
  stopifnot(
    nrow(result) == nrow(plan),
    all(result$df == cell$df),
    identical(as.character(result$input), as.character(plan$input))
  )

  p <- stats::setNames(
    as.list(result$p_value),
    paste0("p_", plan$method_id)
  )
  biased <- result$eigenvalues[[which(!result$ug)[1L]]]
  unbiased <- result$eigenvalues[[which(result$ug)[1L]]]
  eig_cv <- function(x) {
    if (length(x) <= 1L || !is.finite(mean(x)) || mean(x) == 0) return(0)
    stats::sd(x) / mean(x)
  }
  nested_warnings <- unique(c(nested$warnings, attr(result, "warnings")))
  nested_warnings <- nested_warnings[
    !is.na(nested_warnings) & nzchar(nested_warnings)
  ]
  row <- data.frame(
    rep = as.integer(rep_id),
    fit_ok = TRUE,
    nested_ok = TRUE,
    fit_error = "",
    nested_error = "",
    fit_warnings = paste(fits$warnings, collapse = " | "),
    nested_warnings = paste(nested_warnings, collapse = " | "),
    draw_seconds = draw_seconds,
    fit_seconds = fit_seconds,
    nested_seconds = nested_seconds,
    replication_seconds = proc.time()[["elapsed"]] - begin,
    biased_eigen_mean = mean(biased),
    biased_eigen_cv = eig_cv(biased),
    unbiased_eigen_mean = mean(unbiased),
    unbiased_eigen_cv = eig_cv(unbiased),
    draw_mean_skewness = draw_diagnostics[["draw_mean_skewness"]],
    draw_mean_excess_kurtosis =
      draw_diagnostics[["draw_mean_excess_kurtosis"]],
    draw_max_abs_skewness_error =
      draw_diagnostics[["draw_max_abs_skewness_error"]],
    draw_max_abs_excess_kurtosis_error =
      draw_diagnostics[["draw_max_abs_excess_kurtosis_error"]],
    draw_max_abs_correlation_error =
      draw_diagnostics[["draw_max_abs_correlation_error"]],
    draw_max_relative_variance_error =
      draw_diagnostics[["draw_max_relative_variance_error"]],
    stringsAsFactors = FALSE
  )
  for (name in names(cell)) row[[name]] <- cell[[name]]
  row <- cbind(row, as.data.frame(p, check.names = FALSE))
  if (return_details) {
    list(
      row = row,
      sample = sample,
      fits = fits$value,
      nested = result,
      plan = plan
    )
  } else {
    row
  }
}

study2_semtests_parity <- function(cell, details, tolerance = 5e-5) {
  requireNamespace("lavaan", quietly = TRUE)
  requireNamespace("semTests", quietly = TRUE)
  plan <- details$plan
  captured <- tryCatch(study2_capture({
    syntax <- study2_syntax(cell$p)
    h1 <- lavaan::cfa(
      syntax, details$sample$data,
      group = "group", estimator = "MLM",
      std.lv = FALSE, meanstructure = FALSE
    )
    h0 <- lavaan::cfa(
      syntax, details$sample$data,
      group = "group", group.equal = "loadings",
      estimator = "MLM", std.lv = FALSE, meanstructure = FALSE
    )
    p <- semTests::pvalues_nested(
      h0, h1, method = "2000", tests = plan$input,
      A.method = "delta"
    )
    list(h1 = h1, h0 = h0, p = as.numeric(p))
  }), error = function(e) e)
  base <- data.frame(
    cell_id = cell$cell_id,
    p = cell$p,
    groups = cell$groups,
    n_group = cell$n_group,
    df = cell$df,
    distribution = cell$distribution,
    rep = 1L,
    method_id = plan$method_id,
    input = plan$input,
    magmaan = details$nested$p_value,
    semtests = NA_real_,
    abs_diff = NA_real_,
    tolerance = tolerance,
    pass = FALSE,
    warnings = "",
    error = "",
    stringsAsFactors = FALSE
  )
  if (inherits(captured, "error")) {
    base$error <- conditionMessage(captured)
    return(base)
  }
  base$semtests <- captured$value$p
  base$abs_diff <- abs(base$magmaan - base$semtests)
  base$pass <- is.finite(base$abs_diff) & base$abs_diff <= tolerance
  base$warnings <- paste(captured$warnings, collapse = " | ")
  base
}
