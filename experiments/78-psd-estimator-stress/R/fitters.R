scalar_or_na <- function(x) {
  if (is.null(x) || length(x) != 1L) return(NA_real_)
  as.numeric(x)
}

logical_or_na <- function(x) {
  if (is.null(x) || length(x) != 1L) return(NA)
  as.logical(x)
}

serialize_numeric <- function(x) {
  if (!length(x)) return("")
  paste(format(as.numeric(x), digits = 17L, scientific = TRUE), collapse = ";")
}

deserialize_numeric <- function(x) {
  if (is.na(x) || !nzchar(x)) return(numeric())
  as.numeric(strsplit(x, ";", fixed = TRUE)[[1L]])
}

serialized_max_abs_difference <- function(a, b) {
  a <- deserialize_numeric(a)
  b <- deserialize_numeric(b)
  if (!length(a) || length(a) != length(b)) return(NA_real_)
  max(abs(a - b))
}

component_summary <- function(fit) {
  empty <- list(
    checked = NA, covariance_psd = NA, implied_pd = NA, admissible = NA,
    theta_min = NA_real_, psi_min = NA_real_, primitive_min = NA_real_,
    primitive_boundary = NA
  )
  if (!is.list(fit) || !is.list(fit$diagnostics) ||
      !is.list(fit$diagnostics$admissibility)) return(empty)
  a <- fit$diagnostics$admissibility
  block_min <- function(blocks) {
    values <- vapply(blocks %||% list(), function(block) {
      scalar_or_na(block$min_eigenvalue)
    }, numeric(1))
    values <- values[is.finite(values)]
    if (length(values)) min(values) else NA_real_
  }
  theta_min <- block_min(a$theta)
  psi_min <- block_min(a$psi)
  primitive <- c(theta_min, psi_min)
  primitive <- primitive[is.finite(primitive)]
  primitive_min <- if (length(primitive)) min(primitive) else NA_real_
  list(
    checked = logical_or_na(a$checked),
    covariance_psd = logical_or_na(a$covariance_matrices_psd),
    implied_pd = logical_or_na(a$implied_sigma_pd),
    admissible = logical_or_na(a$admissible),
    theta_min = theta_min,
    psi_min = psi_min,
    primitive_min = primitive_min,
    primitive_boundary = if (is.finite(primitive_min)) primitive_min <= 1e-6 else NA
  )
}

implied_summary <- function(fit) {
  out <- tryCatch(magmaan_core$model_implied(fit), error = function(e) NULL)
  if (is.null(out)) {
    return(list(sigma_min = NA_real_, sigma_values = "", mu_values = ""))
  }
  eig <- unlist(lapply(out$sigma, function(sigma) {
    eigen(0.5 * (sigma + t(sigma)), symmetric = TRUE, only.values = TRUE)$values
  }))
  list(
    sigma_min = if (length(eig)) min(eig) else NA_real_,
    sigma_values = serialize_numeric(unlist(out$sigma, use.names = FALSE)),
    mu_values = serialize_numeric(unlist(out$mu, use.names = FALSE))
  )
}

two_group_equality_violation <- function(fit) {
  partable <- fit$partable
  if (!is.data.frame(partable) ||
      !all(c("label", "est") %in% names(partable))) return(NA_real_)
  estimates <- function(label) {
    as.numeric(partable$est[partable$label == label])
  }
  l2_g1 <- estimates("l2_g1")
  l3_g1 <- estimates("l3_g1")
  theta_x1 <- estimates("theta_x1")
  if (length(l2_g1) != 1L || length(l3_g1) != 1L ||
      length(theta_x1) != 2L) return(NA_real_)
  max(abs(l2_g1 + l3_g1 - 1.60), diff(range(theta_x1)))
}

base_result_row <- function(context, method, role, pair_id, criterion,
                            psd, expected_outcome) {
  data.frame(
    task_id = context$task_id,
    family = context$family,
    geometry = context$geometry,
    structure = context$structure,
    data_structure = context$data_structure,
    stress_axis = context$stress_axis,
    parameterization = context$parameterization,
    n = context$n,
    n_ratio = context$n_ratio,
    n_free = context$n_free,
    n_groups = context$n_groups,
    n_per_group = context$n_per_group,
    rep = context$rep,
    seed = context$seed,
    target_population_min_eigenvalue = context$target_min_eigenvalue,
    theta_population_min = context$theta_min,
    psi_population_min = context$psi_min,
    misspecified = context$misspecified,
    method = method,
    role = role,
    pair_id = pair_id,
    criterion = criterion,
    psd = psd,
    expected_outcome = expected_outcome,
    returned = FALSE,
    error = "",
    solver_converged = FALSE,
    audit_stationary = NA,
    audit_status = "",
    admissible = NA,
    covariance_psd = NA,
    implied_sigma_pd = NA,
    primitive_boundary = NA,
    min_theta_eigenvalue = NA_real_,
    min_psi_eigenvalue = NA_real_,
    min_primitive_eigenvalue = NA_real_,
    min_implied_sigma_eigenvalue = NA_real_,
    fmin = NA_real_,
    objective_recomputed = NA_real_,
    objective_abs_error = NA_real_,
    objective_ok = NA,
    weight_condition_target = NA_real_,
    weight_condition_actual = NA_real_,
    weight_effective_rank = NA_integer_,
    weight_dimension = NA_integer_,
    theta_values = "",
    sigma_values = "",
    mu_values = "",
    sigma_population_max_abs_error = NA_real_,
    mu_population_max_abs_error = NA_real_,
    iterations = NA_integer_,
    f_evals = NA_integer_,
    g_evals = NA_integer_,
    elapsed_ms = NA_real_,
    grad_inf_norm = NA_real_,
    equality_violation_inf = NA_real_,
    input_fingerprint_before = "",
    input_fingerprint_after = "",
    input_unchanged = NA,
    stage1_fingerprint = "",
    stage1_unchanged = NA,
    expected_outcome_ok = NA,
    stringsAsFactors = FALSE
  )
}

fit_record <- function(context, method, role, pair_id, criterion, psd,
                       fit_call, objective_call = NULL, input_object = NULL,
                       expected_outcome = "fit",
                       canonical_stage1_fingerprint = "") {
  row <- base_result_row(
    context, method, role, pair_id, criterion, psd, expected_outcome
  )
  before <- if (is.null(input_object)) "" else cache_digest(input_object)
  started <- proc.time()[["elapsed"]]
  value <- tryCatch(suppressWarnings(fit_call()), error = identity)
  elapsed <- 1000 * (proc.time()[["elapsed"]] - started)
  after <- if (is.null(input_object)) "" else cache_digest(input_object)
  row$elapsed_ms <- elapsed
  row$input_fingerprint_before <- before
  row$input_fingerprint_after <- after
  row$input_unchanged <- if (nzchar(before)) identical(before, after) else NA

  if (inherits(value, "error")) {
    row$error <- conditionMessage(value)
    row$expected_outcome_ok <- identical(expected_outcome, "domain_rejection") &&
      grepl("NonPositiveDefiniteSample|not positive definite", row$error)
    return(list(row = row, fit = NULL))
  }

  fit <- value
  row$returned <- TRUE
  row$solver_converged <- isTRUE(fit$converged)
  row$audit_stationary <- logical_or_na(fit$audit$stationary)
  row$audit_status <- as.character(fit$audit$advisory_status %||% "")
  components <- component_summary(fit)
  row$admissible <- components$admissible
  row$covariance_psd <- components$covariance_psd
  row$implied_sigma_pd <- components$implied_pd
  row$primitive_boundary <- components$primitive_boundary
  row$min_theta_eigenvalue <- components$theta_min
  row$min_psi_eigenvalue <- components$psi_min
  row$min_primitive_eigenvalue <- components$primitive_min
  implied <- implied_summary(fit)
  row$min_implied_sigma_eigenvalue <- implied$sigma_min
  row$sigma_values <- implied$sigma_values
  row$mu_values <- implied$mu_values
  row$sigma_population_max_abs_error <- serialized_max_abs_difference(
    row$sigma_values, context$population_sigma_values
  )
  row$mu_population_max_abs_error <- serialized_max_abs_difference(
    row$mu_values, context$population_mu_values
  )
  row$theta_values <- serialize_numeric(fit$theta)
  row$fmin <- scalar_or_na(fit$fmin)
  row$iterations <- as.integer(scalar_or_na(fit$iterations))
  row$f_evals <- as.integer(scalar_or_na(fit$f_evals))
  row$g_evals <- as.integer(scalar_or_na(fit$g_evals))
  row$grad_inf_norm <- scalar_or_na(fit$audit$grad_inf_norm)
  row$equality_violation_inf <- scalar_or_na(
    fit$audit$constraint_violation_inf
  )
  if (identical(context$structure, "two_group_one_factor")) {
    row$equality_violation_inf <- two_group_equality_violation(fit)
  }
  if (is.function(objective_call)) {
    recomputed <- tryCatch(objective_call(fit), error = function(e) NA_real_)
    row$objective_recomputed <- as.numeric(recomputed)
    if (is.finite(row$fmin) && is.finite(row$objective_recomputed)) {
      row$objective_abs_error <- abs(row$fmin - row$objective_recomputed)
      row$objective_ok <- row$objective_abs_error <=
        1e-8 * (1 + abs(row$fmin))
    } else {
      row$objective_ok <- FALSE
    }
  }
  if (is.list(fit$stage1)) {
    row$stage1_fingerprint <- cache_digest(fit$stage1)
    if (nzchar(canonical_stage1_fingerprint)) {
      row$stage1_unchanged <- identical(
        row$stage1_fingerprint, canonical_stage1_fingerprint
      )
    }
  }
  row$expected_outcome_ok <- identical(expected_outcome, "fit")
  list(row = row, fit = fit)
}

run_continuous_task <- function(context, data, control) {
  spec <- continuous_spec(context$structure)
  raw_blocks <- if (is.list(data) && !is.data.frame(data)) {
    lapply(data, as.matrix)
  } else {
    as.matrix(data)
  }
  stats <- magmaan_core$data_sample_stats_from_raw(raw_blocks)
  fixed_condition <- if (is.finite(context$weight_condition_target)) {
    context$weight_condition_target
  } else 100
  fixed_weight <- fixed_wls_weight(stats, fixed_condition)
  weight_eigenvalues <- unlist(lapply(fixed_weight, function(weight) {
    eigen(weight, symmetric = TRUE, only.values = TRUE)$values
  }))
  weight_tolerance <- max(weight_eigenvalues) *
    max(vapply(fixed_weight, nrow, integer(1))) * .Machine$double.eps
  fixed_weight_diagnostics <- list(
    target = fixed_condition,
    actual = max(weight_eigenvalues) / min(weight_eigenvalues),
    rank = sum(weight_eigenvalues > weight_tolerance),
    dimension = length(weight_eigenvalues)
  )
  include_mean <- length(stats$mean) && all(lengths(stats$mean) > 0L)
  gls_weight <- lapply(stats$S, normal_theory_weight,
                       include_mean = include_mean)
  definitions <- list(
    list("ml", "ordinary", "ml", "ml", FALSE,
         function() magmaan_core$fit_ml(spec, stats, optimizer = "nlopt-lbfgs", control = control),
         function(fit) ml_objective(stats, magmaan_core$model_implied(fit))),
    list("psd_ml", "psd", "ml", "ml", TRUE,
         function() frontier_fit_ml_psd(spec, stats, optimizer = "nlopt-slsqp", control = control),
         function(fit) ml_objective(stats, magmaan_core$model_implied(fit))),
    list("uls", "ordinary", "uls", "uls", FALSE,
         function() magmaan_core$fit_uls(spec, stats, optimizer = "nlopt-lbfgs", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit))),
    list("psd_uls", "psd", "uls", "uls", TRUE,
         function() frontier_fit_uls_psd(spec, stats, optimizer = "nlopt-slsqp", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit))),
    list("gls", "ordinary", "gls", "gls", FALSE,
         function() magmaan_core$fit_gls(spec, stats, optimizer = "nlopt-lbfgs", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit), gls_weight)),
    list("psd_gls", "psd", "gls", "gls", TRUE,
         function() frontier_fit_gls_psd(spec, stats, optimizer = "nlopt-slsqp", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit), gls_weight)),
    list("wls", "ordinary", "wls", "fixed_wls", FALSE,
         function() magmaan_core$fit_wls(spec, stats, W = fixed_weight, optimizer = "nlopt-lbfgs", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit), fixed_weight)),
    list("psd_wls", "psd", "wls", "fixed_wls", TRUE,
         function() frontier_fit_wls_psd(spec, stats, W = fixed_weight, optimizer = "nlopt-slsqp", control = control),
         function(fit) quadratic_objective(stats, magmaan_core$model_implied(fit), fixed_weight)),
    list("psd_fitted_gmm", "unpaired", "", "fitted_gmm", TRUE,
         function() frontier_fit_gmm_fitted_weight_psd(spec, stats, optimizer = "nlopt-slsqp", control = control),
         function(fit) {
           implied <- magmaan_core$model_implied(fit)
           weights <- lapply(implied$sigma, normal_theory_weight,
                             include_mean = include_mean)
           quadratic_objective(stats, implied, weights)
         })
  )
  if (identical(context$fit_plan, "fixed_wls")) {
    definitions <- definitions[vapply(definitions, function(d) {
      identical(d[[4L]], "fixed_wls")
    }, logical(1))]
  }
  lapply(definitions, function(d) {
    pair_id <- if (nzchar(d[[3L]])) paste(context$task_id, d[[3L]], sep = "::") else ""
    result <- fit_record(
      context, d[[1L]], d[[2L]], pair_id, d[[4L]], d[[5L]], d[[6L]], d[[7L]],
      input_object = stats
    )
    if (identical(d[[4L]], "fixed_wls")) {
      result$row$weight_condition_target <- fixed_weight_diagnostics$target
      result$row$weight_condition_actual <- fixed_weight_diagnostics$actual
      result$row$weight_effective_rank <- fixed_weight_diagnostics$rank
      result$row$weight_dimension <- fixed_weight_diagnostics$dimension
    }
    result
  })
}

run_fiml_task <- function(context, data, control) {
  spec <- continuous_spec()
  missing_data <- apply_missingness(data, context$seed)
  definitions <- list(
    list("fiml", "ordinary", FALSE,
         function() magmaan_core$fit_fiml(spec, missing_data, optimizer = "nlopt-lbfgs", control = control)),
    list("psd_fiml", "psd", TRUE,
         function() frontier_fit_fiml_psd(spec, missing_data, optimizer = "nlopt-slsqp", control = control))
  )
  lapply(definitions, function(d) {
    fit_record(
      context, d[[1L]], d[[2L]], paste(context$task_id, "fiml", sep = "::"),
      "fiml", d[[3L]], d[[4L]],
      function(fit) fiml_objective(missing_data, magmaan_core$model_implied(fit)),
      input_object = missing_data
    )
  })
}

ml2s_objective_function <- function(stage1, weight, dls_a = 0.5) {
  stats <- list(S = stage1$cov, mean = stage1$mean, nobs = stage1$n_obs)
  if (identical(weight, "nt")) {
    return(function(fit) ml_objective(stats, magmaan_core$model_implied(fit)))
  }
  weights <- magmaan:::two_stage_stage2_weight_blocks_impl(
    stage1, stage2_weight = weight, dls_a = dls_a
  )
  function(fit) quadratic_objective(
    stats, magmaan_core$model_implied(fit), weights
  )
}

run_ml2s_task <- function(context, data, control) {
  spec <- continuous_spec()
  missing_data <- apply_missingness(data, context$seed)
  nt <- fit_record(
    context, "ml2s_nt", "ordinary",
    paste(context$task_id, "ml2s_nt", sep = "::"), "ml2s_nt", FALSE,
    function() magmaan_core$fit_ml2s(
      spec, missing_data, optimizer = "nlopt-lbfgs", control = control,
      stage2_weight = "nt"
    ),
    objective_call = NULL, input_object = missing_data
  )
  if (is.null(nt$fit) || !is.list(nt$fit$stage1)) return(list(nt))
  stage1 <- nt$fit$stage1
  stage1_fingerprint <- cache_digest(stage1)
  nt$row$stage1_fingerprint <- stage1_fingerprint
  nt$row$stage1_unchanged <- TRUE
  nt_objective <- ml2s_objective_function(stage1, "nt")
  nt$row$objective_recomputed <- nt_objective(nt$fit)
  nt$row$objective_abs_error <- abs(nt$row$fmin - nt$row$objective_recomputed)
  nt$row$objective_ok <- nt$row$objective_abs_error <=
    1e-8 * (1 + abs(nt$row$fmin))

  weights <- if (identical(context$geometry, "interior")) {
    c("nt", "uls", "dwls", "adf", "dls")
  } else {
    c("nt", "adf")
  }
  results <- list(nt)
  for (weight in weights) {
    objective <- ml2s_objective_function(stage1, weight)
    pair_id <- paste(context$task_id, paste0("ml2s_", weight), sep = "::")
    ordinary <- if (identical(weight, "nt")) nt else fit_record(
      context, paste0("ml2s_", weight), "ordinary", pair_id,
      paste0("ml2s_", weight), FALSE,
      function() magmaan_core$fit_ml2s(
        spec, missing_data, optimizer = "nlopt-lbfgs", control = control,
        stage1 = stage1, stage2_weight = weight, dls_a = 0.5
      ),
      objective, input_object = stage1,
      canonical_stage1_fingerprint = stage1_fingerprint
    )
    ordinary$row$pair_id <- pair_id
    psd <- fit_record(
      context, paste0("psd_ml2s_", weight), "psd", pair_id,
      paste0("ml2s_", weight), TRUE,
      function() frontier_fit_ml2s_psd(
        spec, missing_data, optimizer = "nlopt-slsqp", control = control,
        stage1 = stage1, stage2_weight = weight, dls_a = 0.5
      ),
      objective, input_object = stage1,
      canonical_stage1_fingerprint = stage1_fingerprint
    )
    if (!identical(weight, "nt")) results[[length(results) + 1L]] <- ordinary
    results[[length(results) + 1L]] <- psd
  }
  results
}

run_ordinal_task <- function(context, data, control) {
  parameterization <- context$parameterization
  spec <- ordinal_spec(parameterization)
  ordinal_data <- ordinalize(data)
  stats <- magmaan_core$data_ordinal_stats_from_df(ordinal_data, spec)
  weights <- if (identical(context$geometry, "interior")) {
    c("uls", "dwls", "wls")
  } else {
    "uls"
  }
  results <- list()
  for (weight in weights) {
    pair_id <- paste(context$task_id, paste0("ordinal_", weight), sep = "::")
    ordinary_call <- switch(
      weight,
      uls = function() magmaan_core$fit_uls_ordinal(spec, stats, optimizer = "nlopt-lbfgs", control = control),
      dwls = function() magmaan_core$fit_dwls_ordinal(spec, stats, optimizer = "nlopt-lbfgs", control = control),
      wls = function() magmaan_core$fit_wls_ordinal(spec, stats, optimizer = "nlopt-lbfgs", control = control)
    )
    objective <- function(fit) ordinal_objective(
      fit, stats, weight, parameterization
    )
    results[[length(results) + 1L]] <- fit_record(
      context, paste0("ordinal_", weight), "ordinary", pair_id,
      paste0("ordinal_", weight), FALSE, ordinary_call, objective,
      input_object = stats
    )
    results[[length(results) + 1L]] <- fit_record(
      context, paste0("psd_ordinal_", weight), "psd", pair_id,
      paste0("ordinal_", weight), TRUE,
      function() frontier_fit_ordinal_psd(
        spec, stats, estimator = toupper(weight), optimizer = "nlopt-slsqp",
        control = control
      ),
      objective, input_object = stats
    )
  }
  if (identical(parameterization, "delta")) {
    results[[length(results) + 1L]] <- fit_record(
      context, "psd_catml", "unpaired", "", "catml", TRUE,
      function() frontier_fit_catml_psd(
        spec, stats, optimizer = "nlopt-slsqp", control = control
      ),
      function(fit) catml_objective(fit, stats), input_object = stats
    )
  }
  results
}

run_mixed_task <- function(context, data, control) {
  parameterization <- context$parameterization
  spec <- mixed_spec(parameterization)
  mixed_data <- mixed_ordinalize(data)
  stats <- magmaan_core$data_mixed_ordinal_stats_from_df(
    mixed_data, spec, ordered = c("x3", "x4")
  )
  weights <- if (identical(context$geometry, "interior")) {
    c("uls", "dwls", "wls")
  } else {
    "uls"
  }
  results <- list()
  for (weight in weights) {
    objective <- function(fit) mixed_objective(
      fit, stats, weight, parameterization
    )
    pair_id <- if (weight %in% c("dwls", "wls")) {
      paste(context$task_id, paste0("mixed_", weight), sep = "::")
    } else ""
    if (weight %in% c("dwls", "wls")) {
      ordinary_call <- if (identical(weight, "dwls")) {
        function() magmaan_core$fit_dwls_mixed_ordinal(
          spec, stats, optimizer = "nlopt-lbfgs", control = control
        )
      } else {
        function() magmaan_core$fit_wls_mixed_ordinal(
          spec, stats, optimizer = "nlopt-lbfgs", control = control
        )
      }
      results[[length(results) + 1L]] <- fit_record(
        context, paste0("mixed_", weight), "ordinary", pair_id,
        paste0("mixed_", weight), FALSE, ordinary_call, objective,
        input_object = stats
      )
    }
    results[[length(results) + 1L]] <- fit_record(
      context, paste0("psd_mixed_", weight),
      if (nzchar(pair_id)) "psd" else "unpaired", pair_id,
      paste0("mixed_", weight), TRUE,
      function() frontier_fit_mixed_ordinal_psd(
        spec, stats, estimator = toupper(weight), optimizer = "nlopt-slsqp",
        control = control
      ),
      objective, input_object = stats
    )
  }
  results
}

run_catml_nonpd_task <- function(context, data, control) {
  spec <- ordinal_spec("delta")
  stats <- magmaan_core$data_ordinal_stats_from_df(ordinalize(data), spec)
  bad <- matrix(c(
    1, 0.9, 0.9, 0,
    0.9, 1, -0.9, 0,
    0.9, -0.9, 1, 0,
    0, 0, 0, 1
  ), 4L, 4L, byrow = TRUE)
  dimnames(bad) <- dimnames(stats$R[[1L]])
  stats$R[[1L]] <- bad
  list(fit_record(
    context, "psd_catml_nonpd", "expected_rejection", "", "catml", TRUE,
    function() frontier_fit_catml_psd(
      spec, stats, optimizer = "nlopt-slsqp", control = control
    ),
    objective_call = NULL, input_object = stats,
    expected_outcome = "domain_rejection"
  ))
}

run_task <- function(task, seed_base, control) {
  context <- as.list(task)
  context$seed <- simulation_seed(seed_base, context$geometry, context$rep)
  population <- population_moments(context)
  context$population_sigma_values <- serialize_numeric(
    unlist(population$sigma, use.names = FALSE)
  )
  context$population_mu_values <- serialize_numeric(
    unlist(population$mu, use.names = FALSE)
  )
  data <- simulate_geometry_data(context, context$seed)
  results <- switch(
    context$family,
    continuous = run_continuous_task(context, data, control),
    fiml = run_fiml_task(context, data, control),
    ml2s = run_ml2s_task(context, data, control),
    ordinal_delta = run_ordinal_task(context, data, control),
    ordinal_theta = run_ordinal_task(context, data, control),
    mixed_delta = run_mixed_task(context, data, control),
    mixed_theta = run_mixed_task(context, data, control),
    catml_nonpd = run_catml_nonpd_task(context, data, control),
    stop("unknown task family: ", context$family, call. = FALSE)
  )
  do.call(rbind, lapply(results, `[[`, "row"))
}
