safe_rate <- function(x) {
  x <- x[!is.na(x)]
  if (length(x)) mean(x) else NA_real_
}

safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

max_vector_difference <- function(a, b) {
  serialized_max_abs_difference(a, b)
}

safe_quantile <- function(x, probability) {
  x <- x[is.finite(x)]
  if (length(x)) {
    as.numeric(stats::quantile(x, probability, names = FALSE, type = 8))
  } else NA_real_
}

wilson_interval <- function(successes, attempts, level = 0.95) {
  if (!attempts) return(c(NA_real_, NA_real_))
  z <- stats::qnorm(1 - (1 - level) / 2)
  p <- successes / attempts
  denominator <- 1 + z^2 / attempts
  centre <- (p + z^2 / (2 * attempts)) / denominator
  half_width <- z * sqrt(
    p * (1 - p) / attempts + z^2 / (4 * attempts^2)
  ) / denominator
  c(max(0, centre - half_width), min(1, centre + half_width))
}

pair_results <- function(raw) {
  eligible <- raw[
    !is.na(raw$pair_id) & nzchar(raw$pair_id),
    , drop = FALSE
  ]
  if (!nrow(eligible)) return(data.frame())
  groups <- split(eligible, eligible$pair_id)
  rows <- lapply(groups, function(x) {
    ordinary <- x[x$role == "ordinary", , drop = FALSE]
    psd <- x[x$role == "psd", , drop = FALSE]
    if (nrow(ordinary) != 1L || nrow(psd) != 1L) return(NULL)
    data.frame(
      task_id = ordinary$task_id,
      family = ordinary$family,
      geometry = ordinary$geometry,
      structure = ordinary$structure,
      stress_axis = ordinary$stress_axis,
      parameterization = ordinary$parameterization,
      n = ordinary$n,
      n_ratio = ordinary$n_ratio,
      rep = ordinary$rep,
      seed = ordinary$seed,
      target_population_min_eigenvalue =
        ordinary$target_population_min_eigenvalue,
      pair_id = ordinary$pair_id,
      criterion = ordinary$criterion,
      ordinary_fmin = ordinary$fmin,
      psd_fmin = psd$fmin,
      ordinary_returned = ordinary$returned,
      psd_returned = psd$returned,
      ordinary_stationary = ordinary$audit_stationary,
      psd_stationary = psd$audit_stationary,
      ordinary_admissible = ordinary$admissible,
      psd_admissible = psd$admissible,
      psd_boundary = psd$primitive_boundary,
      objective_gap = if (ordinary$returned && psd$returned) {
        psd$fmin - ordinary$fmin
      } else NA_real_,
      theta_max_abs_difference = max_vector_difference(
        ordinary$theta_values, psd$theta_values
      ),
      sigma_max_abs_difference = max_vector_difference(
        ordinary$sigma_values, psd$sigma_values
      ),
      time_ratio = if (ordinary$elapsed_ms > 0 && psd$elapsed_ms >= 0) {
        psd$elapsed_ms / ordinary$elapsed_ms
      } else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

pilot_cell_summary <- function(raw) {
  groups <- split(raw, interaction(
    raw$structure, raw$stress_axis, raw$geometry, raw$method, drop = TRUE
  ))
  rows <- lapply(groups, function(x) {
    returned <- x$returned
    stationary <- returned & !is.na(x$audit_stationary) & x$audit_stationary
    psd_accepted <- x$psd & stationary
    return_ci <- wilson_interval(sum(returned), length(returned))
    stationary_ci <- wilson_interval(sum(stationary), length(stationary))
    admissible_ci <- wilson_interval(
      sum(psd_accepted & !is.na(x$admissible) & x$admissible),
      sum(psd_accepted)
    )
    data.frame(
      structure = x$structure[[1L]],
      stress_axis = x$stress_axis[[1L]],
      geometry = x$geometry[[1L]],
      method = x$method[[1L]],
      psd = x$psd[[1L]],
      n = x$n[[1L]],
      n_ratio = x$n_ratio[[1L]],
      target_population_min_eigenvalue =
        x$target_population_min_eigenvalue[[1L]],
      attempts = nrow(x),
      returned_rate = mean(returned),
      returned_ci_low = return_ci[[1L]],
      returned_ci_high = return_ci[[2L]],
      stationary_rate = mean(stationary),
      stationary_ci_low = stationary_ci[[1L]],
      stationary_ci_high = stationary_ci[[2L]],
      admissible_rate = if (sum(psd_accepted)) {
        mean(x$admissible[psd_accepted])
      } else NA_real_,
      admissible_ci_low = admissible_ci[[1L]],
      admissible_ci_high = admissible_ci[[2L]],
      boundary_rate = if (sum(psd_accepted)) {
        mean(x$primitive_boundary[psd_accepted])
      } else NA_real_,
      median_sigma_population_error = safe_median(
        x$sigma_population_max_abs_error[stationary]
      ),
      p95_sigma_population_error = safe_quantile(
        x$sigma_population_max_abs_error[stationary], 0.95
      ),
      median_elapsed_ms = safe_median(x$elapsed_ms[returned]),
      p95_elapsed_ms = safe_quantile(x$elapsed_ms[returned], 0.95),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

pilot_pair_summary <- function(pairs) {
  if (!nrow(pairs)) return(data.frame())
  groups <- split(pairs, interaction(
    pairs$structure, pairs$stress_axis, pairs$geometry, pairs$criterion,
    drop = TRUE
  ))
  rows <- lapply(groups, function(x) {
    comparable <- x$ordinary_returned & x$psd_returned &
      !is.na(x$ordinary_stationary) & x$ordinary_stationary &
      !is.na(x$psd_stationary) & x$psd_stationary
    data.frame(
      structure = x$structure[[1L]],
      stress_axis = x$stress_axis[[1L]],
      geometry = x$geometry[[1L]],
      criterion = x$criterion[[1L]],
      n = x$n[[1L]],
      n_ratio = x$n_ratio[[1L]],
      target_population_min_eigenvalue =
        x$target_population_min_eigenvalue[[1L]],
      pairs = nrow(x),
      comparable_rate = mean(comparable),
      ordinary_admissible_rate = safe_rate(x$ordinary_admissible[comparable]),
      psd_admissible_rate = safe_rate(x$psd_admissible[comparable]),
      median_abs_objective_gap = safe_median(abs(x$objective_gap[comparable])),
      p95_abs_objective_gap = safe_quantile(
        abs(x$objective_gap[comparable]), 0.95
      ),
      median_sigma_difference = safe_median(
        x$sigma_max_abs_difference[comparable]
      ),
      p95_sigma_difference = safe_quantile(
        x$sigma_max_abs_difference[comparable], 0.95
      ),
      median_time_ratio = safe_median(x$time_ratio[comparable]),
      p95_time_ratio = safe_quantile(x$time_ratio[comparable], 0.95),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

family_summary <- function(raw) {
  groups <- split(raw, raw$family)
  rows <- lapply(groups, function(x) {
    psd <- x[x$psd & x$expected_outcome == "fit", , drop = FALSE]
    returned <- x[x$returned & x$expected_outcome == "fit", , drop = FALSE]
    objective_checked <- returned[!is.na(returned$objective_ok), , drop = FALSE]
    input_checked <- x[!is.na(x$input_unchanged), , drop = FALSE]
    data.frame(
      family = x$family[[1L]],
      fits = nrow(x),
      returned_rate = safe_rate(x$returned[x$expected_outcome == "fit"]),
      audit_stationary_rate = safe_rate(returned$audit_stationary),
      psd_admissible_rate = safe_rate(psd$admissible[psd$returned]),
      psd_boundary_rate = safe_rate(psd$primitive_boundary[psd$returned]),
      objective_checks = nrow(objective_checked),
      objective_pass_rate = safe_rate(objective_checked$objective_ok),
      input_preservation_rate = safe_rate(input_checked$input_unchanged),
      median_elapsed_ms = safe_median(returned$elapsed_ms),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

invariant_summary <- function(raw, pairs = pair_results(raw)) {
  accepted_psd <- raw[
    raw$psd & raw$expected_outcome == "fit" & raw$returned &
      !is.na(raw$audit_stationary) & raw$audit_stationary,
    , drop = FALSE
  ]
  returned <- raw[
    raw$expected_outcome == "fit" & raw$returned,
    , drop = FALSE
  ]
  input_checked <- raw[!is.na(raw$input_unchanged), , drop = FALSE]
  stage1_checked <- raw[
    !is.na(raw$stage1_fingerprint) & nzchar(raw$stage1_fingerprint),
    , drop = FALSE
  ]
  domain <- raw[raw$expected_outcome == "domain_rejection", , drop = FALSE]
  interior_pairs <- pairs[
    pairs$target_population_min_eigenvalue >= 0.20 &
      pairs$ordinary_returned & pairs$psd_returned &
      !is.na(pairs$ordinary_stationary) & pairs$ordinary_stationary &
      !is.na(pairs$psd_stationary) & pairs$psd_stationary &
      !is.na(pairs$ordinary_admissible) & pairs$ordinary_admissible,
    , drop = FALSE
  ]
  agreement_ok <- abs(interior_pairs$objective_gap) <=
    1e-8 * (1 + abs(interior_pairs$ordinary_fmin)) &
    interior_pairs$sigma_max_abs_difference <= 1e-5
  data.frame(
    invariant = c(
      "accepted_psd_is_admissible",
      "independent_objective_matches",
      "input_object_is_unchanged",
      "ml2s_stage1_is_unchanged",
      "expected_domain_rejection",
      "interior_pair_agreement"
    ),
    checked = c(
      nrow(accepted_psd), nrow(returned), nrow(input_checked),
      nrow(stage1_checked), nrow(domain), nrow(interior_pairs)
    ),
    violations = c(
      sum(is.na(accepted_psd$admissible) | !accepted_psd$admissible),
      sum(is.na(returned$objective_ok) | !returned$objective_ok),
      sum(is.na(input_checked$input_unchanged) | !input_checked$input_unchanged),
      sum(is.na(stage1_checked$stage1_unchanged) |
            !stage1_checked$stage1_unchanged),
      sum(is.na(domain$expected_outcome_ok) | !domain$expected_outcome_ok),
      sum(is.na(agreement_ok) | !agreement_ok)
    ),
    stringsAsFactors = FALSE
  )
}
