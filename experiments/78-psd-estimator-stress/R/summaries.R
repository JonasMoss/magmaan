safe_rate <- function(x) {
  x <- x[!is.na(x)]
  if (length(x)) mean(x) else NA_real_
}

safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

max_vector_difference <- function(a, b) {
  a <- deserialize_numeric(a)
  b <- deserialize_numeric(b)
  if (!length(a) || length(a) != length(b)) return(NA_real_)
  max(abs(a - b))
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
      parameterization = ordinary$parameterization,
      n = ordinary$n,
      rep = ordinary$rep,
      seed = ordinary$seed,
      pair_id = ordinary$pair_id,
      criterion = ordinary$criterion,
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

invariant_summary <- function(raw) {
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
  data.frame(
    invariant = c(
      "accepted_psd_is_admissible",
      "independent_objective_matches",
      "input_object_is_unchanged",
      "ml2s_stage1_is_unchanged",
      "expected_domain_rejection"
    ),
    checked = c(
      nrow(accepted_psd), nrow(returned), nrow(input_checked),
      nrow(stage1_checked), nrow(domain)
    ),
    violations = c(
      sum(is.na(accepted_psd$admissible) | !accepted_psd$admissible),
      sum(is.na(returned$objective_ok) | !returned$objective_ok),
      sum(is.na(input_checked$input_unchanged) | !input_checked$input_unchanged),
      sum(is.na(stage1_checked$stage1_unchanged) |
            !stage1_checked$stage1_unchanged),
      sum(is.na(domain$expected_outcome_ok) | !domain$expected_outcome_ok)
    ),
    stringsAsFactors = FALSE
  )
}
