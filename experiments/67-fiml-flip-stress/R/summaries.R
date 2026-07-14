stress_method_rows <- function(raw) {
  do.call(rbind, lapply(stress_p_columns, function(column) {
    data.frame(raw[c(stress_design_columns, "rep")],
               method = sub("^p_", "", column), p_value = raw[[column]],
               stringsAsFactors = FALSE)
  }))
}

stress_summarize_methods <- function(method_rows, alpha = .05) {
  pieces <- split(method_rows, interaction(
    method_rows$cell_id, method_rows$method, drop = TRUE))
  out <- do.call(rbind, lapply(pieces, function(x) {
    p <- x$p_value[is.finite(x$p_value)]
    n <- length(p)
    rejected_le <- sum(p <= alpha)
    rejected_lt <- sum(p < alpha)
    ci <- stress_wilson(rejected_le, n)
    data.frame(
      x[1L, c(stress_design_columns, "method")],
      n = n, rejected_le = rejected_le, rejected_lt = rejected_lt,
      rejection_rate = if (n) rejected_le / n else NA_real_,
      rejection_rate_strict = if (n) rejected_lt / n else NA_real_,
      ci_lower = ci[[1L]], ci_upper = ci[[2L]],
      mcse = if (n) sqrt((rejected_le / n) * (1 - rejected_le / n) / n)
             else NA_real_)
  }))
  row.names(out) <- NULL
  out[order(out$cell_id, out$method), ]
}

.stress_calibration_row <- function(x, scope, level) {
  z <- x$rejection_rate[is.finite(x$rejection_rate)]
  data.frame(
    method = x$method[[1L]], scope = scope, level = level,
    cells = length(z),
    rmse_from_05 = if (length(z)) sqrt(mean((z - .05)^2)) else NA_real_,
    mad_from_05 = if (length(z)) mean(abs(z - .05)) else NA_real_,
    mean_rejection = if (length(z)) mean(z) else NA_real_,
    min_rejection = if (length(z)) min(z) else NA_real_,
    max_rejection = if (length(z)) max(z) else NA_real_,
    below_025 = if (length(z)) mean(z < .025) else NA_real_,
    acceptable_025_075 = if (length(z)) mean(z >= .025 & z <= .075)
                           else NA_real_,
    above_075 = if (length(z)) mean(z > .075) else NA_real_)
}

stress_calibration_summary <- function(method_summary) {
  factors <- c("distribution", "n_group1", "missingness", "rank")
  rows <- list(); cursor <- 0L
  for (method in unique(method_summary$method)) {
    x <- method_summary[method_summary$method == method, ]
    cursor <- cursor + 1L
    rows[[cursor]] <- .stress_calibration_row(x, "overall", "all")
    for (field in factors) {
      pieces <- split(x, x[[field]])
      for (level in names(pieces)) {
        cursor <- cursor + 1L
        rows[[cursor]] <- .stress_calibration_row(
          pieces[[level]], field, as.character(level))
      }
    }
  }
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$scope, out$level, out$rmse_from_05, out$method), ]
}

stress_availability_summary <- function(method_rows) {
  pieces <- split(method_rows, method_rows$method)
  out <- do.call(rbind, lapply(pieces, function(x) data.frame(
    method = x$method[[1L]], attempted = nrow(x),
    valid = sum(is.finite(x$p_value)),
    availability = mean(is.finite(x$p_value)))))
  row.names(out) <- NULL
  out[order(out$availability, out$method), ]
}

stress_summarize_paired <- function(raw) {
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    keep <- is.finite(x$p_flip_basic) & is.finite(x$p_flip_effective) &
      is.finite(x$p_flip_standardized)
    y <- x[keep, , drop = FALSE]
    n <- nrow(y)
    basic <- y$p_flip_basic <= .05
    effective <- y$p_flip_effective <= .05
    standardized <- y$p_flip_standardized <= .05
    decision_delta <- as.numeric(standardized) - as.numeric(effective)
    abs_delta <- abs(y$p_flip_standardized - y$p_flip_effective)
    shift <- y$mean_variance_relative_shift
    correlation <- if (n >= 3L && stats::sd(abs_delta) > 0 &&
                       stats::sd(shift, na.rm = TRUE) > 0) {
      stats::cor(abs_delta, shift, use = "complete.obs")
    } else NA_real_
    data.frame(
      x[1L, stress_design_columns], n = n,
      rejection_basic = if (n) mean(basic) else NA_real_,
      rejection_effective = if (n) mean(effective) else NA_real_,
      rejection_standardized = if (n) mean(standardized) else NA_real_,
      rescue_basic_to_effective = if (n) mean(basic != effective) else NA_real_,
      rejection_delta = if (n) mean(decision_delta) else NA_real_,
      paired_mcse = if (n > 1L) stats::sd(decision_delta) / sqrt(n) else NA_real_,
      disagreement_rate = if (n) mean(effective != standardized) else NA_real_,
      changed_to_accept = if (n) mean(effective & !standardized) else NA_real_,
      changed_to_reject = if (n) mean(!effective & standardized) else NA_real_,
      mean_abs_delta_p = if (n) mean(abs_delta) else NA_real_,
      median_abs_delta_p = if (n) stats::median(abs_delta) else NA_real_,
      median_variance_shift = if (n) stats::median(shift, na.rm = TRUE)
                              else NA_real_,
      median_max_condition = if (n) stats::median(
        y$max_variance_condition, na.rm = TRUE) else NA_real_,
      delta_shift_correlation = correlation)
  }))
  row.names(out) <- NULL
  out[order(out$cell_id), ]
}

stress_standardization_summary <- function(paired) {
  fields <- c("distribution", "n_group1", "missingness", "rank")
  rows <- list(); cursor <- 0L
  summarize <- function(x, scope, level) {
    weights <- x$n
    ok <- is.finite(weights) & weights > 0
    weighted <- function(field) {
      valid <- ok & is.finite(x[[field]])
      if (any(valid)) stats::weighted.mean(x[[field]][valid], weights[valid])
      else NA_real_
    }
    data.frame(
      scope = scope, level = level, cells = sum(ok), paired_n = sum(weights[ok]),
      effective_rejection = weighted("rejection_effective"),
      standardized_rejection = weighted("rejection_standardized"),
      disagreement_rate = weighted("disagreement_rate"),
      mean_abs_delta_p = weighted("mean_abs_delta_p"),
      median_covariance_shift = stats::median(
        x$median_variance_shift[ok], na.rm = TRUE),
      median_condition = stats::median(
        x$median_max_condition[ok], na.rm = TRUE))
  }
  cursor <- cursor + 1L
  rows[[cursor]] <- summarize(paired, "overall", "all")
  for (field in fields) {
    pieces <- split(paired, paired[[field]])
    for (level in names(pieces)) {
      cursor <- cursor + 1L
      rows[[cursor]] <- summarize(pieces[[level]], field, level)
    }
  }
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

stress_summarize_timing <- function(raw) {
  columns <- c(
    draw = "draw_seconds", missing = "missing_seconds",
    fit_h1_shared = "fit_h1_seconds", fit_h0 = "fit_h0_seconds",
    fit_pair_equivalent = "fit_seconds", flip_total = "flip_seconds",
    flip_setup = "flip_setup_seconds", flip_score = "flip_score_seconds",
    flip_standardization = "flip_standardization_seconds",
    flip_asymptotic = "flip_asymptotic_seconds", nested_fmg = "nested_seconds")
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    do.call(rbind, lapply(names(columns), function(phase) {
      values <- x[[columns[[phase]]]]
      values <- values[is.finite(values)]
      data.frame(
        x[1L, stress_design_columns], phase = phase, n = length(values),
        median_seconds = if (length(values)) stats::median(values) else NA_real_,
        mean_seconds = if (length(values)) mean(values) else NA_real_,
        p90_seconds = if (length(values)) unname(stats::quantile(
          values, .9, type = 8)) else NA_real_)
    }))
  }))
  row.names(out) <- NULL
  out[order(out$cell_id, out$phase), ]
}

stress_summarize_base_timing <- function(raw) {
  first <- raw[!duplicated(raw[c("base_id", "rep")]), ]
  pieces <- split(first, first$base_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    values <- x$replication_seconds[is.finite(x$replication_seconds)]
    data.frame(
      x[1L, c("base_id", "distribution", "n_group1", "n_group2", "n_total",
              "missingness", "missing_rate")],
      n = length(values),
      median_seconds = if (length(values)) stats::median(values) else NA_real_,
      mean_seconds = if (length(values)) mean(values) else NA_real_,
      p90_seconds = if (length(values)) unname(stats::quantile(
        values, .9, type = 8)) else NA_real_)
  }))
  row.names(out) <- NULL
  out[order(out$base_id), ]
}

stress_confirmation_candidates <- function(method_summary, paired) {
  primary <- method_summary$method %in%
    c("flip_effective", "flip_standardized", "score_peba4", "nested_peba4_ml")
  x <- method_summary[primary, ]
  x$outside_band <- x$rejection_rate < .025 | x$rejection_rate > .075
  x$wilson_excludes_05 <- x$ci_upper < .05 | x$ci_lower > .05
  p <- paired[c("cell_id", "disagreement_rate")]
  x <- merge(x, p, by = "cell_id", all.x = TRUE, sort = FALSE)
  x$standardization_disagreement <- x$disagreement_rate > .01
  x$trigger <- x$outside_band | x$wilson_excludes_05 |
    x$standardization_disagreement
  x <- x[x$trigger, ]
  x[order(x$cell_id, x$method), ]
}
