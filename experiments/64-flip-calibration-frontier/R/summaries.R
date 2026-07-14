frontier_design_columns <- c(
  "cell_id", "pair_id", "p", "n_avg", "n_total", "n_min", "groups", "df",
  "information", "allocation", "distribution", "truth",
  "base_multiplier", "power_multiplier")

frontier_method_rows <- function(raw) {
  do.call(rbind, lapply(frontier_p_columns, function(column) {
    data.frame(raw[c(frontier_design_columns, "rep")],
               method = sub("^p_", "", column), p_value = raw[[column]],
               stringsAsFactors = FALSE)
  }))
}

frontier_summarize_methods <- function(method_rows, alpha = .05) {
  pieces <- split(method_rows, interaction(method_rows$cell_id,
                                            method_rows$method, drop = TRUE))
  out <- do.call(rbind, lapply(pieces, function(x) {
    valid <- is.finite(x$p_value)
    p <- x$p_value[valid]
    n <- length(p)
    rejected_le <- sum(p <= alpha)
    rejected_lt <- sum(p < alpha)
    ci <- frontier_wilson(rejected_le, n)
    data.frame(
      x[1L, c(frontier_design_columns, "method")],
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

frontier_summarize_paired <- function(raw) {
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    keep <- is.finite(x$p_flip_effective) & is.finite(x$p_flip_standardized)
    y <- x[keep, , drop = FALSE]
    n <- nrow(y)
    effective <- y$p_flip_effective <= .05
    standardized <- y$p_flip_standardized <= .05
    delta <- as.numeric(standardized) - as.numeric(effective)
    abs_delta <- abs(y$p_flip_standardized - y$p_flip_effective)
    shift <- y$mean_variance_relative_shift
    cor_shift <- if (n >= 3L && stats::sd(abs_delta) > 0 &&
                    stats::sd(shift, na.rm = TRUE) > 0) {
      stats::cor(abs_delta, shift, use = "complete.obs")
    } else NA_real_
    data.frame(
      x[1L, frontier_design_columns], n = n,
      rejection_effective = if (n) mean(effective) else NA_real_,
      rejection_standardized = if (n) mean(standardized) else NA_real_,
      rejection_delta = if (n) mean(delta) else NA_real_,
      paired_mcse = if (n > 1L) stats::sd(delta) / sqrt(n) else NA_real_,
      disagreement_rate = if (n) mean(effective != standardized) else NA_real_,
      changed_to_accept = if (n) mean(effective & !standardized) else NA_real_,
      changed_to_reject = if (n) mean(!effective & standardized) else NA_real_,
      mean_abs_delta_p = if (n) mean(abs_delta) else NA_real_,
      median_abs_delta_p = if (n) stats::median(abs_delta) else NA_real_,
      median_variance_shift = if (n) stats::median(shift, na.rm = TRUE)
                              else NA_real_,
      median_max_condition = if (n) stats::median(
        y$max_variance_condition, na.rm = TRUE) else NA_real_,
      delta_shift_correlation = cor_shift)
  }))
  row.names(out) <- NULL
  out[order(out$cell_id), ]
}

frontier_summarize_matched_power <- function(method_rows, alpha = .05) {
  focus <- method_rows[method_rows$truth %in% c("null", "power"), ]
  pieces <- split(focus, interaction(focus$pair_id, focus$method, drop = TRUE))
  rows <- lapply(pieces, function(x) {
    null <- x$p_value[x$truth == "null" & is.finite(x$p_value)]
    power <- x$p_value[x$truth == "power" & is.finite(x$p_value)]
    if (!length(null) || !length(power)) return(NULL)
    null_sorted <- sort(null)
    empirical_p <- (1 + findInterval(power, null_sorted)) / (length(null) + 1)
    rejected <- sum(empirical_p <= alpha)
    candidates <- sort(unique(c(null, power)))
    eligible <- candidates[(1 + findInterval(candidates, null_sorted)) /
                             (length(null) + 1) <= alpha]
    critical <- if (length(eligible)) max(eligible) else NA_real_
    ci <- frontier_wilson(rejected, length(power))
    data.frame(
      x[x$truth == "power", c("pair_id", "p", "n_avg", "n_total", "n_min",
        "groups", "df", "information", "allocation", "distribution",
        "base_multiplier", "power_multiplier", "method")][1L, ],
      n_null = length(null), n_power = length(power),
      null_critical_value = critical,
      null_rejection_rate = mean(null <= alpha),
      nominal_power = mean(power <= alpha),
      matched_power = rejected / length(power),
      ci_lower = ci[[1L]], ci_upper = ci[[2L]])
  })
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(data.frame())
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$pair_id, out$method), ]
}

frontier_summarize_timing <- function(raw) {
  columns <- c(
    generator_setup = "generator_setup_seconds", draw = "draw_seconds",
    fit = "fit_seconds", flip_total = "flip_seconds",
    flip_setup = "flip_setup_seconds", flip_score = "flip_score_seconds",
    flip_standardization = "flip_standardization_seconds",
    flip_asymptotic = "flip_asymptotic_seconds", nested_fmg = "nested_seconds",
    replication_total = "total_seconds")
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    do.call(rbind, lapply(names(columns), function(phase) {
      values <- x[[columns[[phase]]]]
      if (phase == "generator_setup") values <- values[[1L]]
      values <- values[is.finite(values)]
      data.frame(
        x[1L, frontier_design_columns], phase = phase, n = length(values),
        median_seconds = if (length(values)) stats::median(values) else NA_real_,
        mean_seconds = if (length(values)) mean(values) else NA_real_,
        p90_seconds = if (length(values)) unname(stats::quantile(
          values, .9, type = 8)) else NA_real_)
    }))
  }))
  row.names(out) <- NULL
  out[order(out$cell_id, out$phase), ]
}

frontier_calibration_summary <- function(method_summary) {
  x <- method_summary[method_summary$truth == "null", ]
  pieces <- split(x, x$method)
  out <- do.call(rbind, lapply(pieces, function(y) {
    z <- y$rejection_rate
    finite <- is.finite(z)
    z <- z[finite]
    data.frame(
      method = y$method[[1L]], cells = length(z),
      rmse_from_05 = if (length(z)) sqrt(mean((z - .05)^2)) else NA_real_,
      mad_from_05 = if (length(z)) mean(abs(z - .05)) else NA_real_,
      min_rejection = if (length(z)) min(z) else NA_real_,
      max_rejection = if (length(z)) max(z) else NA_real_,
      below_025 = if (length(z)) mean(z < .025) else NA_real_,
      acceptable_025_075 = if (length(z)) mean(z >= .025 & z <= .075)
                           else NA_real_,
      above_075 = if (length(z)) mean(z > .075) else NA_real_)
  }))
  if (is.null(out))  # power-only slice: no null cells to summarize here
    return(data.frame(method = character(), cells = integer(),
      rmse_from_05 = numeric(), mad_from_05 = numeric(),
      min_rejection = numeric(), max_rejection = numeric(),
      below_025 = numeric(), acceptable_025_075 = numeric(),
      above_075 = numeric()))
  row.names(out) <- NULL
  out[order(out$rmse_from_05), ]
}

frontier_availability_summary <- function(method_rows) {
  pieces <- split(method_rows, method_rows$method)
  out <- do.call(rbind, lapply(pieces, function(x) data.frame(
    method = x$method[[1L]], attempted = nrow(x),
    valid = sum(is.finite(x$p_value)),
    availability = mean(is.finite(x$p_value)))))
  row.names(out) <- NULL
  out[order(out$availability, out$method), ]
}
