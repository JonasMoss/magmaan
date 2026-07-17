study2_wilson <- function(x, n, z = 1.95996398454005) {
  if (!is.finite(n) || n <= 0L) return(c(lower = NA_real_, upper = NA_real_))
  p <- x / n
  den <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / den
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  c(lower = max(0, centre - half), upper = min(1, centre + half))
}

study2_method_rows <- function(raw) {
  methods <- study2_method_table()
  design <- c("cell_id", "p", "groups", "n_group", "n_total", "df",
              "distribution", "rep")
  pieces <- lapply(methods$method_id, function(id) {
    data.frame(
      raw[design],
      method_id = id,
      p_value = raw[[paste0("p_", id)]],
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, pieces)
  merge(out, methods, by = "method_id", all.x = TRUE, sort = FALSE)
}

study2_cell_summary <- function(method_rows) {
  keys <- interaction(
    method_rows$cell_id, method_rows$method_id, drop = TRUE)
  pieces <- split(method_rows, keys)
  out <- do.call(rbind, lapply(pieces, function(x) {
    valid <- is.finite(x$p_value)
    n <- sum(valid)
    rejected <- sum(x$p_value[valid] <= .05)
    strict <- sum(x$p_value[valid] < .05)
    ci <- study2_wilson(rejected, n)
    data.frame(
      x[1L, c("cell_id", "p", "groups", "n_group", "n_total", "df",
              "distribution", "method_id", "family", "paper_family",
              "gamma", "base", "paper_label")],
      n = n,
      rejected = rejected,
      rejection_rate = if (n) rejected / n else NA_real_,
      strict_rejection_rate = if (n) strict / n else NA_real_,
      mcse = if (n) sqrt((rejected / n) * (1 - rejected / n) / n) else
        NA_real_,
      ci_lower = ci[[1L]],
      ci_upper = ci[[2L]],
      stringsAsFactors = FALSE
    )
  }))
  row.names(out) <- NULL
  out[order(out$cell_id, out$method_id), ]
}

study2_calibration_summary <- function(cell_summary) {
  pieces <- split(cell_summary, cell_summary$method_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    ok <- is.finite(x$rejection_rate)
    rates <- x$rejection_rate[ok]
    if (!length(rates)) {
      return(data.frame(
        x[1L, c("method_id", "family", "paper_family", "gamma", "base",
                "paper_label")],
        cells = 0L,
        replications = 0L,
        pooled_rejection = NA_real_,
        rmse_percentage_points = NA_real_,
        mad_percentage_points = NA_real_,
        below_2_5_percent = NA_real_,
        acceptable_percent = NA_real_,
        above_7_5_percent = NA_real_,
        min_rejection = NA_real_,
        median_rejection = NA_real_,
        max_rejection = NA_real_,
        stringsAsFactors = FALSE
      ))
    }
    data.frame(
      x[1L, c("method_id", "family", "paper_family", "gamma", "base",
              "paper_label")],
      cells = length(rates),
      replications = sum(x$n[ok]),
      pooled_rejection = stats::weighted.mean(
        x$rejection_rate[ok], x$n[ok]),
      rmse_percentage_points = 100 * sqrt(mean((rates - .05)^2)),
      mad_percentage_points = 100 * mean(abs(rates - .05)),
      below_2_5_percent = 100 * mean(rates < .025),
      acceptable_percent = 100 * mean(rates >= .025 & rates <= .075),
      above_7_5_percent = 100 * mean(rates > .075),
      min_rejection = min(rates),
      median_rejection = stats::median(rates),
      max_rejection = max(rates),
      stringsAsFactors = FALSE
    )
  }))
  row.names(out) <- NULL
  out[order(out$rmse_percentage_points, out$mad_percentage_points), ]
}

study2_generator_summary <- function(raw) {
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    finite_median <- function(name) {
      values <- x[[name]]
      values <- values[is.finite(values)]
      if (length(values)) stats::median(values) else NA_real_
    }
    finite_mean <- function(name) {
      values <- x[[name]]
      values <- values[is.finite(values)]
      if (length(values)) mean(values) else NA_real_
    }
    data.frame(
      x[1L, c("cell_id", "p", "groups", "n_group", "n_total",
              "distribution")],
      replications = sum(is.finite(x$draw_mean_skewness)),
      mean_achieved_skewness = finite_mean("draw_mean_skewness"),
      mean_achieved_excess_kurtosis =
        finite_mean("draw_mean_excess_kurtosis"),
      median_max_abs_skewness_error =
        finite_median("draw_max_abs_skewness_error"),
      median_max_abs_excess_kurtosis_error =
        finite_median("draw_max_abs_excess_kurtosis_error"),
      median_max_abs_correlation_error =
        finite_median("draw_max_abs_correlation_error"),
      median_max_relative_variance_error =
        finite_median("draw_max_relative_variance_error"),
      stringsAsFactors = FALSE
    )
  }))
  row.names(out) <- NULL
  out
}

study2_timing_summary <- function(raw) {
  phases <- c(
    draw = "draw_seconds",
    fit_pair = "fit_seconds",
    nested_fmg_38 = "nested_seconds",
    replication_total = "replication_seconds"
  )
  pieces <- split(raw, raw$cell_id)
  out <- do.call(rbind, lapply(pieces, function(x) {
    do.call(rbind, lapply(names(phases), function(phase) {
      values <- x[[phases[[phase]]]]
      values <- values[is.finite(values)]
      data.frame(
        x[1L, c("cell_id", "p", "groups", "n_group", "n_total", "df",
                "distribution")],
        phase = phase,
        n = length(values),
        median_seconds = if (length(values)) stats::median(values) else
          NA_real_,
        p90_seconds = if (length(values)) unname(stats::quantile(
          values, .9, type = 8)) else NA_real_,
        stringsAsFactors = FALSE
      )
    }))
  }))
  row.names(out) <- NULL
  out
}
