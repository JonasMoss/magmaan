sem_method_catalog <- function() {
  c(
    "Observed-sensitivity Mammen multiplier" = "p_flip_corrected",
    "Expected-sensitivity Mammen multiplier" = "p_flip_expected_mammen",
    "Expected-sensitivity Rademacher multiplier" = "p_flip_effective",
    "Naive LR/D" = "p_lrt_naive",
    "Scalar robust" = "p_lrt_mlr",
    "LR/D FMG SB" = "p_lrt_sb",
    "LR/D FMG SS" = "p_lrt_ss",
    "LR/D FMG pEBA(4)" = "p_lrt_peba4",
    "LR/D FMG all" = "p_lrt_all",
    "Observed-score FMG SB" = "p_score_corrected_sb",
    "Observed-score FMG SS" = "p_score_corrected_ss",
    "Observed-score FMG pEBA(4)" = "p_score_corrected_peba4",
    "Observed-score FMG all" = "p_score_corrected_all",
    "Legacy-score FMG SB" = "p_score_sb",
    "Legacy-score FMG SS" = "p_score_ss",
    "Legacy-score FMG pEBA(4)" = "p_score_peba4",
    "Legacy-score FMG all" = "p_score_all")
}

sem_method_rows <- function(raw) {
  catalog <- sem_method_catalog()
  catalog <- catalog[catalog %in% names(raw)]
  keys <- c(
    "cell_id", "pair_id", "model_id", "model_label", "distribution",
    "missingness", "analysis_region", "truth", "alternative_effect",
    "population_fml", "estimator", "expected_df", "n", "rep", "seed")
  pieces <- lapply(names(catalog), function(method) {
    out <- raw[keys]
    out$method <- method
    out$p_value <- raw[[catalog[[method]]]]
    out
  })
  do.call(rbind, pieces)
}

sem_null_summary <- function(method_rows) {
  x <- method_rows[method_rows$truth == "null", , drop = FALSE]
  if (!nrow(x)) return(data.frame())
  keys <- c(
    "model_id", "model_label", "distribution", "missingness",
    "analysis_region", "estimator", "expected_df", "n", "method")
  groups <- split(seq_len(nrow(x)), interaction(
    x[keys], drop = TRUE, lex.order = TRUE))
  out <- do.call(rbind, lapply(groups, function(ii) {
    z <- x[ii, , drop = FALSE]
    p <- z$p_value[is.finite(z$p_value)]
    data.frame(
      z[1L, keys, drop = FALSE], attempted = nrow(z), finite = length(p),
      rejection_le_05 = if (length(p)) mean(p <= .05) else NA_real_,
      rejection_lt_05 = if (length(p)) mean(p < .05) else NA_real_,
      stringsAsFactors = FALSE)
  }))
  row.names(out) <- NULL
  out
}

sem_power_summary <- function(method_rows) {
  power <- method_rows[method_rows$truth != "null", , drop = FALSE]
  if (!nrow(power)) return(data.frame())
  null <- method_rows[method_rows$truth == "null", , drop = FALSE]
  keys <- c(
    "model_id", "model_label", "distribution", "missingness",
    "analysis_region", "estimator", "expected_df", "n", "method")
  power_keys <- unique(power[c(keys, "truth", "alternative_effect",
                               "population_fml")])
  rows <- lapply(seq_len(nrow(power_keys)), function(index) {
    key <- power_keys[index, , drop = FALSE]
    match_key <- function(x) {
      keep <- rep(TRUE, nrow(x))
      for (name in keys) keep <- keep & x[[name]] == key[[name]][[1L]]
      keep
    }
    null_p <- null$p_value[match_key(null)]
    power_match <- match_key(power) & power$truth == key$truth[[1L]]
    power_p <- power$p_value[power_match]
    null_p <- null_p[is.finite(null_p)]
    power_p <- power_p[is.finite(power_p)]
    cutoff <- if (length(null_p)) {
      unname(stats::quantile(null_p, .05, type = 8))
    } else NA_real_
    data.frame(
      key,
      null_finite = length(null_p),
      power_finite = length(power_p),
      empirical_null_cutoff = cutoff,
      nominal_size = if (length(null_p)) mean(null_p <= .05) else NA_real_,
      raw_power = if (length(power_p)) mean(power_p <= .05) else NA_real_,
      matched_null_power = if (length(power_p) && is.finite(cutoff)) {
        mean(power_p <= cutoff)
      } else NA_real_,
      stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

sem_write_method_summaries <- function(raw, results) {
  method_rows <- sem_method_rows(raw)
  null_summary <- sem_null_summary(method_rows)
  power_summary <- sem_power_summary(method_rows)
  if (nrow(null_summary)) {
    write_csv(null_summary, file.path(results, "null_summary.csv"))
  }
  if (nrow(power_summary)) {
    write_csv(power_summary, file.path(results, "power_summary.csv"))
  }
  invisible(list(null = null_summary, power = power_summary))
}
